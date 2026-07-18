"""Named-block residual evaluation and dense/structured linearization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import torch

from . import kernels as _kernels
from .autograd import _tangent_value_and_grad
from .kernels import L2, RobustKernel, _broadcast_weight, _group_rows
from .manifolds import Bounds, Euclidean, Manifold, RobotConfig, SE3Manifold, SO3Manifold
from .providers import EvaluationContext, Provider, RobotStateProvider
from .temporal import StructuredNormal, analyze_temporal_problem, assemble_structured_normal
from .variables import Values, VarSpec

Weight: TypeAlias = float | torch.Tensor
JacobianStrategy: TypeAlias = Literal["auto", "analytic", "jacrev", "jacfwd", "finite_difference"]
_JACOBIAN_STRATEGIES = frozenset(("auto", "analytic", "jacrev", "jacfwd", "finite_difference"))
_L2_KERNEL = L2()


@runtime_checkable
class Residual(Protocol):
    name: str
    reads: tuple[str, ...]
    dim: int

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor: ...


@dataclass(frozen=True)
class ResidualItem:
    name: str
    residual: Residual
    weight: Weight = 1.0
    kernel: RobustKernel | None = None
    group_size: int = 1

    def __post_init__(self) -> None:
        residual_name = getattr(self.residual, "name", None)
        dim = getattr(self.residual, "dim", None)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"ResidualItem name must be a non-empty string, got {self.name!r}")
        if not isinstance(residual_name, str) or not residual_name:
            raise TypeError("A residual must declare a non-empty string name")
        if residual_name != self.name:
            raise ValueError(f"ResidualItem name {self.name!r} does not match residual name {residual_name!r}")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Residual {self.name!r} must declare a positive static dim, got {dim!r}")
        if not isinstance(self.group_size, int) or self.group_size <= 0 or dim % self.group_size:
            raise ValueError(
                f"ResidualItem {self.name!r} group_size must be a positive divisor of dim={dim}, got {self.group_size!r}"
            )
        if self.kernel is not None and not isinstance(self.kernel, RobustKernel):
            raise TypeError(
                f"ResidualItem {self.name!r} kernel must implement rho(squared_norm) and weight(squared_norm)"
            )
        _kernels._validate_weight_type(self.name, self.weight)


@dataclass(frozen=True)
class _CallableResidual:
    fn: Callable[[Mapping[str, Any]], torch.Tensor]
    name: str
    reads: tuple[str, ...]
    dim: int

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return self.fn(ctx)

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self.fn, attribute)


def _named(items: Sequence[Any], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        name = getattr(item, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(f"Every {label} must have a non-empty string name")
        if name in result:
            raise ValueError(f"duplicate {label} name {name!r}")
        result[name] = item
    return result


def _offsets(items: Sequence[Any], width: Callable[[Any], int]) -> Mapping[str, slice]:
    start, result = 0, {}
    for item in items:
        stop = start + width(item)
        result[item.name], start = slice(start, stop), stop
    return MappingProxyType(result)


def _check_strategy(strategy: JacobianStrategy, create_graph: bool) -> None:
    if strategy not in _JACOBIAN_STRATEGIES:
        raise ValueError(f"unknown Jacobian strategy {strategy!r}")
    if create_graph and strategy == "finite_difference":
        raise ValueError("finite_difference is a graph-free debug strategy")


class Problem:
    def __init__(  # noqa: PLR0912, PLR0915
        self,
        *,
        vars: Sequence[VarSpec] = (),
        residuals: Sequence[ResidualItem] = (),
        providers: Sequence[Provider] = (),
        parameters: Mapping[str, torch.Tensor] | None = None,
        differentiable_parameters: Sequence[str] = (),
    ) -> None:
        self.vars, self.residuals, self.providers = tuple(vars), tuple(residuals), tuple(providers)
        self.parameters = MappingProxyType(dict(parameters or {}))
        differentiable_parameters = tuple(differentiable_parameters)
        self.parameter_gradients = frozenset(differentiable_parameters)
        if any(not isinstance(name, str) or not name for name in differentiable_parameters):
            raise TypeError("differentiable_parameters must contain non-empty string names")
        if any(not isinstance(spec, VarSpec) for spec in self.vars):
            raise TypeError("Problem.vars must contain only VarSpec values")
        if any(not isinstance(item, ResidualItem) for item in self.residuals):
            raise TypeError("Problem.residuals must contain only ResidualItem values")

        # A single RobotConfig variable can supply the conventional ``data`` context lazily.
        if len(self.vars) == 1 and isinstance(self.vars[0].manifold, RobotConfig):
            robot = self.vars[0].manifold
            consumers = [item for item in self.residuals if "data" in (getattr(item.residual, "reads", None) or ())]
            for item in consumers:
                model = getattr(item.residual, "model", None)
                if model is not None and model is not robot.model:
                    raise ValueError(
                        f"Residual {item.name!r} model must match RobotConfig.model for automatic robot state"
                    )
            outputs = {
                output
                for provider in self.providers
                for output in getattr(provider, "outputs", ())
                if isinstance(output, str)
            }
            if consumers and "data" not in outputs:
                self.providers = (*self.providers, RobotStateProvider(robot.model, var=self.vars[0].name))

        self._vars_by_name = _named(self.vars, "variable")
        self._residuals_by_name = _named(self.residuals, "residual")
        _named(self.providers, "provider")
        if any(not isinstance(name, str) or not name for name in self.parameters):
            raise TypeError("Problem parameter names must be non-empty strings")
        if any(not isinstance(value, torch.Tensor) for value in self.parameters.values()):
            raise TypeError("Problem parameters must be a mapping of names to tensors")
        unknown_gradients = self.parameter_gradients - set(self.parameters)
        if unknown_gradients:
            raise ValueError(
                f"differentiable_parameters names must exist in Problem.parameters; unknown {sorted(unknown_gradients)}"
            )

        variable_names, parameter_names = set(self._vars_by_name), set(self.parameters)
        if duplicate := variable_names & parameter_names:
            raise ValueError(f"variable/parameter names collide: {sorted(duplicate)}")
        base_names = variable_names | parameter_names
        providers_by_output: dict[str, Provider] = {}
        for provider in self.providers:
            if not isinstance(provider, Provider):
                raise TypeError(f"Provider {provider!r} does not implement the Provider protocol")
            for label in ("reads", "outputs"):
                names = getattr(provider, label)
                if not isinstance(names, tuple) or any(not isinstance(name, str) or not name for name in names):
                    raise TypeError(f"Provider {provider.name!r} {label} must be tuple[str, ...]")
            if not provider.outputs:
                raise ValueError(f"Provider {provider.name!r} must declare at least one output")
            for output in provider.outputs:
                if output in base_names or output in providers_by_output:
                    raise ValueError(f"provider output {output!r} collides with another context name")
                providers_by_output[output] = provider

        known_names = base_names | set(providers_by_output)
        for provider in self.providers:
            if unknown := set(provider.reads) - known_names:
                raise ValueError(f"Provider {provider.name!r} declares unknown reads {sorted(unknown)}")
        self._providers_by_output = MappingProxyType(providers_by_output)

        dependencies: dict[str, frozenset[str]] = {}
        resolving: set[str] = set()

        def variable_dependencies(name: str) -> frozenset[str]:
            if name in variable_names:
                return frozenset((name,))
            if name in parameter_names:
                return frozenset()
            if name in dependencies:
                return dependencies[name]
            provider = providers_by_output[name]
            if provider.name in resolving:
                raise ValueError(f"provider dependency cycle involving {provider.name!r}")
            resolving.add(provider.name)
            try:
                result = frozenset().union(*(variable_dependencies(item) for item in provider.reads))
            finally:
                resolving.remove(provider.name)
            dependencies.update(dict.fromkeys(provider.outputs, result))
            return result

        for output in providers_by_output:
            variable_dependencies(output)
        self._item_variable_reads: dict[str, frozenset[str]] = {}
        for item in self.residuals:
            reads = getattr(item.residual, "reads", None)
            if not isinstance(reads, tuple) or any(not isinstance(name, str) for name in reads):
                raise TypeError(f"Item {item.name!r} must declare reads as tuple[str, ...]")
            if unknown := set(reads) - known_names:
                raise ValueError(f"Item {item.name!r} declares unknown reads {sorted(unknown)}")
            self._item_variable_reads[item.name] = frozenset().union(*(variable_dependencies(name) for name in reads))

        self.row_offsets = _offsets(self.residuals, lambda item: item.residual.dim)
        self.column_offsets = _offsets(self.vars, lambda spec: spec.free_dim)
        self.dim_total = sum(item.residual.dim for item in self.residuals)
        self.tangent_dim_total = sum(spec.free_dim for spec in self.vars)
        self.temporal_analysis = analyze_temporal_problem(self)

    def _rebuild(
        self, *, vars: Sequence[VarSpec] | None = None, residuals: Sequence[ResidualItem] | None = None
    ) -> None:
        self.__init__(
            vars=self.vars if vars is None else vars,
            residuals=self.residuals if residuals is None else residuals,
            providers=self.providers,
            parameters=self.parameters,
            differentiable_parameters=tuple(self.parameter_gradients),
        )

    def add_variable(
        self,
        name: str,
        *,
        shape: tuple[int, ...] | None = None,
        manifold: Manifold = Euclidean(),
        bounds: Bounds | None = None,
        mask: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        time_axis: int | None = None,
    ) -> VarSpec:
        if shape is None:
            if isinstance(manifold, RobotConfig):
                shape = (manifold.model.nq,)
            elif isinstance(manifold, SO3Manifold):
                shape = (4,)
            elif isinstance(manifold, SE3Manifold):
                shape = (7,)
            else:
                raise ValueError("shape is required unless the manifold has a canonical state shape")
        spec = VarSpec(name, shape, manifold=manifold, bounds=bounds, mask=mask, scale=scale, time_axis=time_axis)
        self._rebuild(vars=(*self.vars, spec))
        return spec

    def add_residual(
        self,
        residual: Callable[[Mapping[str, Any]], torch.Tensor],
        *,
        weight: Weight = 1.0,
        kernel: RobustKernel | None = None,
        name: str | None = None,
        dim: int | None = None,
    ) -> ResidualItem:
        item_name = name or getattr(residual, "name", None) or getattr(residual, "__name__", None)
        residual_dim = dim if dim is not None else getattr(residual, "dim", None)
        reads = getattr(residual, "reads", None)
        if reads is None and len(self.vars) == 1:
            reads = (self.vars[0].name,)
        if not isinstance(reads, tuple) or any(not isinstance(value, str) or not value for value in reads):
            raise ValueError(
                f"Residual {item_name!r} must declare reads when the problem does not have exactly one variable"
            )
        item = ResidualItem(item_name, _CallableResidual(residual, item_name, reads, residual_dim), weight, kernel)
        self._rebuild(residuals=(*self.residuals, item))
        return item

    def _validate_values(self, values: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
        if set(values) != set(self._vars_by_name):
            raise ValueError(f"Problem values names must be {tuple(self._vars_by_name)}, got {tuple(values)}")
        batch_shape, dtype, device = None, None, None
        for spec in self.vars:
            value = values[spec.name]
            spec.validate_value(value)
            current = tuple(value.shape[: value.ndim - len(spec.shape)])
            if batch_shape is not None and current != batch_shape:
                raise ValueError(
                    f"All Values must share batch shape {batch_shape}; VarSpec {spec.name!r} has {current}"
                )
            batch_shape = current if batch_shape is None else batch_shape
            if dtype is not None and (value.dtype != dtype or value.device != device):
                raise ValueError(
                    f"All Values must share dtype/device {dtype}/{device}; "
                    f"VarSpec {spec.name!r} has {value.dtype}/{value.device}"
                )
            dtype, device = (value.dtype, value.device) if dtype is None else (dtype, device)
            if spec.scale is not None and (spec.scale.dtype != value.dtype or spec.scale.device != value.device):
                raise ValueError(
                    f"VarSpec {spec.name!r} scale and value must share dtype/device; "
                    f"got {spec.scale.dtype}/{spec.scale.device} and {value.dtype}/{value.device}"
                )
        return batch_shape or ()

    def _make_context(
        self, values: Mapping[str, torch.Tensor], *, parameters: Mapping[str, torch.Tensor] | None = None
    ) -> EvaluationContext:
        seeded = {**values, **(self.parameters if parameters is None else parameters)}
        free = {name: spec.free_indices for name, spec in self._vars_by_name.items()}
        temporal = {
            name: spec.temporal_free_indices
            for name, spec in self._vars_by_name.items()
            if spec.time_axis is not None and spec.temporal_mask_is_separable
        }
        return EvaluationContext(seeded, self._providers_by_output, free, temporal)

    @staticmethod
    def _weights_for(item: ResidualItem, weights: Mapping[str, Weight] | None) -> Weight:
        return item.weight if weights is None or item.name not in weights else weights[item.name]

    def _validate_weights(self, weights: Mapping[str, Weight] | None) -> None:
        if weights is None:
            return
        if any(not isinstance(name, str) or not name for name in weights):
            raise TypeError("weight override names must be non-empty strings")
        if unknown := set(weights) - set(self._residuals_by_name):
            raise ValueError(f"weight overrides contain unknown item names {sorted(unknown)}")
        for name, weight in weights.items():
            _kernels._validate_weight_type(name, weight)

    def _prepare(self, values: Mapping[str, torch.Tensor], weights: Mapping[str, Weight] | None) -> tuple[int, ...]:
        self._validate_weights(weights)
        return self._validate_values(values)

    @staticmethod
    def _validate_residual_output(
        item: ResidualItem, output: torch.Tensor, batch_shape: tuple[int, ...], exemplar: torch.Tensor
    ) -> None:
        expected = (*batch_shape, item.residual.dim)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected:
            actual = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
            raise ValueError(f"Residual {item.name!r} must return shape {expected}, got {actual}")
        if output.dtype != exemplar.dtype or output.device != exemplar.device:
            raise ValueError(
                f"Residual {item.name!r} must preserve working dtype/device "
                f"{exemplar.dtype}/{exemplar.device}, got {output.dtype}/{output.device}"
            )

    def _residual_with_context(
        self,
        values: Mapping[str, torch.Tensor],
        batch_shape: tuple[int, ...],
        ctx: EvaluationContext,
        weights: Mapping[str, Weight] | None,
        *,
        validate_runtime: bool = True,
    ) -> torch.Tensor:
        exemplar = values[self.vars[0].name]
        result = exemplar.new_zeros(*batch_shape, self.dim_total)
        for item in self.residuals:
            weight = self._weights_for(item, weights)
            if validate_runtime:
                _kernels._validate_runtime_weight(item.name, weight, batch_shape, exemplar)
            if _kernels._is_inactive(weight):
                continue
            output = item.residual(ctx)
            if validate_runtime:
                self._validate_residual_output(item, output, batch_shape, exemplar)
            result[..., self.row_offsets[item.name]] = output * _broadcast_weight(weight, output)
        return result

    def residual(
        self, values: Mapping[str, torch.Tensor], *, weights: Mapping[str, Weight] | None = None
    ) -> torch.Tensor:
        batch_shape = self._prepare(values, weights)
        return self._residual_with_context(values, batch_shape, self._make_context(values), weights)

    def _objective_with_context(
        self,
        values: Mapping[str, torch.Tensor],
        batch_shape: tuple[int, ...],
        ctx: EvaluationContext,
        weights: Mapping[str, Weight] | None,
        *,
        validate_runtime: bool = True,
    ) -> torch.Tensor:
        residual = self._residual_with_context(values, batch_shape, ctx, weights, validate_runtime=validate_runtime)
        cost = values[self.vars[0].name].new_zeros(batch_shape)
        for item in self.residuals:
            if _kernels._is_inactive(self._weights_for(item, weights)):
                continue
            groups = _group_rows(residual[..., self.row_offsets[item.name]], item.group_size)
            kernel = item.kernel if item.kernel is not None else _L2_KERNEL
            cost = cost + kernel.rho(groups.square().sum(dim=-1)).sum(dim=-1)
        return cost

    def objective(
        self, values: Mapping[str, torch.Tensor], *, weights: Mapping[str, Weight] | None = None
    ) -> torch.Tensor:
        batch_shape = self._prepare(values, weights)
        return self._objective_with_context(values, batch_shape, self._make_context(values), weights)

    def gradient(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        weights: Mapping[str, Weight] | None = None,
        create_graph: bool = False,
    ) -> Values:
        batch_shape = self._prepare(values, weights)
        return self._objective_gradient(values, batch_shape=batch_shape, weights=weights, create_graph=create_graph)[1]

    def _objective_gradient(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        batch_shape: tuple[int, ...],
        weights: Mapping[str, Weight] | None = None,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, Values]:
        def closure(perturbed: Values) -> torch.Tensor:
            return self._objective_with_context(perturbed, batch_shape, self._make_context(perturbed), weights)

        return _tangent_value_and_grad(
            closure,
            self.vars,
            values,
            batch_shape=batch_shape,
            create_graph=create_graph,
            graph_inputs=tuple(self.differentiable_external_parameters.values()),
        )

    def structured_normal(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        weights: Mapping[str, Weight] | None = None,
        row_scale: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        create_graph: bool = False,
    ) -> StructuredNormal:
        if not isinstance(create_graph, bool):
            raise TypeError("create_graph must be a static bool")
        batch_shape = self._prepare(values, weights)
        return assemble_structured_normal(
            self,
            dict(values),
            batch_shape=batch_shape,
            weights=weights,
            row_scale=row_scale,
            residual=residual,
            create_graph=create_graph,
            validate_runtime=True,
        )

    def retract(self, values: Mapping[str, torch.Tensor], steps: Mapping[str, torch.Tensor]) -> Values:
        self._validate_values(values)
        if set(steps) != set(self._vars_by_name):
            raise ValueError("Problem steps must contain exactly one tensor per variable")
        return {spec.name: spec.retract(values[spec.name], steps[spec.name]) for spec in self.vars}

    def difference(self, x0: Mapping[str, torch.Tensor], x1: Mapping[str, torch.Tensor]) -> Values:
        self._validate_values(x0)
        self._validate_values(x1)
        return {spec.name: spec.difference(x0[spec.name], x1[spec.name]) for spec in self.vars}

    def _ad_block(  # noqa: PLR0913
        self,
        item: ResidualItem,
        spec: VarSpec,
        values: Mapping[str, torch.Tensor],
        batch_shape: tuple[int, ...],
        parameters: Mapping[str, torch.Tensor],
        strategy: Literal["jacrev", "jacfwd"],
        weight: Weight,
        *,
        create_graph: bool,
    ) -> torch.Tensor:
        base = values[spec.name]

        def closure(delta: torch.Tensor) -> torch.Tensor:
            perturbed = dict(values)
            perturbed[spec.name] = spec.retract(base, delta)
            output = item.residual(self._make_context(perturbed, parameters=parameters))
            self._validate_residual_output(item, output, batch_shape, base)
            return output * _broadcast_weight(weight, output)

        transformed = (torch.func.jacrev if strategy == "jacrev" else torch.func.jacfwd)(closure)(
            base.new_zeros(*batch_shape, spec.free_dim)
        )
        if batch_shape:
            count = 1
            for size in batch_shape:
                count *= size
            matrix = transformed.reshape(count, item.residual.dim, count, spec.free_dim)
            transformed = (
                matrix.diagonal(dim1=0, dim2=2).movedim(-1, 0).reshape(*batch_shape, item.residual.dim, spec.free_dim)
            )
        block = transformed.to(base)
        if not create_graph:
            return block.detach()
        anchors = [value.sum() * 0.0 for value in (*values.values(), *parameters.values()) if value.requires_grad]
        return block + sum(anchors[1:], anchors[0]) if anchors else block

    def _fd_block(  # noqa: PLR0913
        self,
        item: ResidualItem,
        spec: VarSpec,
        values: Mapping[str, torch.Tensor],
        batch_shape: tuple[int, ...],
        parameters: Mapping[str, torch.Tensor],
        weight: Weight,
        *,
        eps: float,
    ) -> torch.Tensor:
        columns = []
        for column in range(spec.free_dim):
            delta = values[spec.name].new_zeros(*batch_shape, spec.free_dim)
            delta[..., column] = eps
            plus, minus = dict(values), dict(values)
            plus[spec.name] = spec.retract(values[spec.name], delta)
            minus[spec.name] = spec.retract(values[spec.name], -delta)
            rp = item.residual(self._make_context(plus, parameters=parameters))
            rm = item.residual(self._make_context(minus, parameters=parameters))
            self._validate_residual_output(item, rp, batch_shape, values[spec.name])
            self._validate_residual_output(item, rm, batch_shape, values[spec.name])
            columns.append((rp - rm) / (2.0 * eps))
        if not columns:
            return values[spec.name].new_empty(*batch_shape, item.residual.dim, 0)
        return (torch.stack(columns, dim=-1) * _broadcast_weight(weight, torch.stack(columns, dim=-1))).detach()

    def jacobian_blocks(  # noqa: PLR0912
        self,
        values: Mapping[str, torch.Tensor],
        *,
        weights: Mapping[str, Weight] | None = None,
        strategy: JacobianStrategy = "auto",
        create_graph: bool = False,
        fd_eps: float = 1e-4,
    ) -> dict[tuple[str, str], torch.Tensor]:
        _check_strategy(strategy, create_graph)
        batch_shape = self._prepare(values, weights)
        parameters, ctx = self.parameters, self._make_context(values)
        result: dict[tuple[str, str], torch.Tensor] = {}
        for item in self.residuals:
            weight = self._weights_for(item, weights)
            _kernels._validate_runtime_weight(item.name, weight, batch_shape, values[self.vars[0].name])
            if _kernels._is_inactive(weight):
                continue
            analytic: Mapping[str, torch.Tensor] = {}
            analytic_fn = getattr(item.residual, "jacobian_blocks", None)
            if strategy in {"auto", "analytic"} and analytic_fn is not None:
                analytic = analytic_fn(ctx)
                if unknown := set(analytic) - self._item_variable_reads[item.name]:
                    raise ValueError(
                        f"Residual {item.name!r} returned analytic blocks for unread variables {sorted(unknown)}"
                    )
            for spec in self.vars:
                if spec.name not in self._item_variable_reads[item.name] or spec.free_dim == 0:
                    continue
                key = (item.name, spec.name)
                if spec.name in analytic:
                    block = analytic[spec.name]
                    expected = (*batch_shape, item.residual.dim, spec.free_dim)
                    if not isinstance(block, torch.Tensor) or tuple(block.shape) != expected:
                        actual = tuple(block.shape) if isinstance(block, torch.Tensor) else type(block).__name__
                        raise ValueError(f"Analytic block {key!r} must have reduced shape {expected}, got {actual}")
                    exemplar = values[spec.name]
                    if block.dtype != exemplar.dtype or block.device != exemplar.device:
                        raise ValueError(
                            f"Analytic block {key!r} must preserve working dtype/device "
                            f"{exemplar.dtype}/{exemplar.device}, got {block.dtype}/{block.device}"
                        )
                    graph_inputs = (*values.values(), *parameters.values())
                    if create_graph and any(value.requires_grad for value in graph_inputs) and not block.requires_grad:
                        raise ValueError(
                            f"Analytic block {key!r} cannot honor create_graph=True; it is detached from graph-carrying "
                            "context inputs. Return a differentiable block or omit the analytic block and use an AD strategy."
                        )
                    weighted = block * _broadcast_weight(weight, block)
                    result[key] = weighted if create_graph else weighted.detach()
                elif strategy == "analytic":
                    raise ValueError(f"Residual {item.name!r} has no analytic block for {spec.name!r}")
                elif strategy == "finite_difference":
                    result[key] = self._fd_block(item, spec, values, batch_shape, parameters, weight, eps=fd_eps)
                else:
                    selected = (
                        strategy
                        if strategy != "auto"
                        else ("jacrev" if item.residual.dim <= spec.free_dim else "jacfwd")
                    )
                    result[key] = self._ad_block(
                        item, spec, values, batch_shape, parameters, selected, weight, create_graph=create_graph
                    )
        return result

    def dense_jacobian(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        weights: Mapping[str, Weight] | None = None,
        strategy: JacobianStrategy = "auto",
        create_graph: bool = False,
    ) -> torch.Tensor:
        blocks = self.jacobian_blocks(values, weights=weights, strategy=strategy, create_graph=create_graph)
        exemplar = values[self.vars[0].name]
        batch_shape = tuple(exemplar.shape[: exemplar.ndim - len(self.vars[0].shape)])
        dense = exemplar.new_zeros(*batch_shape, self.dim_total, self.tangent_dim_total)
        for (residual_name, variable_name), block in blocks.items():
            dense[..., self.row_offsets[residual_name], self.column_offsets[variable_name]] = block
        return dense

    @property
    def external_parameters(self) -> Mapping[str, torch.Tensor]:
        return self.parameters

    @property
    def differentiable_external_parameters(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(
            {name: self.parameters[name] for name in self.parameters if name in self.parameter_gradients}
        )
