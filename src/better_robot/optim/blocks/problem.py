"""Named-block evaluation protocol, AD Jacobians, and dense v1 assembly.

Residuals and objective terms are structural/duck-typed: authoring one needs
no import from :mod:`better_robot.optim`. Residuals declare a static ``dim``
and ``reads``; optional analytic blocks use reduced (mask-eliminated) tangent
columns. Providers form a statically checked DAG and are cached only inside one
evaluation-local context.

AD fallback decision table
--------------------------

==============================  ==============================================
Condition                       Strategy
==============================  ==============================================
Residual supplies block         analytic (a raised error is never swallowed)
``residual_dim <= free_dim``     ``torch.func.jacrev``
``free_dim < residual_dim``      ``torch.func.jacfwd``
Gradient/objective              one VJP through tangent retraction
Explicit ``finite_difference``  central FD debug path only
==============================  ==============================================

Dense assembly is deliberately v1: variables define deterministic column
offsets and residual items define row offsets. A trajectory is one dense block.
Symbolic sparsity, banded solvers, and Schur elimination belong to M5; preserved
legacy semantics live in ``plan/design_notes/residual_sparsity.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import torch

from .autograd import tangent_grad
from .providers import EvaluationContext, Provider
from .variables import Values, VarSpec

Weight: TypeAlias = float | torch.Tensor
JacobianStrategy: TypeAlias = Literal["auto", "analytic", "jacrev", "jacfwd", "finite_difference"]


@runtime_checkable
class Residual(Protocol):
    """Structural least-squares residual contract."""

    name: str
    reads: tuple[str, ...]
    dim: int

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor: ...


@runtime_checkable
class ObjectiveTerm(Protocol):
    """Structural scalar objective contract for first-order phases."""

    name: str
    reads: tuple[str, ...]

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor | tuple[torch.Tensor, dict]: ...


@dataclass(frozen=True)
class ResidualItem:
    """A named residual with residual multiplier and future robust kernel.

    ``group_size`` partitions the final residual axis into contiguous robust
    groups. A scalar-row loss uses 1; point displacements commonly use 3. M2a
    records the semantics while M2b applies the kernel/IRLS weights.
    """

    name: str
    residual: Residual
    weight: Weight = 1.0
    kernel: Any | None = None
    group_size: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ResidualItem name must be non-empty")
        residual_name = getattr(self.residual, "name", None)
        if not isinstance(residual_name, str) or not residual_name:
            raise TypeError("A residual must declare a non-empty string name")
        if residual_name != self.name:
            raise ValueError(f"ResidualItem name {self.name!r} does not match residual name {residual_name!r}")
        dim = getattr(self.residual, "dim", None)
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Residual {self.name!r} must declare a positive static dim")
        if not isinstance(self.group_size, int) or self.group_size <= 0 or dim % self.group_size:
            raise ValueError(
                f"ResidualItem {self.name!r} group_size must be a positive divisor "
                f"of dim={dim}, got {self.group_size!r}"
            )
        _validate_weight_type(self.name, self.weight)


@dataclass(frozen=True)
class ObjectiveItem:
    """A named scalar objective and its linear objective weight."""

    name: str
    term: ObjectiveTerm
    weight: Weight = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ObjectiveItem name must be non-empty")
        term_name = getattr(self.term, "name", None)
        if not isinstance(term_name, str) or not term_name:
            raise TypeError("An objective term must declare a non-empty string name")
        if term_name != self.name:
            raise ValueError(f"ObjectiveItem name {self.name!r} does not match term name {term_name!r}")
        _validate_weight_type(self.name, self.weight)


def _validate_weight_type(name: str, weight: Weight) -> None:
    if not isinstance(weight, (Real, torch.Tensor)):
        raise TypeError(f"Weight for item {name!r} must be a real number or torch.Tensor")
    if isinstance(weight, torch.Tensor) and not weight.is_floating_point():
        raise TypeError(f"Tensor weight for item {name!r} must use a floating dtype")


def _is_inactive(weight: Weight) -> bool:
    # Tensor weights stay graph-visible and are never inspected on the host.
    return isinstance(weight, Real) and float(weight) == 0.0


def _broadcast_weight(weight: Weight, output: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(weight, dtype=output.dtype, device=output.device)
    while result.ndim < output.ndim:
        result = result.unsqueeze(-1)
    return result


def _validate_runtime_weight(
    name: str,
    weight: Weight,
    batch_shape: tuple[int, ...],
    exemplar: torch.Tensor,
) -> None:
    if not isinstance(weight, torch.Tensor):
        return
    if tuple(weight.shape) not in ((), batch_shape):
        raise ValueError(
            f"Tensor weight for item {name!r} must be scalar or have exact batch "
            f"shape {batch_shape}, got {tuple(weight.shape)}"
        )
    if weight.dtype != exemplar.dtype or weight.device != exemplar.device:
        raise ValueError(
            f"Tensor weight for item {name!r} must preserve working dtype/device "
            f"{exemplar.dtype}/{exemplar.device}, got {weight.dtype}/{weight.device}"
        )


def _detach_diagnostic(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {name: _detach_diagnostic(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_diagnostic(item) for item in value)
    if isinstance(value, list):
        return [_detach_diagnostic(item) for item in value]
    return value


def _cycle_path(dependencies: Mapping[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    done: set[str] = set()
    visiting: list[str] = []

    def visit(name: str) -> tuple[str, ...] | None:
        if name in done:
            return None
        if name in visiting:
            start = visiting.index(name)
            return (*visiting[start:], name)
        visiting.append(name)
        for dependency in dependencies[name]:
            found = visit(dependency)
            if found is not None:
                return found
        visiting.pop()
        done.add(name)
        return None

    for provider_name in dependencies:
        found = visit(provider_name)
        if found is not None:
            return found
    return None


class Problem:
    """A product-manifold problem with named residual/provider dependencies.

    External differentiable tensors must be enumerated in ``parameters`` so a
    future implicit solve can recover a stable tensor pytree. Arbitrary tensor
    attributes hidden inside residual objects are static configuration, not a
    differentiation contract.
    """

    def __init__(  # noqa: PLR0912, PLR0915 - construction validates the whole static graph
        self,
        *,
        vars: Sequence[VarSpec],
        residuals: Sequence[ResidualItem] = (),
        objectives: Sequence[ObjectiveItem] = (),
        providers: Sequence[Provider] = (),
        parameters: Mapping[str, torch.Tensor] | None = None,
        differentiable_parameters: Sequence[str] = (),
    ) -> None:
        self.vars = tuple(vars)
        self.residuals = tuple(residuals)
        self.objectives = tuple(objectives)
        self.providers = tuple(providers)
        self.parameters = MappingProxyType(dict(parameters or {}))
        differentiable_parameters = tuple(differentiable_parameters)
        if any(not isinstance(name, str) or not name for name in differentiable_parameters):
            raise TypeError("differentiable_parameters must contain non-empty string names")
        self.parameter_gradients = frozenset(differentiable_parameters)

        if not self.vars:
            raise ValueError("Problem requires at least one VarSpec")
        if any(not isinstance(spec, VarSpec) for spec in self.vars):
            raise TypeError("Problem.vars must contain only VarSpec values")
        if any(not isinstance(item, ResidualItem) for item in self.residuals):
            raise TypeError("Problem.residuals must contain only ResidualItem values")
        if any(not isinstance(item, ObjectiveItem) for item in self.objectives):
            raise TypeError("Problem.objectives must contain only ObjectiveItem values")

        self._vars_by_name = self._unique_by_name(self.vars, "variable")
        self._residuals_by_name = self._unique_by_name(self.residuals, "residual")
        self._objectives_by_name = self._unique_by_name(self.objectives, "objective")
        duplicate_items = set(self._residuals_by_name) & set(self._objectives_by_name)
        if duplicate_items:
            raise ValueError(f"residual/objective names collide: {sorted(duplicate_items)}")
        if any(not isinstance(name, str) or not name for name in self.parameters):
            raise TypeError("Problem parameter names must be non-empty strings")
        if any(not isinstance(value, torch.Tensor) for value in self.parameters.values()):
            raise TypeError("Problem parameters must be a mapping of names to tensors")
        unknown_parameter_gradients = self.parameter_gradients - set(self.parameters)
        if unknown_parameter_gradients:
            raise ValueError(
                "differentiable_parameters names must exist in Problem.parameters; "
                f"unknown {sorted(unknown_parameter_gradients)}"
            )

        base_names = set(self._vars_by_name) | set(self.parameters)
        duplicate_base = set(self._vars_by_name) & set(self.parameters)
        if duplicate_base:
            raise ValueError(f"variable/parameter names collide: {sorted(duplicate_base)}")

        providers_by_name = self._unique_by_name(self.providers, "provider")
        providers_by_output: dict[str, Provider] = {}
        for provider in self.providers:
            if not isinstance(provider, Provider):
                raise TypeError(f"Provider {provider!r} does not implement the Provider protocol")
            if not isinstance(provider.inputs, tuple) or any(
                not isinstance(name, str) or not name for name in provider.inputs
            ):
                raise TypeError(f"Provider {provider.name!r} inputs must be tuple[str, ...]")
            if not isinstance(provider.outputs, tuple) or any(
                not isinstance(name, str) or not name for name in provider.outputs
            ):
                raise TypeError(f"Provider {provider.name!r} outputs must be tuple[str, ...]")
            if not provider.outputs:
                raise ValueError(f"Provider {provider.name!r} must declare at least one output")
            for output in provider.outputs:
                if output in base_names or output in providers_by_output:
                    raise ValueError(f"provider output {output!r} collides with another context name")
                providers_by_output[output] = provider

        known_names = base_names | set(providers_by_output)
        dependencies: dict[str, tuple[str, ...]] = {}
        for provider in self.providers:
            unknown = set(provider.inputs) - known_names
            if unknown:
                raise ValueError(f"Provider {provider.name!r} declares unknown inputs {sorted(unknown)}")
            dependencies[provider.name] = tuple(
                providers_by_output[name].name for name in provider.inputs if name in providers_by_output
            )
        cycle = _cycle_path(dependencies)
        if cycle is not None:
            raise ValueError(
                f"provider dependency cycle: {' -> '.join(cycle)}. "
                "Providers must form a DAG over declared inputs/outputs."
            )

        # Stable topological order for inspection/debugging. Lazy evaluation
        # still starts only from names an active residual actually reads.
        ordered: list[Provider] = []
        remaining = set(providers_by_name)
        while remaining:
            ready = [
                name
                for name in providers_by_name
                if name in remaining and all(dep not in remaining for dep in dependencies[name])
            ]
            if not ready:  # defensive: the explicit cycle check above owns the message
                raise RuntimeError("provider DAG topological sort failed")
            for name in ready:
                ordered.append(providers_by_name[name])
                remaining.remove(name)
        self.provider_order = tuple(ordered)
        self._providers_by_output = MappingProxyType(providers_by_output)

        self._provider_var_dependencies: dict[str, frozenset[str]] = {}

        def output_vars(name: str) -> frozenset[str]:
            if name in self._vars_by_name:
                return frozenset((name,))
            if name in self.parameters:
                return frozenset()
            if name in self._provider_var_dependencies:
                return self._provider_var_dependencies[name]
            provider = providers_by_output[name]
            result = frozenset().union(*(output_vars(item) for item in provider.inputs))
            for output in provider.outputs:
                self._provider_var_dependencies[output] = result
            return result

        for output in providers_by_output:
            output_vars(output)

        self._item_variable_reads: dict[str, frozenset[str]] = {}
        for item in (*self.residuals, *self.objectives):
            subject = item.residual if isinstance(item, ResidualItem) else item.term
            reads = getattr(subject, "reads", None)
            if not isinstance(reads, tuple) or any(not isinstance(name, str) for name in reads):
                raise TypeError(f"Item {item.name!r} must declare reads as tuple[str, ...]")
            unknown = set(reads) - known_names
            if unknown:
                raise ValueError(f"Item {item.name!r} declares unknown reads {sorted(unknown)}")
            self._item_variable_reads[item.name] = frozenset().union(*(output_vars(name) for name in reads))

        self.row_offsets: Mapping[str, slice] = MappingProxyType(self._make_row_offsets())
        self.column_offsets: Mapping[str, slice] = MappingProxyType(self._make_column_offsets())
        self.dim_total = sum(item.residual.dim for item in self.residuals)
        self.tangent_dim_total = sum(spec.free_dim for spec in self.vars)

    @staticmethod
    def _unique_by_name(items: Sequence[Any], label: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in items:
            name = getattr(item, "name", None)
            if not isinstance(name, str) or not name:
                raise ValueError(f"Every {label} must have a non-empty string name")
            if name in result:
                raise ValueError(f"duplicate {label} name {name!r}")
            result[name] = item
        return result

    def _make_row_offsets(self) -> dict[str, slice]:
        offsets: dict[str, slice] = {}
        start = 0
        for item in self.residuals:
            stop = start + item.residual.dim
            offsets[item.name] = slice(start, stop)
            start = stop
        return offsets

    def _make_column_offsets(self) -> dict[str, slice]:
        offsets: dict[str, slice] = {}
        start = 0
        for spec in self.vars:
            stop = start + spec.free_dim
            offsets[spec.name] = slice(start, stop)
            start = stop
        return offsets

    def _validate_values(self, values: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
        if set(values) != set(self._vars_by_name):
            raise ValueError(f"Problem values names must be {tuple(self._vars_by_name)}, got {tuple(values)}")
        batch_shape: tuple[int, ...] | None = None
        dtype: torch.dtype | None = None
        device: torch.device | None = None
        for spec in self.vars:
            value = values[spec.name]
            spec.validate_value(value)
            # Validation above establishes the event suffix; avoid repeating
            # its finite/manifold/bounds host checks just to read batch axes.
            current = tuple(value.shape[: value.ndim - len(spec.shape)])
            if batch_shape is None:
                batch_shape = current
            elif current != batch_shape:
                raise ValueError(
                    f"All Values must share batch shape {batch_shape}; VarSpec {spec.name!r} has {current}"
                )
            if dtype is None:
                dtype = value.dtype
                device = value.device
            elif value.dtype != dtype or value.device != device:
                raise ValueError(
                    f"All Values must share dtype/device {dtype}/{device}; "
                    f"VarSpec {spec.name!r} has {value.dtype}/{value.device}"
                )
            if spec.scale is not None and (spec.scale.dtype != value.dtype or spec.scale.device != value.device):
                raise ValueError(
                    f"VarSpec {spec.name!r} scale and value must share dtype/device; "
                    f"got {spec.scale.dtype}/{spec.scale.device} and "
                    f"{value.dtype}/{value.device}"
                )
        return batch_shape or ()

    def _make_context(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        parameters: Mapping[str, torch.Tensor] | None = None,
    ) -> EvaluationContext:
        seeded = dict(values)
        seeded.update(self.parameters if parameters is None else parameters)
        return EvaluationContext(
            seeded,
            self._providers_by_output,
            {name: spec.free_indices for name, spec in self._vars_by_name.items()},
        )

    @staticmethod
    def _weights_for(
        item: ResidualItem | ObjectiveItem,
        weights: Mapping[str, Weight] | None,
    ) -> Weight:
        return item.weight if weights is None or item.name not in weights else weights[item.name]

    def _validate_weights(self, weights: Mapping[str, Weight] | None) -> None:
        if weights is None:
            return
        if any(not isinstance(name, str) or not name for name in weights):
            raise TypeError("weight override names must be non-empty strings")
        known = set(self._residuals_by_name) | set(self._objectives_by_name)
        unknown = set(weights) - known
        if unknown:
            raise ValueError(f"weight overrides contain unknown item names {sorted(unknown)}")
        for name, weight in weights.items():
            _validate_weight_type(name, weight)

    @staticmethod
    def _validate_residual_output(
        item: ResidualItem,
        output: torch.Tensor,
        batch_shape: tuple[int, ...],
        exemplar: torch.Tensor,
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
                _validate_runtime_weight(item.name, weight, batch_shape, exemplar)
            if _is_inactive(weight):
                continue
            output = item.residual(ctx.restrict(item.residual.reads))
            if validate_runtime:
                self._validate_residual_output(item, output, batch_shape, exemplar)
            result[..., self.row_offsets[item.name]] = output * _broadcast_weight(weight, output)
        return result

    def _residual_prevalidated(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        batch_shape: tuple[int, ...],
        weights: Mapping[str, Weight] | None = None,
    ) -> torch.Tensor:
        """Evaluate residuals after public values/weights validation has run."""
        return self._residual_with_context(
            values,
            batch_shape,
            self._make_context(values),
            weights,
            validate_runtime=False,
        )

    def residual(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        weights: Mapping[str, Weight] | None = None,
    ) -> torch.Tensor:
        """Return the deterministic dense residual ``(B..., dim_total)``."""
        self._validate_weights(weights)
        batch_shape = self._validate_values(values)
        ctx = self._make_context(values)
        return self._residual_with_context(values, batch_shape, ctx, weights)

    def _objective_with_context(
        self,
        values: Mapping[str, torch.Tensor],
        batch_shape: tuple[int, ...],
        ctx: EvaluationContext,
        weights: Mapping[str, Weight] | None,
    ) -> tuple[torch.Tensor, dict[str, dict]]:
        residual = self._residual_with_context(values, batch_shape, ctx, weights)
        total = 0.5 * residual.square().sum(dim=-1)
        diagnostics: dict[str, dict] = {}
        for item in self.objectives:
            weight = self._weights_for(item, weights)
            exemplar = values[self.vars[0].name]
            _validate_runtime_weight(item.name, weight, batch_shape, exemplar)
            if _is_inactive(weight):
                continue
            raw = item.term(ctx.restrict(item.term.reads))
            value, info = raw if isinstance(raw, tuple) else (raw, {})
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != batch_shape:
                actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                raise ValueError(
                    f"Objective term {item.name!r} must return scalar batch shape {batch_shape}, got {actual}"
                )
            if value.dtype != exemplar.dtype or value.device != exemplar.device:
                raise ValueError(
                    f"Objective term {item.name!r} must preserve working dtype/device "
                    f"{exemplar.dtype}/{exemplar.device}, got {value.dtype}/{value.device}"
                )
            total = total + value * _broadcast_weight(weight, value)
            if not isinstance(info, Mapping):
                raise TypeError(
                    f"Objective term {item.name!r} diagnostics must be a mapping, got {type(info).__name__}"
                )
            diagnostics[item.name] = _detach_diagnostic(info)
        return total, diagnostics

    def objective(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        weights: Mapping[str, Weight] | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, dict]]:
        """Evaluate the per-batch scalar objective without host-synced logging."""
        self._validate_weights(weights)
        batch_shape = self._validate_values(values)
        result = self._objective_with_context(
            values,
            batch_shape,
            self._make_context(values),
            weights,
        )
        return result if return_diagnostics else result[0]

    def gradient(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        weights: Mapping[str, Weight] | None = None,
        create_graph: bool = False,
    ) -> Values:
        """Return one VJP per variable in reduced tangent coordinates."""
        self._validate_weights(weights)
        self._validate_values(values)

        def closure(perturbed: Values) -> torch.Tensor:
            batch_shape = next(spec.batch_shape(perturbed[spec.name]) for spec in self.vars)
            return self._objective_with_context(
                perturbed,
                batch_shape,
                self._make_context(perturbed),
                weights,
            )[0]

        return tangent_grad(
            closure,
            self.vars,
            values,
            create_graph=create_graph,
            graph_inputs=tuple(self.differentiable_external_parameters.values()),
        )

    def retract(
        self,
        values: Mapping[str, torch.Tensor],
        steps: Mapping[str, torch.Tensor],
    ) -> Values:
        """Apply one reduced tangent step to every block."""
        self._validate_values(values)
        if set(steps) != set(self._vars_by_name):
            raise ValueError("Problem steps must contain exactly one tensor per variable")
        return {spec.name: spec.retract(values[spec.name], steps[spec.name]) for spec in self.vars}

    def _retract_prevalidated(
        self,
        values: Mapping[str, torch.Tensor],
        steps: Mapping[str, torch.Tensor],
        *,
        batch_shape: tuple[int, ...],
    ) -> Values:
        """Retract already-validated solver values and reduced tangent steps."""
        return {
            spec.name: spec._retract_prevalidated(
                values[spec.name],
                steps[spec.name],
                batch_shape=batch_shape,
            )
            for spec in self.vars
        }

    def _difference_prevalidated(
        self,
        x0: Mapping[str, torch.Tensor],
        x1: Mapping[str, torch.Tensor],
        *,
        batch_shape: tuple[int, ...],
    ) -> Values:
        """Return per-variable full tangents between prevalidated solver values."""
        return {
            spec.name: spec._difference_prevalidated(
                x0[spec.name],
                x1[spec.name],
                batch_shape=batch_shape,
            )
            for spec in self.vars
        }

    def _ad_block(
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
            perturbed[spec.name] = spec._retract_prevalidated(
                base,
                delta,
                batch_shape=batch_shape,
            )
            ctx = self._make_context(perturbed, parameters=parameters)
            output = item.residual(ctx.restrict(item.residual.reads))
            self._validate_residual_output(item, output, batch_shape, base)
            return output * _broadcast_weight(weight, output)

        zero = base.new_zeros(*batch_shape, spec.free_dim)
        transform = torch.func.jacrev if strategy == "jacrev" else torch.func.jacfwd
        # Torch 2.13 forward AD can promote an fp32 tangent to fp64 when a
        # residual contains Python floating literals even though the primal
        # output remains fp32. The block boundary preserves the working dtype.
        transformed = transform(closure)(zero)
        if batch_shape:
            count = 1
            for size in batch_shape:
                count *= size
            # Whole-batch AD produces output-batch × input-batch axes. The
            # evaluation contract makes those elements independent, so retain
            # the block diagonal. This avoids guessing whether an external
            # tensor with a matching leading size is shared or batched.
            transformed = transformed.reshape(
                count,
                item.residual.dim,
                count,
                spec.free_dim,
            )
            block = transformed.diagonal(dim1=0, dim2=2).movedim(-1, 0)
            block = block.reshape(*batch_shape, item.residual.dim, spec.free_dim)
        else:
            block = transformed
        block = block.to(dtype=base.dtype, device=base.device)
        if create_graph:
            # Constant Jacobians (notably jacrev of affine residuals) may be
            # numerically correct but disconnected from every graph input.
            # Preserve a usable higher-order contract with a zero-valued edge
            # to every graph-carrying value and explicit external parameter.
            anchors = [value.sum() * 0.0 for value in (*values.values(), *parameters.values()) if value.requires_grad]
            if anchors:
                block = block + sum(anchors[1:], anchors[0])
            return block
        return block.detach()

    def _fd_block(
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
        columns: list[torch.Tensor] = []
        for column in range(spec.free_dim):
            delta = values[spec.name].new_zeros(*batch_shape, spec.free_dim)
            delta[..., column] = eps
            plus = dict(values)
            minus = dict(values)
            plus[spec.name] = spec._retract_prevalidated(
                values[spec.name],
                delta,
                batch_shape=batch_shape,
            )
            minus[spec.name] = spec._retract_prevalidated(
                values[spec.name],
                -delta,
                batch_shape=batch_shape,
            )
            ctx_plus = self._make_context(plus, parameters=parameters)
            ctx_minus = self._make_context(minus, parameters=parameters)
            rp = item.residual(ctx_plus.restrict(item.residual.reads))
            rm = item.residual(ctx_minus.restrict(item.residual.reads))
            self._validate_residual_output(item, rp, batch_shape, values[spec.name])
            self._validate_residual_output(item, rm, batch_shape, values[spec.name])
            columns.append((rp - rm) / (2.0 * eps))
        if not columns:
            return values[spec.name].new_empty(*batch_shape, item.residual.dim, 0)
        block = torch.stack(columns, dim=-1)
        # Finite differences are an explicit graph-free debugging strategy,
        # including when primals or tensor weights carry autograd history.
        return (block * _broadcast_weight(weight, block)).detach()

    def jacobian_blocks(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        weights: Mapping[str, Weight] | None = None,
        strategy: JacobianStrategy = "auto",
        create_graph: bool = False,
        fd_eps: float = 1e-4,
    ) -> dict[tuple[str, str], torch.Tensor]:
        """Return structurally present reduced blocks keyed ``(residual, variable)``."""
        if strategy not in {"auto", "analytic", "jacrev", "jacfwd", "finite_difference"}:
            raise ValueError(f"unknown Jacobian strategy {strategy!r}")
        if create_graph and strategy == "finite_difference":
            raise ValueError("finite_difference is a graph-free debug strategy")
        self._validate_weights(weights)
        batch_shape = self._validate_values(values)
        return self._jacobian_blocks_impl(
            dict(values),
            dict(self.parameters),
            batch_shape,
            weights,
            strategy,
            create_graph=create_graph,
            fd_eps=fd_eps,
        )

    def _jacobian_blocks_impl(  # noqa: PLR0912 - one loop dispatches all block strategies
        self,
        values: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor],
        batch_shape: tuple[int, ...],
        weights: Mapping[str, Weight] | None,
        strategy: JacobianStrategy,
        *,
        create_graph: bool,
        fd_eps: float,
    ) -> dict[tuple[str, str], torch.Tensor]:
        ctx = self._make_context(values, parameters=parameters)
        result: dict[tuple[str, str], torch.Tensor] = {}
        for item in self.residuals:
            weight = self._weights_for(item, weights)
            _validate_runtime_weight(
                item.name,
                weight,
                batch_shape,
                values[self.vars[0].name],
            )
            if _is_inactive(weight):
                continue
            analytic: Mapping[str, torch.Tensor] = {}
            analytic_fn = getattr(item.residual, "jacobian_blocks", None)
            if strategy in {"auto", "analytic"} and analytic_fn is not None:
                analytic = analytic_fn(ctx.restrict(item.residual.reads))
                unknown = set(analytic) - self._item_variable_reads[item.name]
                if unknown:
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
                    if create_graph:
                        graph_inputs = (*values.values(), *parameters.values())
                        if any(value.requires_grad for value in graph_inputs) and not block.requires_grad:
                            raise ValueError(
                                f"Analytic block {key!r} cannot honor create_graph=True; "
                                "it is detached from graph-carrying context inputs. "
                                "Return a differentiable block or omit the analytic block "
                                "and use an AD strategy."
                            )
                    weighted = block * _broadcast_weight(weight, block)
                    result[key] = weighted if create_graph else weighted.detach()
                    continue
                if strategy == "analytic":
                    raise ValueError(f"Residual {item.name!r} has no analytic block for {spec.name!r}")
                selected: JacobianStrategy = strategy
                if strategy == "auto":
                    selected = "jacrev" if item.residual.dim <= spec.free_dim else "jacfwd"
                if selected == "finite_difference":
                    result[key] = self._fd_block(
                        item,
                        spec,
                        values,
                        batch_shape,
                        parameters,
                        weight,
                        eps=fd_eps,
                    )
                else:
                    result[key] = self._ad_block(
                        item,
                        spec,
                        values,
                        batch_shape,
                        parameters,
                        selected,
                        weight,
                        create_graph=create_graph,
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
        """Preallocate and fill dense ``(B..., dim_total, tangent_total_free)`` J."""
        batch_shape = self._validate_values(values)
        exemplar = values[self.vars[0].name]
        dense = exemplar.new_zeros(*batch_shape, self.dim_total, self.tangent_dim_total)
        for (residual_name, variable_name), block in self.jacobian_blocks(
            values,
            weights=weights,
            strategy=strategy,
            create_graph=create_graph,
        ).items():
            dense[
                ...,
                self.row_offsets[residual_name],
                self.column_offsets[variable_name],
            ] = block
        return dense

    def _dense_jacobian_prevalidated(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        batch_shape: tuple[int, ...],
        weights: Mapping[str, Weight] | None = None,
        strategy: JacobianStrategy = "auto",
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Assemble dense J after public values/weights/strategy validation."""
        exemplar = values[self.vars[0].name]
        dense = exemplar.new_zeros(*batch_shape, self.dim_total, self.tangent_dim_total)
        blocks = self._jacobian_blocks_impl(
            values,
            self.parameters,
            batch_shape,
            weights,
            strategy,
            create_graph=create_graph,
            fd_eps=1e-4,
        )
        for (residual_name, variable_name), block in blocks.items():
            dense[
                ...,
                self.row_offsets[residual_name],
                self.column_offsets[variable_name],
            ] = block
        return dense

    def normal_matrix(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        weights: Mapping[str, Weight] | None = None,
        strategy: JacobianStrategy = "auto",
    ) -> torch.Tensor:
        """Return dense ``JᵀJ`` for M2a correctness tests and M2b hand-off."""
        jacobian = self.dense_jacobian(values, weights=weights, strategy=strategy)
        return jacobian.mT @ jacobian

    @property
    def external_parameters(self) -> Mapping[str, torch.Tensor]:
        """Stable, explicitly enumerated external tensor parameter pytree."""
        return self.parameters

    @property
    def differentiable_external_parameters(self) -> Mapping[str, torch.Tensor]:
        """The explicitly declared future implicit-gradient parameter pytree."""
        return MappingProxyType(
            {name: self.parameters[name] for name in self.parameters if name in self.parameter_gradients}
        )

    def require_least_squares(self, method: str = "Gauss-Newton/Levenberg-Marquardt") -> None:
        """Reject scalar terms at a second-order least-squares entry boundary."""
        if self.objectives:
            names = [item.name for item in self.objectives]
            raise ValueError(
                f"problem contains scalar objective term(s) {names} — "
                f"{method} minimize sums of squared residual vectors and cannot consume "
                "scalar terms. Run these terms in a first-order phase (Adam/LBFGS), "
                "or reformulate them as residual vectors."
            )
