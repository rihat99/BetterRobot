"""Object-referenced residual problems and tangent-coordinate linearization."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import copy
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal, TypeAlias
import warnings

import torch

from ..residuals.base import Residual
from .kernels import L2, RobustKernel, _group_rows
from .variables import Variable

_TensorMap: TypeAlias = dict[str, torch.Tensor]
JacobianStrategy: TypeAlias = Literal["auto", "analytic", "jacrev", "jacfwd", "finite_difference"]
_JACOBIAN_STRATEGIES = frozenset(("auto", "analytic", "jacrev", "jacfwd", "finite_difference"))
_L2_KERNEL = L2()


class AutodiffFallbackWarning(RuntimeWarning):
    """Warn that automatic linearization selected a slower fallback path."""


@dataclass(frozen=True)
class _EvaluationBundle:
    """Node-free tensors captured during one residual evaluation scope."""

    rows: torch.Tensor
    active_groups: tuple[torch.Tensor, ...]
    coefficients: tuple[torch.Tensor, ...]
    group_costs: tuple[torch.Tensor, ...]
    term_costs: tuple[torch.Tensor, ...]
    cost: torch.Tensor


def _check_strategy(strategy: JacobianStrategy, create_graph: bool) -> None:
    if strategy not in _JACOBIAN_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(_JACOBIAN_STRATEGIES)}, got {strategy!r}")
    if create_graph and strategy == "finite_difference":
        raise ValueError("finite_difference must be graph-free, got create_graph=True")


def _offsets(items: Sequence[Any], width: Callable[[Any], int]) -> Mapping[str, slice]:
    start = 0
    result: dict[str, slice] = {}
    for item in items:
        stop = start + width(item)
        result[item.name] = slice(start, stop)
        start = stop
    return MappingProxyType(result)


def _detach(mapping: Mapping[str, torch.Tensor]) -> _TensorMap:
    return {name: value.detach() for name, value in mapping.items()}


class Problem:
    """A frozen-at-first-use graph of residuals and referenced variables."""

    def __init__(
        self,
        residuals: Sequence[Residual] = (),
    ) -> None:
        self.residuals = list(residuals)
        self._frozen = False
        self._update_serial = 0
        self._ordered_variables: tuple[Variable, ...] = ()
        self.vars: tuple[Variable, ...] = ()
        self.variables: Mapping[str, Variable] = MappingProxyType({})
        self._vars_by_name: Mapping[str, Variable] = self.variables
        self._residuals_by_name: Mapping[str, Residual] = MappingProxyType({})
        self._item_variable_reads: dict[str, frozenset[str]] = {}
        self._nodes: tuple[Any, ...] = ()
        self.row_offsets: Mapping[str, slice] = MappingProxyType({})
        self.column_offsets: Mapping[str, slice] = MappingProxyType({})
        self.dim_total = 0
        self.tangent_dim_total = 0
        self.temporal_analysis = None
        self._warned_fallbacks: set[tuple[str, str]] = set()

    @property
    def frozen(self) -> bool:
        return self._frozen

    def add_residual(self, item: Residual) -> Residual:
        if self._frozen:
            raise RuntimeError("add_residual must run before the Problem is frozen")
        if not isinstance(item, Residual):
            raise TypeError(f"item must be a Residual, got {type(item).__name__}")
        self.residuals.append(item)
        return item

    @staticmethod
    def _node_variables(node: Any) -> tuple[Variable, ...]:
        values = getattr(node, "variables", ())
        if not isinstance(values, tuple) or any(not isinstance(value, Variable) for value in values):
            raise TypeError(f"node {type(node).__name__} variables must be tuple[Variable, ...]")
        return values

    def _freeze(self) -> None:  # noqa: PLR0912, PLR0915 - freezes and validates one graph transaction
        if self._frozen:
            return
        self._warned_fallbacks.clear()
        if any(not isinstance(item, Residual) for item in self.residuals):
            invalid = next(item for item in self.residuals if not isinstance(item, Residual))
            raise TypeError(f"Problem residuals must contain Residual objects, got {type(invalid).__name__}")

        residual_names: dict[str, Residual] = {}
        variables: list[Variable] = []
        variable_ids: set[int] = set()
        nodes: list[Any] = []
        node_ids: set[int] = set()
        mergeable_nodes: dict[tuple[object, ...], Any] = {}
        dependencies: dict[str, frozenset[str]] = {}
        for item in self.residuals:
            if item.name in residual_names:
                raise ValueError(f"duplicate residual name {item.name!r}")
            residual_names[item.name] = item
            direct = list(item.variables)
            for declared_node in getattr(item, "nodes", ()):
                node = declared_node
                merge_key = getattr(node, "merge_key", None)
                if merge_key is not None:
                    canonical = mergeable_nodes.get(merge_key)
                    if canonical is None:
                        mergeable_nodes[merge_key] = node
                    else:
                        share = getattr(node, "_share_with", None)
                        if not callable(share):
                            raise TypeError(f"mergeable node {type(node).__name__} must implement _share_with")
                        share(canonical)
                        node = canonical
                if id(node) not in node_ids:
                    nodes.append(node)
                    node_ids.add(id(node))
                direct.extend(self._node_variables(node))
            item_variables: list[Variable] = []
            item_ids: set[int] = set()
            for variable in direct:
                if not isinstance(variable, Variable):
                    raise TypeError(
                        f"Residual {item.name!r} variable references must be Variable objects, "
                        f"got {type(variable).__name__}"
                    )
                if id(variable) not in item_ids:
                    item_variables.append(variable)
                    item_ids.add(id(variable))
                if id(variable) not in variable_ids:
                    variables.append(variable)
                    variable_ids.add(id(variable))
            dependencies[item.name] = frozenset(variable.name for variable in item_variables if variable.trainable)

        by_name: dict[str, Variable] = {}
        for variable in variables:
            previous = by_name.get(variable.name)
            if previous is not None and previous is not variable:
                raise ValueError(f"duplicate variable name {variable.name!r}")
            by_name[variable.name] = variable
        trainable = tuple(variable for variable in variables if variable.trainable)

        self._ordered_variables = tuple(variables)
        self.vars = trainable
        self.variables = MappingProxyType(by_name)
        self._vars_by_name = self.variables
        self._residuals_by_name = MappingProxyType(residual_names)
        self._item_variable_reads = dependencies
        self._nodes = tuple(nodes)
        self.row_offsets = _offsets(self.residuals, lambda item: item.dim)
        self.column_offsets = _offsets(trainable, lambda variable: variable.free_dim)
        self.dim_total = sum(item.dim for item in self.residuals)
        self.tangent_dim_total = sum(variable.free_dim for variable in trainable)
        self._frozen = True
        self._validate_values(self._current_values())
        self._refresh_temporal_analysis()

    def _refresh_temporal_analysis(self) -> None:
        from .temporal import LinearizationReason, TemporalAnalysis, analyze_temporal_problem  # noqa: PLC0415

        if any(variable.time_axis is not None for variable in self.vars):
            self.temporal_analysis = analyze_temporal_problem(self)
        else:
            self.temporal_analysis = TemporalAnalysis(
                False,
                LinearizationReason.NO_TIME_VARIABLE,
                "no optimized variable declares a time axis",
            )

    def _current_values(self) -> _TensorMap:
        return {variable.name: variable.tensor for variable in self._ordered_variables}

    def _trainable_values(self) -> _TensorMap:
        self._freeze()
        return {variable.name: variable.tensor for variable in self.vars}

    def _validate_values(self, values: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
        expected = set(self.variables)
        if set(values) != expected:
            raise ValueError(f"values names must be {tuple(self.variables)}, got {tuple(values)}")
        batch_shape: tuple[int, ...] | None = None
        dtype: torch.dtype | None = None
        device: torch.device | None = None
        for variable in self._ordered_variables:
            value = values[variable.name]
            variable.validate_value(value)
            if variable.trainable:
                current = variable.batch_shape_of(value)
                if batch_shape is not None and current != batch_shape:
                    raise ValueError(
                        f"all trainable variables must share batch shape {batch_shape}, "
                        f"got {current} for {variable.name!r}"
                    )
                batch_shape = current if batch_shape is None else batch_shape
            if dtype is not None and (value.dtype != dtype or value.device != device):
                raise ValueError(
                    f"all variables must share dtype/device {dtype}/{device}, "
                    f"got {value.dtype}/{value.device} for {variable.name!r}"
                )
            dtype, device = (value.dtype, value.device) if dtype is None else (dtype, device)
        return batch_shape or ()

    def _invalidate_nodes(self) -> None:
        for node in self._nodes:
            invalidate = getattr(node, "_invalidate", None)
            if callable(invalidate):
                invalidate()

    @contextmanager
    def _node_evaluation(self) -> Iterator[None]:
        entered: list[tuple[Any, bool, bool]] = []
        try:
            for node in self._nodes:
                begin = getattr(node, "_begin_evaluation", None)
                end = getattr(node, "_end_evaluation", None)
                scoped = callable(begin) and callable(end)
                if scoped:
                    nested = begin()
                else:
                    nested = False
                    invalidate = getattr(node, "_invalidate", None)
                    if callable(invalidate):
                        invalidate()
                entered.append((node, scoped, nested))
            yield
        finally:
            for node, scoped, nested in reversed(entered):
                if scoped:
                    node._end_evaluation(nested)
                else:
                    invalidate = getattr(node, "_invalidate", None)
                    if callable(invalidate):
                        invalidate()

    @contextmanager
    def _assigned(self, values: Mapping[str, torch.Tensor]) -> Iterator[None]:
        originals = {name: self.variables[name].tensor for name in values}
        for name, value in values.items():
            self.variables[name].tensor = value
        self._invalidate_nodes()
        try:
            yield
        finally:
            for name, value in originals.items():
                self.variables[name].tensor = value
            self._invalidate_nodes()

    @contextmanager
    def _evaluation(self, values: Mapping[str, torch.Tensor] | None = None) -> Iterator[None]:
        if values is None:
            with self._node_evaluation():
                yield
        else:
            with self._assigned(values):
                with self._node_evaluation():
                    yield

    def update(self, values: Mapping[str, torch.Tensor]) -> None:
        self._freeze()
        if any(name not in self.variables for name in values):
            unknown = sorted(set(values) - set(self.variables))
            raise ValueError(f"update names must belong to the Problem, got unknown {unknown}")
        combined = self._current_values()
        combined.update(values)
        self._validate_values(combined)
        for name, value in values.items():
            self.variables[name].tensor = value
        self._update_serial += 1
        self._invalidate_nodes()

    def _set_trainable(self, values: Mapping[str, torch.Tensor], *, detach: bool = False) -> None:
        expected = {variable.name for variable in self.vars}
        if set(values) != expected:
            raise ValueError(f"trainable values names must be {tuple(expected)}, got {tuple(values)}")
        for variable in self.vars:
            value = values[variable.name]
            variable.tensor = value.detach() if detach else value
        self._invalidate_nodes()

    def _validate_trainable_values(self, values: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
        expected = {variable.name for variable in self.vars}
        if set(values) != expected:
            raise ValueError(f"trainable values names must be {tuple(expected)}, got {tuple(values)}")
        combined = self._current_values()
        combined.update(values)
        return self._validate_values(combined)

    def _batch_and_exemplar(self) -> tuple[tuple[int, ...], torch.Tensor]:
        self._freeze()
        if not self.residuals:
            raise ValueError("Problem must contain at least one residual")
        if not self._ordered_variables:
            raise ValueError("Problem residuals must reference at least one variable")
        values = self._current_values()
        batch_shape = self._validate_values(values)
        exemplar = self.vars[0].tensor if self.vars else self._ordered_variables[0].tensor
        return batch_shape, exemplar

    @staticmethod
    def _validate_output(
        item: Residual, output: torch.Tensor, batch_shape: tuple[int, ...], exemplar: torch.Tensor
    ) -> None:
        expected = (*batch_shape, item.dim)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected:
            actual = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
            raise ValueError(f"Residual {item.name!r} must return shape {expected}, got {actual}")
        if output.dtype != exemplar.dtype or output.device != exemplar.device:
            raise ValueError(
                f"Residual {item.name!r} must preserve working dtype/device "
                f"{exemplar.dtype}/{exemplar.device}, got {output.dtype}/{output.device}"
            )

    @staticmethod
    def _kernel(item: Residual, *, detach_tensors: bool) -> RobustKernel:
        kernel = item.kernel if item.kernel is not None else _L2_KERNEL
        if not isinstance(kernel, RobustKernel):
            raise TypeError(f"Residual {item.name!r} kernel must implement the RobustKernel protocol")
        if not detach_tensors:
            return kernel
        tensors = {
            name: value.detach()
            for name, value in getattr(kernel, "__dict__", {}).items()
            if isinstance(value, torch.Tensor)
        }
        if not tensors:
            return kernel
        detached = copy.copy(kernel)
        for name, value in tensors.items():
            object.__setattr__(detached, name, value)
        return detached

    @staticmethod
    def _active_groups(
        item: Residual,
        batch_shape: tuple[int, ...],
        exemplar: torch.Tensor,
    ) -> torch.Tensor:
        group_count = item.dim // item.group_size
        declared = item.active_groups()
        if declared is None:
            return torch.ones((*batch_shape, group_count), dtype=torch.bool, device=exemplar.device)
        expected = (*batch_shape, group_count)
        if not isinstance(declared, torch.Tensor) or tuple(declared.shape) != expected:
            actual = tuple(declared.shape) if isinstance(declared, torch.Tensor) else type(declared).__name__
            raise ValueError(f"Residual {item.name!r} active_groups must have shape {expected}, got {actual}")
        if declared.dtype != torch.bool or declared.device != exemplar.device:
            raise ValueError(
                f"Residual {item.name!r} active_groups must be bool on {exemplar.device}, "
                f"got {declared.dtype}/{declared.device}"
            )
        return declared.detach()

    @staticmethod
    def _coefficients(
        item: Residual,
        active_groups: torch.Tensor,
        batch_shape: tuple[int, ...],
        exemplar: torch.Tensor,
    ) -> torch.Tensor:
        group_count = item.dim // item.group_size
        weight = item.weight
        if isinstance(weight, torch.Tensor):
            if weight.dtype != exemplar.dtype or weight.device != exemplar.device:
                raise ValueError(
                    f"Residual {item.name!r} weight must preserve working dtype/device "
                    f"{exemplar.dtype}/{exemplar.device}, got {weight.dtype}/{weight.device}"
                )
            shape = tuple(weight.shape)
            if shape == ():
                coefficient = weight.expand(*batch_shape, group_count)
            elif shape == batch_shape:
                coefficient = weight.unsqueeze(-1).expand(*batch_shape, group_count)
            elif shape == (*batch_shape, group_count):
                coefficient = weight
            else:
                expected = ((), batch_shape, (*batch_shape, group_count))
                raise ValueError(f"Residual {item.name!r} weight shape must be one of {expected}, got {shape}")
        else:
            assert isinstance(weight, Real)
            coefficient = exemplar.new_full((*batch_shape, group_count), float(weight))

        if item.reduce == "mean":
            coefficient = coefficient / group_count
        elif item.reduce == "mean_active":
            active_count = active_groups.sum(dim=-1).clamp(min=1).detach()
            coefficient = coefficient / active_count.unsqueeze(-1)
        return coefficient

    def _evaluate_current(
        self,
        *,
        validate_runtime: bool = True,
        detach_kernel_tensors: bool = False,
    ) -> _EvaluationBundle:
        batch_shape, exemplar = self._batch_and_exemplar()
        result = exemplar.new_zeros(*batch_shape, self.dim_total)
        active_masks: list[torch.Tensor] = []
        coefficients: list[torch.Tensor] = []
        group_costs: list[torch.Tensor] = []
        term_costs: list[torch.Tensor] = []
        for item in self.residuals:
            group_count = item.dim // item.group_size
            if item.is_inactive():
                inactive = torch.zeros((*batch_shape, group_count), dtype=torch.bool, device=exemplar.device)
                zeros = exemplar.new_zeros(*batch_shape, group_count)
                active_masks.append(inactive)
                coefficients.append(zeros)
                group_costs.append(zeros)
                term_costs.append(exemplar.new_zeros(batch_shape))
                continue
            output = item.error()
            if validate_runtime:
                self._validate_output(item, output, batch_shape, exemplar)
            rows = item.row_weight.apply(output)
            result[..., self.row_offsets[item.name]] = rows
            active = self._active_groups(item, batch_shape, exemplar)
            coefficient = self._coefficients(item, active, batch_shape, exemplar)
            groups = _group_rows(rows, item.group_size)
            kernel = self._kernel(item, detach_tensors=detach_kernel_tensors)
            grouped = active.to(dtype=rows.dtype) * coefficient * kernel.rho(groups.square().sum(dim=-1))
            active_masks.append(active)
            coefficients.append(coefficient)
            group_costs.append(grouped)
            term_costs.append(grouped.sum(dim=-1))
        total = torch.stack(term_costs, dim=-1).sum(dim=-1)
        return _EvaluationBundle(
            result,
            tuple(active_masks),
            tuple(coefficients),
            tuple(group_costs),
            tuple(term_costs),
            total,
        )

    def _evaluate_at(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        validate_runtime: bool = True,
        detach_kernel_tensors: bool = False,
    ) -> _EvaluationBundle:
        with self._evaluation(values):
            return self._evaluate_current(
                validate_runtime=validate_runtime,
                detach_kernel_tensors=detach_kernel_tensors,
            )

    def _error_current(self, *, validate_runtime: bool = True) -> torch.Tensor:
        return self._evaluate_current(validate_runtime=validate_runtime).rows

    def error(self) -> torch.Tensor:
        """Return concatenated square-root-information-whitened rows."""
        self._freeze()
        with self._evaluation():
            return self._error_current()

    def _error_at(self, values: Mapping[str, torch.Tensor], *, validate_runtime: bool = True) -> torch.Tensor:
        with self._evaluation(values):
            return self._error_current(validate_runtime=validate_runtime)

    def _objective_current(self, *, validate_runtime: bool = True) -> torch.Tensor:
        return self._evaluate_current(validate_runtime=validate_runtime).cost

    def objective(self, values: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        self._freeze()
        with self._evaluation(values):
            return self._objective_current()

    def term_costs(self, values: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Return one objective contribution per residual from one evaluation."""
        self._freeze()
        with self._evaluation(values):
            evaluation = self._evaluate_current()
        return {item.name: cost for item, cost in zip(self.residuals, evaluation.term_costs, strict=True)}

    def gradient(self, *, create_graph: bool = False) -> _TensorMap:
        self._freeze()
        batch_shape, _ = self._batch_and_exemplar()
        base = self._trainable_values()
        deltas: dict[str, torch.Tensor] = {}
        active: list[torch.Tensor] = []
        for variable in self.vars:
            delta = variable.tensor.new_zeros(*batch_shape, variable.free_dim)
            if variable.free_dim:
                delta.requires_grad_(True)
                active.append(delta)
            deltas[variable.name] = delta
        candidate = {
            variable.name: variable._retract_from(base[variable.name], deltas[variable.name]) for variable in self.vars
        }
        output = self.objective(candidate)
        computed = (
            torch.autograd.grad(output.sum(), active, create_graph=create_graph, allow_unused=True)
            if active and output.requires_grad
            else ()
        )
        by_name = (
            dict(zip((variable.name for variable in self.vars if variable.free_dim), computed, strict=True))
            if computed
            else {}
        )
        anchors = [
            variable.tensor.sum() * 0.0
            for variable in self._ordered_variables
            if create_graph and variable.tensor.requires_grad
        ]
        anchor = sum(anchors[1:], anchors[0]) if anchors else None
        result: _TensorMap = {}
        for variable in self.vars:
            value = by_name.get(variable.name)
            gradient = torch.zeros_like(deltas[variable.name]) if value is None else value
            result[variable.name] = gradient + anchor if anchor is not None else gradient
        return result

    def _ad_block(
        self,
        item: Residual,
        variable: Variable,
        batch_shape: tuple[int, ...],
        strategy: Literal["jacrev", "jacfwd"],
        *,
        create_graph: bool,
    ) -> torch.Tensor:
        base = variable.tensor

        def closure(delta: torch.Tensor) -> torch.Tensor:
            candidate = variable._retract_from(base, delta)
            with self._evaluation({variable.name: candidate}):
                output = item.error()
                self._validate_output(item, output, batch_shape, base)
                return item.row_weight.apply(output)

        transform = torch.func.jacrev if strategy == "jacrev" else torch.func.jacfwd
        block = transform(closure)(base.new_zeros(*batch_shape, variable.free_dim))
        if batch_shape:
            count = 1
            for size in batch_shape:
                count *= size
            matrix = block.reshape(count, item.dim, count, variable.free_dim)
            block = matrix.diagonal(dim1=0, dim2=2).movedim(-1, 0).reshape(*batch_shape, item.dim, variable.free_dim)
        block = block.to(base)
        if not create_graph:
            return block.detach()
        anchors = [value.sum() * 0.0 for value in self._current_values().values() if value.requires_grad]
        return block + sum(anchors[1:], anchors[0]) if anchors else block

    def _fd_block(
        self,
        item: Residual,
        variable: Variable,
        batch_shape: tuple[int, ...],
        *,
        eps: float,
    ) -> torch.Tensor:
        columns: list[torch.Tensor] = []
        base = variable.tensor
        for column in range(variable.free_dim):
            delta = base.new_zeros(*batch_shape, variable.free_dim)
            delta[..., column] = eps
            with self._evaluation({variable.name: variable._retract_from(base, delta)}):
                plus_raw = item.error()
                self._validate_output(item, plus_raw, batch_shape, base)
                plus = item.row_weight.apply(plus_raw)
            with self._evaluation({variable.name: variable._retract_from(base, -delta)}):
                minus_raw = item.error()
                self._validate_output(item, minus_raw, batch_shape, base)
                minus = item.row_weight.apply(minus_raw)
            columns.append((plus - minus) / (2.0 * eps))
        if not columns:
            return base.new_empty(*batch_shape, item.dim, 0)
        return torch.stack(columns, dim=-1).detach()

    def jacobian_blocks(  # noqa: PLR0912 - validates and dispatches every supported block strategy
        self,
        *,
        strategy: JacobianStrategy = "auto",
        create_graph: bool = False,
        fd_eps: float = 1e-4,
    ) -> dict[tuple[str, str], torch.Tensor]:
        self._freeze()
        _check_strategy(strategy, create_graph)
        batch_shape, exemplar = self._batch_and_exemplar()
        result: dict[tuple[str, str], torch.Tensor] = {}
        analytic_by_name: dict[str, tuple[torch.Tensor, ...] | None] = {}
        if strategy in {"auto", "analytic"}:
            # All analytic blocks see one exact tensor assignment, so shared
            # nodes (notably RobotState) compute once for the whole pass.
            with self._evaluation():
                analytic_by_name = {item.name: item.jacobian() for item in self.residuals if not item.is_inactive()}
        for item in self.residuals:
            if item.is_inactive():
                continue
            dependencies = [variable for variable in self.vars if variable.name in self._item_variable_reads[item.name]]
            analytic = analytic_by_name.get(item.name)
            if strategy in {"auto", "analytic"}:
                if analytic is not None and len(analytic) != len(dependencies):
                    raise ValueError(
                        f"Residual {item.name!r} jacobian must return {len(dependencies)} blocks, got {len(analytic)}"
                    )
            weighted_analytic = item.row_weight.apply_jacobian(analytic) if analytic is not None else None
            warning_key = ("autodiff", item.name)
            if strategy == "auto" and analytic is None and dependencies and warning_key not in self._warned_fallbacks:
                selected_transforms = sorted(
                    {"jacrev" if item.dim <= variable.free_dim else "jacfwd" for variable in dependencies}
                )
                transform_names = " and ".join(f"torch.func.{name}" for name in selected_transforms)
                self._warned_fallbacks.add(warning_key)
                warnings.warn(
                    f"{type(item).__name__} residual {item.name!r} has no analytic jacobian(); "
                    f"strategy='auto' is using {transform_names}. Provide jacobian() or pass an explicit "
                    "Jacobian strategy to silence this warning.",
                    AutodiffFallbackWarning,
                    stacklevel=2,
                )
            for index, variable in enumerate(dependencies):
                key = (item.name, variable.name)
                if weighted_analytic is not None:
                    block = weighted_analytic[index]
                    expected = (*batch_shape, item.dim, variable.free_dim)
                    if not isinstance(block, torch.Tensor) or tuple(block.shape) != expected:
                        actual = tuple(block.shape) if isinstance(block, torch.Tensor) else type(block).__name__
                        raise ValueError(f"Analytic block {key!r} must have shape {expected}, got {actual}")
                    if block.dtype != exemplar.dtype or block.device != exemplar.device:
                        raise ValueError(
                            f"Analytic block {key!r} must preserve working dtype/device "
                            f"{exemplar.dtype}/{exemplar.device}, got {block.dtype}/{block.device}"
                        )
                    if create_graph and variable.tensor.requires_grad and not block.requires_grad:
                        raise ValueError(f"Analytic block {key!r} cannot honor create_graph=True")
                    result[key] = block if create_graph else block.detach()
                elif strategy == "analytic":
                    raise ValueError(f"Residual {item.name!r} must provide an analytic Jacobian")
                elif strategy == "finite_difference":
                    result[key] = self._fd_block(item, variable, batch_shape, eps=fd_eps)
                else:
                    selected = (
                        strategy if strategy != "auto" else ("jacrev" if item.dim <= variable.free_dim else "jacfwd")
                    )
                    result[key] = self._ad_block(
                        item,
                        variable,
                        batch_shape,
                        selected,
                        create_graph=create_graph,
                    )
        return result

    def dense_jacobian(
        self,
        *,
        strategy: JacobianStrategy = "auto",
        create_graph: bool = False,
    ) -> torch.Tensor:
        self._freeze()
        batch_shape, exemplar = self._batch_and_exemplar()
        dense = exemplar.new_zeros(*batch_shape, self.dim_total, self.tangent_dim_total)
        for (residual_name, variable_name), block in self.jacobian_blocks(
            strategy=strategy,
            create_graph=create_graph,
        ).items():
            dense[..., self.row_offsets[residual_name], self.column_offsets[variable_name]] = block
        return dense

    def _dense_jacobian_at(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        strategy: JacobianStrategy = "auto",
        create_graph: bool = False,
    ) -> torch.Tensor:
        with self._evaluation(values):
            return self.dense_jacobian(strategy=strategy, create_graph=create_graph)

    def retract(self, values: Mapping[str, torch.Tensor], steps: Mapping[str, torch.Tensor]) -> _TensorMap:
        self._freeze()
        if set(values) != {variable.name for variable in self.vars} or set(steps) != set(values):
            raise ValueError("values and steps must contain exactly one entry per trainable variable")
        return {
            variable.name: variable._retract_from(values[variable.name], steps[variable.name]) for variable in self.vars
        }

    def difference(self, x0: Mapping[str, torch.Tensor], x1: Mapping[str, torch.Tensor]) -> _TensorMap:
        self._freeze()
        if set(x0) != {variable.name for variable in self.vars} or set(x1) != set(x0):
            raise ValueError("difference inputs must contain exactly one entry per trainable variable")
        return {
            variable.name: variable._difference_from(x0[variable.name], x1[variable.name]) for variable in self.vars
        }

    def structured_normal(
        self,
        values: Mapping[str, torch.Tensor] | None = None,
        *,
        row_scale: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        create_graph: bool = False,
    ):
        from .temporal import assemble_structured_normal  # noqa: PLC0415

        self._freeze()
        current = self._trainable_values() if values is None else dict(values)
        batch_shape = self._validate_trainable_values(current)
        with self._evaluation(current):
            return assemble_structured_normal(
                self,
                batch_shape=batch_shape,
                row_scale=row_scale,
                residual=residual,
                create_graph=create_graph,
                validate_runtime=True,
            )


__all__ = ["AutodiffFallbackWarning", "JacobianStrategy", "Problem"]
