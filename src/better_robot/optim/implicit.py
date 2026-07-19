"""Exact first-order implicit gradients for detached optimizer solutions.

Backward recomputes the undamped robust Hessian in the terminal manifold
chart.  Invalid terminal states, unstable active sets, nonsmooth points, and
singular adjoint systems fail explicitly instead of returning a wrong gradient.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import math
from typing import Any, Literal, Protocol, TypeAlias

import torch
from torch.autograd.function import once_differentiable

from ._solver_common import _state_coordinates
from .kernels import Huber, L2, RobustKernel, _group_rows
from .problem import Problem
from .variables import RobotVariable, SE3Variable, SO3Variable

_ForwardLinearization: TypeAlias = Literal["dense", "banded"]
_TensorValues: TypeAlias = dict[str, torch.Tensor]

_CONVERGED = 1
_STALLED_AT_BOUNDS = 2
_OPTIMALITY_TOLERANCE = 1e-5
_ACTIVE_SET_TOLERANCE = 1e-7
_STRICT_COMPLEMENTARITY_TOLERANCE = 1e-8
_NONSMOOTH_TOLERANCE = 1e-7
_LINEAR_SOLVE_ATOL = 1e-10
_LINEAR_SOLVE_RTOL = 1e-5
_STATUS_NAMES = {0: "running", 1: "converged", 2: "stalled_at_bounds", 3: "maxiter", 4: "failed"}


class ImplicitDifferentiationError(RuntimeError):
    """The requested implicit backward has no contract-valid gradient."""

    def __init__(
        self,
        message: str,
        *,
        invalid_indices: tuple[tuple[int, ...], ...] = (),
        statuses: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.invalid_indices = invalid_indices
        self.statuses = statuses


@dataclass(frozen=True)
class ImplicitDiffConfig:
    """Controls dense-backward size and structured-forward opt-in."""

    max_dense_tangent_dim: int = 512
    allow_banded_dense_backward: bool = False

    def __post_init__(self) -> None:
        size = self.max_dense_tangent_dim
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("max_dense_tangent_dim must be a positive int")
        if not isinstance(self.allow_banded_dense_backward, bool):
            raise TypeError("allow_banded_dense_backward must be a bool")


class _ImplicitTerminalState(Protocol):
    """Terminal LM/GN tensors required by implicit differentiation."""

    status: torch.Tensor
    implicit_valid: torch.Tensor
    active_mask: torch.Tensor
    gradient: torch.Tensor
    projected_grad_norm: torch.Tensor
    bound_state_index: torch.Tensor
    bound_lower: torch.Tensor
    bound_upper: torch.Tensor
    bounded_mask: torch.Tensor


@dataclass(frozen=True)
class _StateSnapshot:
    status: torch.Tensor
    implicit_valid: torch.Tensor
    active_mask: torch.Tensor
    gradient: torch.Tensor
    projected_grad_norm: torch.Tensor
    bound_state_index: torch.Tensor
    bound_lower: torch.Tensor
    bound_upper: torch.Tensor
    bounded_mask: torch.Tensor


@dataclass(frozen=True)
class _Payload:
    problem: Problem
    value_names: tuple[str, ...]
    static_names: tuple[str, ...]
    batch_shape: tuple[int, ...]
    config: ImplicitDiffConfig
    state: _StateSnapshot


def _snapshot(state: _ImplicitTerminalState) -> _StateSnapshot:
    return _StateSnapshot(*(getattr(state, name).detach().clone() for name in _StateSnapshot.__dataclass_fields__))


def _validate_state(state: _StateSnapshot, batch: tuple[int, ...], tangent_dim: int, exemplar: torch.Tensor) -> None:
    for name in ("status", "implicit_valid", "projected_grad_norm"):
        if tuple(getattr(state, name).shape) != batch:
            raise ValueError(f"implicit state {name} must have batch shape {batch}")
    for name in ("active_mask", "gradient"):
        if tuple(getattr(state, name).shape) != (*batch, tangent_dim):
            raise ValueError(f"implicit state {name} must have shape {(*batch, tangent_dim)}")
    for name in ("bound_state_index", "bound_lower", "bound_upper", "bounded_mask"):
        if tuple(getattr(state, name).shape) != (tangent_dim,):
            raise ValueError(f"implicit state {name} must have shape {(tangent_dim,)}")
    for name in _StateSnapshot.__dataclass_fields__:
        if getattr(state, name).device != exemplar.device:
            raise ValueError(f"implicit state {name} must be on terminal value device {exemplar.device}")
    if state.gradient.dtype != exemplar.dtype:
        raise ValueError("implicit state gradient must preserve terminal value dtype")


def _validate_route(problem: Problem, route: _ForwardLinearization, config: ImplicitDiffConfig) -> None:
    if route not in {"dense", "banded"}:
        raise ValueError(f"unknown forward_linearization {route!r}; expected 'dense' or 'banded'")
    if route == "banded" and not config.allow_banded_dense_backward:
        raise ValueError(
            "a banded forward uses a dense implicit backward; set "
            "ImplicitDiffConfig(allow_banded_dense_backward=True) for this small-problem oracle"
        )
    if problem.tangent_dim_total > config.max_dense_tangent_dim:
        raise ValueError(
            f"dense implicit Hessian dimension {problem.tangent_dim_total}; "
            f"configured cap is {config.max_dense_tangent_dim}"
        )


def _batch_indices(mask: torch.Tensor, batch: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not batch:
        return ((),) if bool(mask) else ()
    return tuple(tuple(int(value) for value in row) for row in torch.nonzero(mask).detach().cpu())


def _status_names(status: torch.Tensor, invalid: torch.Tensor) -> tuple[str, ...]:
    values = status.reshape(-1)[invalid.reshape(-1)].detach().cpu().tolist()
    return tuple(_STATUS_NAMES.get(int(value), f"unknown({int(value)})") for value in values)


def _eligibility(values: _TensorValues, payload: _Payload) -> tuple[torch.Tensor, torch.Tensor]:
    state = payload.state
    stalled = state.status == _STALLED_AT_BOUNDS
    terminal = ((state.status == _CONVERGED) & state.implicit_valid) | stalled
    finite = torch.isfinite(state.gradient).all(dim=-1) & torch.isfinite(state.projected_grad_norm)
    finite &= state.projected_grad_norm <= _OPTIMALITY_TOLERANCE
    for value in values.values():
        finite &= torch.isfinite(value).reshape(*payload.batch_shape, -1).all(dim=-1)

    coordinate = _state_coordinates(values, payload.problem, state.bound_state_index)
    bounded = state.bounded_mask
    at_lower = coordinate <= state.bound_lower + _ACTIVE_SET_TOLERANCE
    at_upper = coordinate >= state.bound_upper - _ACTIVE_SET_TOLERANCE
    fixed = bounded & (state.bound_lower == state.bound_upper)
    outward = (at_lower & (state.gradient > _STRICT_COMPLEMENTARITY_TOLERANCE)) | (
        at_upper & (state.gradient < -_STRICT_COMPLEMENTARITY_TOLERANCE)
    )
    active = state.active_mask
    stable = (~active | fixed | outward).all(dim=-1)
    stable &= (~active | bounded).all(dim=-1)
    # Inactive coordinates exactly on a finite boundary are active-set switch points.
    stable &= (active | ~(bounded & (at_lower | at_upper))).all(dim=-1)
    stable &= ~stalled | active.any(dim=-1)
    return terminal & finite & stable, stable


def _smooth_quaternion_representative(values: _TensorValues, payload: _Payload) -> torch.Tensor:
    smooth = torch.ones(payload.batch_shape, dtype=torch.bool, device=values[payload.value_names[0]].device)
    for spec in payload.problem.vars:
        value = values[spec.name]
        scalar_parts: torch.Tensor | None = None
        if isinstance(spec, (SO3Variable, SE3Variable)):
            scalar_parts = value[..., -1:]
        elif isinstance(spec, RobotVariable):
            indices = tuple(
                event.stop - 1
                for event in spec.unit_coordinate_slices
                if event.start is not None and event.stop is not None and event.stop - event.start == 4
            )
            if indices:
                scalar_parts = value.index_select(-1, torch.tensor(indices, device=value.device))
        if scalar_parts is not None:
            threshold = max(_NONSMOOTH_TOLERANCE, 8.0 * torch.finfo(value.dtype).eps)
            smooth &= ~(scalar_parts.abs() <= threshold).reshape(*payload.batch_shape, -1).any(dim=-1)
    return smooth


def _detached_kernel(kernel: RobustKernel) -> RobustKernel:
    attributes = getattr(kernel, "__dict__", {})
    tensors = {name: value.detach() for name, value in attributes.items() if isinstance(value, torch.Tensor)}
    if not tensors:
        return kernel
    result = copy.copy(kernel)
    for name, value in tensors.items():
        object.__setattr__(result, name, value)
    return result


def _objective(
    values: _TensorValues, statics: Mapping[str, torch.Tensor], payload: _Payload
) -> tuple[torch.Tensor, torch.Tensor]:
    problem = payload.problem
    residual = problem._error_at({**values, **statics}, validate_runtime=False)
    costs: list[torch.Tensor] = []
    smooth = torch.ones(payload.batch_shape, dtype=torch.bool, device=residual.device)
    for item in problem.residuals:
        rows = residual[..., problem.row_offsets[item.name]]
        groups = _group_rows(rows, item.group_size)
        squared_norm = groups.square().sum(dim=-1)
        kernel = _detached_kernel(item.kernel or L2())
        costs.append(kernel.rho(squared_norm).sum(dim=-1))
        if isinstance(kernel, Huber):
            delta2 = torch.as_tensor(kernel.delta, dtype=residual.dtype, device=residual.device).square()
            scale = torch.maximum(torch.ones_like(squared_norm), delta2)
            smooth &= ~((squared_norm - delta2).abs() <= _NONSMOOTH_TOLERANCE * scale).any(dim=-1)
    return torch.stack(costs, dim=-1).sum(dim=-1), smooth


def _local_values(terminal: _TensorValues, delta: torch.Tensor, payload: _Payload) -> _TensorValues:
    pieces = {spec.name: delta[..., payload.problem.column_offsets[spec.name]] for spec in payload.problem.vars}
    return payload.problem.retract(terminal, pieces)


def _exact_system(
    terminal: _TensorValues,
    statics: Mapping[str, torch.Tensor],
    payload: _Payload,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    exemplar = terminal[payload.value_names[0]]
    delta = exemplar.new_zeros(*payload.batch_shape, payload.problem.tangent_dim_total, requires_grad=True)
    objective, smooth = _objective(_local_values(terminal, delta, payload), statics, payload)
    if not objective.requires_grad:
        raise ImplicitDifferentiationError(
            "the robust objective has no tangent derivative; implicit system is singular"
        )
    optimality = torch.autograd.grad(objective.sum(), delta, create_graph=True)[0]
    if not optimality.requires_grad:
        raise ImplicitDifferentiationError(
            "the robust optimality has no tangent derivative; implicit system is singular"
        )
    rows = []
    for index in range(payload.problem.tangent_dim_total):
        selector = torch.zeros_like(optimality)
        selector[..., index] = 1.0
        rows.append(torch.autograd.grad(optimality, delta, selector, retain_graph=True)[0])
    return optimality, torch.stack(rows, dim=-2).detach(), smooth


def _output_cotangent(
    terminal: _TensorValues, gradients: tuple[torch.Tensor | None, ...], payload: _Payload
) -> torch.Tensor:
    exemplar = terminal[payload.value_names[0]]
    delta = exemplar.new_zeros(*payload.batch_shape, payload.problem.tangent_dim_total, requires_grad=True)
    local = _local_values(terminal, delta, payload)
    outputs = tuple(local[name] for name in payload.value_names)
    materialized = tuple(
        torch.zeros_like(value) if grad is None else grad for value, grad in zip(outputs, gradients, strict=True)
    )
    return torch.autograd.grad(outputs, delta, materialized)[0].detach()


def _solve_adjoint(hessian: torch.Tensor, cotangent: torch.Tensor, payload: _Payload) -> torch.Tensor:
    tangent_dim = payload.problem.tangent_dim_total
    count = math.prod(payload.batch_shape) if payload.batch_shape else 1
    matrices = hessian.reshape(count, tangent_dim, tangent_dim)
    vectors = cotangent.reshape(count, tangent_dim)
    active = payload.state.active_mask.reshape(count, tangent_dim)
    result = torch.zeros_like(vectors)
    failed = torch.zeros(count, dtype=torch.bool, device=vectors.device)
    for index in range(count):
        free = ~active[index]
        free_count = int(free.sum())
        if not free_count:
            continue
        system = matrices[index][free][:, free].mT
        system = 0.5 * (system + system.mT)
        rhs = vectors[index, free]
        if not bool(torch.isfinite(system).all() and torch.isfinite(rhs).all()):
            failed[index] = True
            continue
        solved = torch.linalg.lstsq(system, rhs.unsqueeze(-1))
        solution = solved.solution.squeeze(-1)
        residual = torch.linalg.vector_norm(system @ solution - rhs)
        threshold = _LINEAR_SOLVE_ATOL + _LINEAR_SOLVE_RTOL * torch.linalg.vector_norm(rhs)
        if int(solved.rank) != free_count or not bool(torch.isfinite(solution).all() and residual <= threshold):
            failed[index] = True
            continue
        result[index, free] = solution
    if bool(failed.any()):
        invalid = _batch_indices(failed.reshape(payload.batch_shape), payload.batch_shape)
        raise ImplicitDifferentiationError(
            f"implicit adjoint system is singular or invalid for batch indices {invalid}",
            invalid_indices=invalid,
        )
    return result.reshape(*payload.batch_shape, tangent_dim)


class _ImplicitValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any) -> tuple[torch.Tensor, ...]:
        payload = args[-1]
        if not isinstance(payload, _Payload):
            raise TypeError("internal implicit payload is invalid")
        ctx.payload = payload
        ctx.value_count = len(payload.value_names)
        ctx.save_for_backward(*args[:-1])
        return tuple(value.clone() for value in args[: ctx.value_count])

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        payload: _Payload = ctx.payload
        value_count: int = ctx.value_count
        saved = ctx.saved_tensors
        terminal = dict(zip(payload.value_names, saved[:value_count], strict=True))
        eligible, stable = _eligibility(terminal, payload)
        if not bool(eligible.all()):
            invalid = ~eligible
            indices = _batch_indices(invalid, payload.batch_shape)
            statuses = _status_names(payload.state.status, invalid)
            unstable = _batch_indices(~stable, payload.batch_shape)
            raise ImplicitDifferentiationError(
                f"implicit backward requires a valid terminal state; invalid batch indices {indices}, "
                f"statuses {statuses}, unstable active sets {unstable}",
                invalid_indices=indices,
                statuses=statuses,
            )
        representative_smooth = _smooth_quaternion_representative(terminal, payload)
        if not bool(representative_smooth.all()):
            invalid = _batch_indices(~representative_smooth, payload.batch_shape)
            raise ImplicitDifferentiationError(
                f"absolute-pi principal-log branch cut at batch indices {invalid}",
                invalid_indices=invalid,
            )

        static_tensors = saved[value_count:]
        needed = ctx.needs_input_grad[value_count : value_count + len(payload.static_names)]
        statics: dict[str, torch.Tensor] = {}
        differentiable: list[torch.Tensor] = []
        positions: list[int] = []
        for position, (name, tensor, needs_grad) in enumerate(
            zip(payload.static_names, static_tensors, needed, strict=True)
        ):
            local = tensor.detach().requires_grad_(bool(needs_grad))
            statics[name] = local
            if local.requires_grad:
                differentiable.append(local)
                positions.append(position)

        with torch.enable_grad():
            optimality, hessian, smooth = _exact_system(terminal, statics, payload)
            cotangent = _output_cotangent(terminal, grad_outputs, payload)
            free_optimality = torch.where(payload.state.active_mask, 0.0, optimality)
            exact_valid = torch.isfinite(free_optimality).all(dim=-1) & torch.isfinite(hessian).all(dim=(-2, -1))
            exact_valid &= free_optimality.detach().abs().amax(dim=-1) <= _OPTIMALITY_TOLERANCE
            exact_valid &= smooth
            if not bool(exact_valid.all()):
                invalid = _batch_indices(~exact_valid, payload.batch_shape)
                raise ImplicitDifferentiationError(
                    f"recomputed robust optimality is non-KKT or nonsmooth for batch indices {invalid}",
                    invalid_indices=invalid,
                )
            adjoint = _solve_adjoint(hessian, cotangent, payload)
            computed = (
                torch.autograd.grad(optimality, differentiable, -adjoint, allow_unused=True) if differentiable else ()
            )

        disconnected = tuple(
            payload.static_names[position]
            for position, gradient in zip(positions, computed, strict=True)
            if gradient is None
        )
        if disconnected:
            raise ImplicitDifferentiationError(
                f"graph-carrying static variables are disconnected from terminal optimality: {disconnected}"
            )
        static_gradients: list[torch.Tensor | None] = [None] * len(payload.static_names)
        for position, gradient in zip(positions, computed, strict=True):
            static_gradients[position] = gradient
        return (*([None] * value_count), *static_gradients, None)


def _attach_implicit_gradients(
    terminal_values: Mapping[str, torch.Tensor],
    state: _ImplicitTerminalState,
    problem: Problem,
    *,
    forward_linearization: _ForwardLinearization = "dense",
    config: ImplicitDiffConfig | None = None,
) -> _TensorValues:
    """Attach an exact first-order implicit backward to terminal values."""
    if not isinstance(problem, Problem):
        raise TypeError("problem must be a named-block Problem")
    if not problem.residuals:
        raise ValueError("implicit differentiation requires at least one residual vector")
    resolved_config = ImplicitDiffConfig() if config is None else config
    if not isinstance(resolved_config, ImplicitDiffConfig):
        raise TypeError("config must be ImplicitDiffConfig or None")
    _validate_route(problem, forward_linearization, resolved_config)

    detached = {name: value.detach() for name, value in terminal_values.items()}
    batch_shape = problem._validate_trainable_values(detached)
    snapshot = _snapshot(state)
    _validate_state(snapshot, batch_shape, problem.tangent_dim_total, detached[problem.vars[0].name])
    value_names = tuple(spec.name for spec in problem.vars)
    static_variables = tuple(variable for variable in problem._ordered_variables if not variable.trainable)
    static_names = tuple(variable.name for variable in static_variables)
    nonfloating = tuple(
        variable.name
        for variable in static_variables
        if variable.tensor.requires_grad and not variable.tensor.is_floating_point()
    )
    if nonfloating:
        raise TypeError(f"implicit graph-carrying static variables must be floating tensors: {nonfloating}")
    payload = _Payload(
        problem,
        value_names,
        static_names,
        batch_shape,
        resolved_config,
        snapshot,
    )
    static_tensors = tuple(
        variable.tensor if variable.tensor.requires_grad else variable.tensor.detach() for variable in static_variables
    )
    raw = _ImplicitValues.apply(*(detached[name] for name in value_names), *static_tensors, payload)
    outputs = (raw,) if isinstance(raw, torch.Tensor) else raw
    return dict(zip(value_names, outputs, strict=True))


__all__ = [
    "ImplicitDiffConfig",
    "ImplicitDifferentiationError",
]
