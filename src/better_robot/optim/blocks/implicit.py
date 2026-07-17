"""First-order implicit differentiation for detached named-block solutions.

The forward optimiser deliberately remains graph-free.  This module attaches
one custom autograd node to a *terminal* :class:`Values` pytree and recomputes
the exact robust tangent optimality system in backward.  It consequently does
not retain, or differentiate through, the iteration trajectory.

The initial implementation is intentionally conservative:

* gradients are returned only to explicitly declared external parameters;
* the backward system is the undamped exact Hessian of the robust objective in
  a local product-manifold chart;
* stable active bounds are eliminated and have zero sensitivity;
* any invalid batch element, Huber kink, terminal quaternion representative at
  absolute pi, or singular solve rejects the complete backward; and
* dense recomputation has a fixed size cap.  A small banded forward can opt in
  to this exact dense oracle, but matrix-free forwards are rejected rather
  than being silently materialised.

This is a first-order API.  The custom backward is marked
``once_differentiable`` and does not promise derivative composition.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import math
from typing import Any, Literal, Protocol, TypeAlias

import torch
from torch.autograd.function import once_differentiable

from ..kernels import Huber, L2
from ..kernels.base import RobustKernel
from .manifolds import RobotConfig, SE3Manifold, SO3Manifold
from .problem import Problem
from .variables import Values

ForwardLinearization: TypeAlias = Literal["dense", "banded", "matrix_free"]

_CONVERGED = 1
_STALLED_AT_BOUNDS = 2
_STATUS_NAMES = {
    0: "running",
    _CONVERGED: "converged",
    _STALLED_AT_BOUNDS: "stalled_at_bounds",
    3: "maxiter",
    4: "failed",
}


class ImplicitDifferentiationError(RuntimeError):
    """An implicit backward cannot produce contract-valid gradients.

    ``invalid_indices`` are indices in the leading solve batch.  The unbatched
    solve is represented by ``()``.  ``statuses`` contains the corresponding
    terminal status names when the failure happened before factorisation.
    """

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
    """Static safety policy for the exact dense implicit backward.

    Parameters are Python values on purpose: bounds, tolerances, and solver
    policy are static configuration and never differentiation inputs.
    ``allow_banded_dense_backward`` makes a small structured-forward/dense-
    backward oracle visible at the call site.  The tangent-size cap applies to
    every route, including that opt-in oracle.
    """

    max_dense_tangent_dim: int = 512
    optimality_tolerance: float = 1e-5
    active_set_tolerance: float = 1e-7
    strict_complementarity_tolerance: float = 1e-8
    nonsmooth_tolerance: float = 1e-7
    linear_solve_atol: float = 1e-10
    linear_solve_rtol: float = 1e-5
    lstsq_rcond: float | None = None
    allow_banded_dense_backward: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_dense_tangent_dim, bool)
            or not isinstance(self.max_dense_tangent_dim, int)
            or self.max_dense_tangent_dim <= 0
        ):
            raise ValueError("max_dense_tangent_dim must be a positive int")
        for name in (
            "optimality_tolerance",
            "active_set_tolerance",
            "strict_complementarity_tolerance",
            "nonsmooth_tolerance",
            "linear_solve_atol",
            "linear_solve_rtol",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, torch.Tensor))
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative Python number")
        if self.lstsq_rcond is not None:
            if (
                isinstance(self.lstsq_rcond, (bool, torch.Tensor))
                or not isinstance(self.lstsq_rcond, (int, float))
                or not math.isfinite(float(self.lstsq_rcond))
                or float(self.lstsq_rcond) < 0.0
            ):
                raise ValueError("lstsq_rcond must be None or a finite non-negative Python number")
        if not isinstance(self.allow_banded_dense_backward, bool):
            raise TypeError("allow_banded_dense_backward must be a bool")


class ImplicitTerminalState(Protocol):
    """Tensor fields consumed from a canonical final LM/GN state."""

    @property
    def status(self) -> torch.Tensor: ...

    @property
    def implicit_valid(self) -> torch.Tensor: ...

    @property
    def active_mask(self) -> torch.Tensor: ...

    @property
    def gradient(self) -> torch.Tensor: ...

    @property
    def projected_grad_norm(self) -> torch.Tensor: ...

    @property
    def bound_state_index(self) -> torch.Tensor: ...

    @property
    def bound_lower(self) -> torch.Tensor: ...

    @property
    def bound_upper(self) -> torch.Tensor: ...

    @property
    def bounded_mask(self) -> torch.Tensor: ...


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
class _ImplicitPayload:
    problem: Problem
    value_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    differentiable_names: frozenset[str]
    batch_shape: tuple[int, ...]
    default_kernel: RobustKernel
    config: ImplicitDiffConfig
    snapshot: _StateSnapshot


def _snapshot_state(state: ImplicitTerminalState) -> _StateSnapshot:
    return _StateSnapshot(
        status=state.status.detach().clone(),
        implicit_valid=state.implicit_valid.detach().clone(),
        active_mask=state.active_mask.detach().clone(),
        gradient=state.gradient.detach().clone(),
        projected_grad_norm=state.projected_grad_norm.detach().clone(),
        bound_state_index=state.bound_state_index.detach().clone(),
        bound_lower=state.bound_lower.detach().clone(),
        bound_upper=state.bound_upper.detach().clone(),
        bounded_mask=state.bounded_mask.detach().clone(),
    )


def _validate_state_layout(
    snapshot: _StateSnapshot,
    *,
    batch_shape: tuple[int, ...],
    tangent_dim: int,
    exemplar: torch.Tensor,
) -> None:
    batch_fields = {
        "status": snapshot.status,
        "implicit_valid": snapshot.implicit_valid,
        "projected_grad_norm": snapshot.projected_grad_norm,
    }
    for name, tensor in batch_fields.items():
        if tuple(tensor.shape) != batch_shape:
            raise ValueError(f"implicit state {name} must have batch shape {batch_shape}, got {tuple(tensor.shape)}")
    vector_fields = {
        "active_mask": snapshot.active_mask,
        "gradient": snapshot.gradient,
    }
    for name, tensor in vector_fields.items():
        expected = (*batch_shape, tangent_dim)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"implicit state {name} must have shape {expected}, got {tuple(tensor.shape)}")
    static_fields = {
        "bound_state_index": snapshot.bound_state_index,
        "bound_lower": snapshot.bound_lower,
        "bound_upper": snapshot.bound_upper,
        "bounded_mask": snapshot.bounded_mask,
    }
    for name, tensor in static_fields.items():
        if tuple(tensor.shape) != (tangent_dim,):
            raise ValueError(f"implicit state {name} must have shape {(tangent_dim,)}, got {tuple(tensor.shape)}")
    for name, tensor in (*batch_fields.items(), *vector_fields.items(), *static_fields.items()):
        if tensor.device != exemplar.device:
            raise ValueError(
                f"implicit state {name} must be on terminal Values device {exemplar.device}, got {tensor.device}"
            )
    if snapshot.gradient.dtype != exemplar.dtype:
        raise ValueError(
            "implicit state gradient must preserve terminal Values dtype; "
            f"got {snapshot.gradient.dtype} and {exemplar.dtype}"
        )


def _validate_route(
    problem: Problem,
    route: ForwardLinearization,
    config: ImplicitDiffConfig,
) -> None:
    if route not in {"dense", "banded", "matrix_free"}:
        raise ValueError(f"unknown forward_linearization {route!r}")
    if route == "matrix_free":
        raise ValueError(
            "implicit differentiation does not materialize a matrix-free forward; "
            "use the dense route for a small problem or wait for an exact operator backward"
        )
    if route == "banded" and not config.allow_banded_dense_backward:
        raise ValueError(
            "a banded forward currently requires an exact dense implicit backward; set "
            "ImplicitDiffConfig(allow_banded_dense_backward=True) for a small oracle, "
            "or defer until the structured backward is available"
        )
    if problem.tangent_dim_total > config.max_dense_tangent_dim:
        raise ValueError(
            "exact implicit backward would materialize a dense tangent Hessian of size "
            f"{problem.tangent_dim_total}; configured cap is {config.max_dense_tangent_dim}. "
            "Increase the explicit cap only for a reviewed small oracle, or use a future "
            "structured/operator backward."
        )


def _batch_indices(mask: torch.Tensor, batch_shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    flat_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten().detach().cpu().tolist()
    if not batch_shape:
        return ((),) if flat_indices else ()
    result: list[tuple[int, ...]] = []
    for flat_index in flat_indices:
        coordinates: list[int] = []
        remainder = int(flat_index)
        for size in reversed(batch_shape):
            coordinates.append(remainder % size)
            remainder //= size
        result.append(tuple(reversed(coordinates)))
    return tuple(result)


def _status_names(status: torch.Tensor, invalid: torch.Tensor) -> tuple[str, ...]:
    values = status.reshape(-1)[invalid.reshape(-1)].detach().cpu().tolist()
    return tuple(_STATUS_NAMES.get(int(value), f"unknown({int(value)})") for value in values)


def _state_coordinates(
    values: Mapping[str, torch.Tensor],
    problem: Problem,
    batch_shape: tuple[int, ...],
    state_index: torch.Tensor,
) -> torch.Tensor:
    flattened = torch.cat(
        tuple(values[spec.name].reshape(*batch_shape, -1) for spec in problem.vars),
        dim=-1,
    )
    mapped = state_index >= 0
    gathered = flattened.index_select(-1, state_index.clamp(min=0))
    return torch.where(mapped, gathered, torch.zeros_like(gathered))


def _state_eligibility(
    values: Mapping[str, torch.Tensor],
    payload: _ImplicitPayload,
) -> tuple[torch.Tensor, torch.Tensor]:
    snapshot = payload.snapshot
    config = payload.config
    status = snapshot.status
    converged = (status == _CONVERGED) & snapshot.implicit_valid
    stalled = status == _STALLED_AT_BOUNDS
    terminal = converged | stalled
    finite = (
        torch.isfinite(snapshot.gradient).all(dim=-1)
        & torch.isfinite(snapshot.projected_grad_norm)
        & (snapshot.projected_grad_norm <= config.optimality_tolerance)
    )
    for value in values.values():
        finite = finite & torch.isfinite(value).reshape(*payload.batch_shape, -1).all(dim=-1)

    coordinate = _state_coordinates(
        values,
        payload.problem,
        payload.batch_shape,
        snapshot.bound_state_index,
    )
    bounded = snapshot.bounded_mask
    at_lower = coordinate <= snapshot.bound_lower + config.active_set_tolerance
    at_upper = coordinate >= snapshot.bound_upper - config.active_set_tolerance
    fixed = bounded & (snapshot.bound_lower == snapshot.bound_upper)
    gradient = snapshot.gradient
    strict_outward = (
        at_lower & (gradient > config.strict_complementarity_tolerance)
    ) | (
        at_upper & (gradient < -config.strict_complementarity_tolerance)
    )
    active = snapshot.active_mask
    active_stable = (~active | fixed | strict_outward).all(dim=-1)
    active_is_bounded = (~active | bounded).all(dim=-1)
    # A nominally inactive coordinate at a finite boundary is an active-set
    # switch point.  Even a zero gradient there is not assigned a derivative.
    inactive_clear = (active | ~(bounded & (at_lower | at_upper))).all(dim=-1)
    stalled_has_active = ~stalled | active.any(dim=-1)
    stable = active_stable & active_is_bounded & inactive_clear & stalled_has_active
    return terminal & finite & stable, stable


def _terminal_quaternion_representative_is_smooth(
    values: Mapping[str, torch.Tensor],
    payload: _ImplicitPayload,
) -> torch.Tensor:
    """Reject terminal quaternion representatives at the absolute-pi seam."""
    smooth = torch.ones(
        payload.batch_shape,
        dtype=torch.bool,
        device=values[payload.value_names[0]].device,
    )
    for spec in payload.problem.vars:
        value = values[spec.name]
        scalar_parts: torch.Tensor | None = None
        if isinstance(spec.manifold, (SO3Manifold, SE3Manifold)):
            scalar_parts = value[..., -1:]
        elif isinstance(spec.manifold, RobotConfig):
            scalar_indices = tuple(
                unit_slice.stop - 1
                for unit_slice in spec.manifold.unit_coordinate_slices
                if unit_slice.start is not None
                and unit_slice.stop is not None
                and unit_slice.stop - unit_slice.start == 4
            )
            if scalar_indices:
                index = torch.tensor(scalar_indices, dtype=torch.long, device=value.device)
                scalar_parts = value.index_select(-1, index)
        if scalar_parts is None:
            continue
        # Unit quaternions make this an absolute, dimensionless test. Keep an
        # epsilon floor so setting the user tolerance to zero cannot miss an
        # exact-pi rotation represented by finite-precision sin/cos.
        threshold = max(
            payload.config.nonsmooth_tolerance,
            8.0 * torch.finfo(value.dtype).eps,
        )
        branch_cut = scalar_parts.abs() <= threshold
        smooth = smooth & ~branch_cut.reshape(*payload.batch_shape, -1).any(dim=-1)
    return smooth


def validate_implicit_input_roles(
    values: Mapping[str, torch.Tensor],
    problem: Problem,
) -> None:
    """Require optimized values and external parameters to be distinct tensors."""
    collisions = tuple(
        (value_name, parameter_name)
        for value_name, value in values.items()
        for parameter_name, parameter in problem.parameters.items()
        if value is parameter
    )
    if collisions:
        details = ", ".join(
            f"Values[{value_name!r}] / parameters[{parameter_name!r}]"
            for value_name, parameter_name in collisions
        )
        raise ValueError(
            "implicit differentiation requires optimized values and external "
            f"parameters to occupy distinct tensor roles; identity collision: {details}"
        )


def _weight_overrides(
    problem: Problem,
) -> dict[str, torch.Tensor]:
    """Detach item-level tensor weights until they have a named binding."""
    result: dict[str, torch.Tensor] = {}
    for item in problem.residuals:
        if not isinstance(item.weight, torch.Tensor):
            continue
        result[item.name] = item.weight.detach()
    return result


def _detach_kernel_tensor_attributes(
    kernel: RobustKernel,
) -> RobustKernel:
    """Detach tensor kernel attributes until they have a named binding."""
    attributes = getattr(kernel, "__dict__", None)
    if not isinstance(attributes, dict) or not any(isinstance(value, torch.Tensor) for value in attributes.values()):
        return kernel
    substituted = copy.copy(kernel)
    for name, value in attributes.items():
        if not isinstance(value, torch.Tensor):
            continue
        object.__setattr__(substituted, name, value.detach())
    return substituted


def _robust_objective(
    values: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
    payload: _ImplicitPayload,
) -> tuple[torch.Tensor, torch.Tensor]:
    problem = payload.problem
    context = problem._make_context(values, parameters=parameters)
    residual = problem._residual_with_context(
        values,
        payload.batch_shape,
        context,
        _weight_overrides(problem),
        validate_runtime=False,
    )
    costs: list[torch.Tensor] = []
    smooth = torch.ones(payload.batch_shape, dtype=torch.bool, device=residual.device)
    for item in problem.residuals:
        rows = residual[..., problem.row_offsets[item.name]]
        groups = rows.reshape(
            *payload.batch_shape,
            item.residual.dim // item.group_size,
            item.group_size,
        )
        squared_norm = groups.square().sum(dim=-1)
        raw_kernel = item.kernel if item.kernel is not None else payload.default_kernel
        kernel = _detach_kernel_tensor_attributes(raw_kernel)
        costs.append(kernel.rho(squared_norm).sum(dim=-1))
        if isinstance(kernel, Huber):
            delta_squared = torch.as_tensor(kernel.delta, dtype=residual.dtype, device=residual.device).square()
            scale = torch.maximum(torch.ones_like(squared_norm), delta_squared)
            at_kink = (squared_norm - delta_squared).abs() <= payload.config.nonsmooth_tolerance * scale
            smooth = smooth & ~at_kink.any(dim=-1)
    objective = torch.stack(costs, dim=-1).sum(dim=-1)
    return objective, smooth


def _split_delta(delta: torch.Tensor, problem: Problem) -> Values:
    return {spec.name: delta[..., problem.column_offsets[spec.name]] for spec in problem.vars}


def _local_values(
    terminal_values: Mapping[str, torch.Tensor],
    delta: torch.Tensor,
    payload: _ImplicitPayload,
) -> Values:
    return payload.problem._retract_prevalidated(
        terminal_values,
        _split_delta(delta, payload.problem),
        batch_shape=payload.batch_shape,
    )


def _exact_optimality_and_hessian(
    terminal_values: Mapping[str, torch.Tensor],
    parameters: Mapping[str, torch.Tensor],
    payload: _ImplicitPayload,
) -> tuple[torch.Tensor, torch.Tensor, Values, torch.Tensor]:
    exemplar = terminal_values[payload.value_names[0]]
    delta = exemplar.new_zeros(*payload.batch_shape, payload.problem.tangent_dim_total, requires_grad=True)
    local_values = _local_values(terminal_values, delta, payload)
    objective, smooth = _robust_objective(local_values, parameters, payload)
    if not objective.requires_grad:
        raise ImplicitDifferentiationError(
            "the robust objective has no tangent derivative; the implicit system is singular"
        )
    optimality = torch.autograd.grad(objective.sum(), delta, create_graph=True)[0]
    if not optimality.requires_grad:
        raise ImplicitDifferentiationError(
            "the robust optimality residual has no tangent derivative; the implicit system is singular"
        )
    rows: list[torch.Tensor] = []
    for row_index in range(payload.problem.tangent_dim_total):
        selector = torch.zeros_like(optimality)
        selector[..., row_index] = 1.0
        row = torch.autograd.grad(
            optimality,
            delta,
            grad_outputs=selector,
            retain_graph=True,
            create_graph=False,
        )[0]
        rows.append(row)
    hessian = torch.stack(rows, dim=-2).detach()
    return optimality, hessian, local_values, smooth


def _terminal_cotangent(
    local_values: Mapping[str, torch.Tensor],
    delta: torch.Tensor,
    grad_outputs: tuple[torch.Tensor | None, ...],
    payload: _ImplicitPayload,
) -> torch.Tensor:
    materialized = tuple(
        torch.zeros_like(local_values[name]) if gradient is None else gradient
        for name, gradient in zip(payload.value_names, grad_outputs, strict=True)
    )
    tangent = torch.autograd.grad(
        tuple(local_values[name] for name in payload.value_names),
        delta,
        grad_outputs=materialized,
        retain_graph=True,
        create_graph=False,
    )[0]
    return tangent.detach()


def _solve_adjoint(
    hessian: torch.Tensor,
    cotangent: torch.Tensor,
    active_mask: torch.Tensor,
    payload: _ImplicitPayload,
) -> torch.Tensor:
    tangent_dim = payload.problem.tangent_dim_total
    count = math.prod(payload.batch_shape) if payload.batch_shape else 1
    flat_hessian = hessian.reshape(count, tangent_dim, tangent_dim)
    flat_cotangent = cotangent.reshape(count, tangent_dim)
    flat_active = active_mask.reshape(count, tangent_dim)
    adjoint = torch.zeros_like(flat_cotangent)
    failed = torch.zeros(count, dtype=torch.bool, device=cotangent.device)
    for batch_index in range(count):
        free = ~flat_active[batch_index]
        free_count = int(free.sum().item())  # implicit backward is an eager CPU contract
        if free_count == 0:
            continue
        system = flat_hessian[batch_index][free][:, free].mT
        system = 0.5 * (system + system.mT)
        rhs = flat_cotangent[batch_index, free]
        if not bool(torch.isfinite(system).all() and torch.isfinite(rhs).all()):
            failed[batch_index] = True
            continue
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info.item()) == 0:
            solution = torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
        else:
            least_squares = torch.linalg.lstsq(system, rhs.unsqueeze(-1), rcond=payload.config.lstsq_rcond)
            if int(least_squares.rank.item()) != free_count:
                failed[batch_index] = True
                continue
            solution = least_squares.solution.squeeze(-1)
        residual_norm = torch.linalg.vector_norm(system @ solution - rhs)
        rhs_norm = torch.linalg.vector_norm(rhs)
        threshold = payload.config.linear_solve_atol + payload.config.linear_solve_rtol * rhs_norm
        if not bool(torch.isfinite(solution).all() and torch.isfinite(residual_norm) and residual_norm <= threshold):
            failed[batch_index] = True
            continue
        adjoint[batch_index, free] = solution
    if bool(failed.any()):
        invalid = _batch_indices(failed.reshape(payload.batch_shape), payload.batch_shape)
        raise ImplicitDifferentiationError(
            "implicit adjoint system is singular, nonfinite, or failed its residual check "
            f"for batch indices {invalid}",
            invalid_indices=invalid,
        )
    return adjoint.reshape(*payload.batch_shape, tangent_dim)


class _ImplicitValuesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any) -> tuple[torch.Tensor, ...]:
        payload = args[-1]
        if not isinstance(payload, _ImplicitPayload):
            raise TypeError("internal implicit payload is invalid")
        tensors = args[:-1]
        value_count = len(payload.value_names)
        ctx.payload = payload
        ctx.value_count = value_count
        ctx.save_for_backward(*tensors)
        return tuple(tensor.clone() for tensor in tensors[:value_count])

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        payload: _ImplicitPayload = ctx.payload
        saved = ctx.saved_tensors
        value_count: int = ctx.value_count
        terminal_tensors = saved[:value_count]
        parameter_tensors = saved[value_count:]
        terminal_values = dict(zip(payload.value_names, terminal_tensors, strict=True))

        eligible, stable = _state_eligibility(terminal_values, payload)
        if not bool(eligible.all()):
            invalid_mask = ~eligible
            indices = _batch_indices(invalid_mask, payload.batch_shape)
            statuses = _status_names(payload.snapshot.status, invalid_mask)
            unstable = _batch_indices(~stable, payload.batch_shape)
            raise ImplicitDifferentiationError(
                "implicit backward requires a KKT-valid terminal state and a locally stable active set; "
                f"invalid batch indices {indices}, statuses {statuses}, unstable active sets {unstable}",
                invalid_indices=indices,
                statuses=statuses,
            )

        representative_smooth = _terminal_quaternion_representative_is_smooth(
            terminal_values,
            payload,
        )
        if not bool(representative_smooth.all()):
            invalid = _batch_indices(~representative_smooth, payload.batch_shape)
            raise ImplicitDifferentiationError(
                "implicit backward rejects a terminal manifold quaternion "
                "representative at the absolute-pi principal-log branch cut; "
                f"batch indices {invalid}",
                invalid_indices=invalid,
            )

        needs_input_grad = ctx.needs_input_grad[value_count : value_count + len(payload.parameter_names)]
        parameter_values: dict[str, torch.Tensor] = {}
        differentiable_inputs: list[torch.Tensor] = []
        differentiable_positions: list[int] = []
        for position, (name, tensor, needed) in enumerate(
            zip(payload.parameter_names, parameter_tensors, needs_input_grad, strict=True)
        ):
            requires_grad = bool(needed and name in payload.differentiable_names)
            local = tensor.detach().requires_grad_(requires_grad)
            parameter_values[name] = local
            if requires_grad:
                differentiable_inputs.append(local)
                differentiable_positions.append(position)

        with torch.enable_grad():
            optimality, hessian, local_values, smooth = _exact_optimality_and_hessian(
                terminal_values,
                parameter_values,
                payload,
            )
            delta = next(iter(local_values.values())).new_zeros(
                *payload.batch_shape,
                payload.problem.tangent_dim_total,
                requires_grad=True,
            )
            # Rebuild the local chart once so the output VJP shares the same
            # manifold/bound projection convention as the optimality chart.
            cotangent_values = _local_values(terminal_values, delta, payload)
            terminal_cotangent = _terminal_cotangent(
                cotangent_values,
                delta,
                grad_outputs,
                payload,
            )

            free_optimality = torch.where(
                payload.snapshot.active_mask,
                torch.zeros_like(optimality),
                optimality,
            )
            exact_finite = torch.isfinite(free_optimality).all(dim=-1) & torch.isfinite(hessian).all(dim=(-2, -1))
            exact_kkt = free_optimality.detach().abs().amax(dim=-1) <= payload.config.optimality_tolerance
            exact_valid = exact_finite & exact_kkt & smooth
            if not bool(exact_valid.all()):
                invalid = _batch_indices(~exact_valid, payload.batch_shape)
                raise ImplicitDifferentiationError(
                    "recomputed exact robust optimality is nonfinite, non-KKT, or nonsmooth "
                    f"for batch indices {invalid}",
                    invalid_indices=invalid,
                )

            adjoint = _solve_adjoint(
                hessian,
                terminal_cotangent,
                payload.snapshot.active_mask,
                payload,
            )
            computed = (
                torch.autograd.grad(
                    optimality,
                    differentiable_inputs,
                    grad_outputs=-adjoint,
                    allow_unused=True,
                    create_graph=False,
                )
                if differentiable_inputs
                else ()
            )

        disconnected = tuple(
            payload.parameter_names[position]
            for position, gradient in zip(differentiable_positions, computed, strict=True)
            if gradient is None
        )
        if disconnected:
            raise ImplicitDifferentiationError(
                "declared differentiable external parameters are disconnected "
                "from the terminal robust optimality system; read them by stable "
                f"name in residual/provider context: {disconnected}"
            )

        parameter_gradients: list[torch.Tensor | None] = [None] * len(payload.parameter_names)
        for position, gradient in zip(
            differentiable_positions,
            computed,
            strict=True,
        ):
            assert gradient is not None
            parameter_gradients[position] = gradient
        return (*([None] * value_count), *parameter_gradients, None)


def attach_implicit_gradients(
    terminal_values: Mapping[str, torch.Tensor],
    state: ImplicitTerminalState,
    problem: Problem,
    *,
    default_kernel: RobustKernel | None = None,
    forward_linearization: ForwardLinearization = "dense",
    config: ImplicitDiffConfig | None = None,
) -> Values:
    """Attach an exact first-order implicit backward to detached terminal values.

    The returned tensors carry gradients only to names in
    ``Problem.differentiable_external_parameters``.  The initial guess is not
    an input to this function and therefore receives no implicit gradient.
    Validation of terminal statuses and exact KKT conditions happens in
    backward so ordinary detached forward use remains cheap; an invalid
    element rejects the complete requested batch.
    """

    if not isinstance(problem, Problem):
        raise TypeError("problem must be a named-block Problem")
    validate_implicit_input_roles(terminal_values, problem)
    problem.require_least_squares("implicit differentiation")
    if not problem.residuals:
        raise ValueError("implicit differentiation requires at least one residual vector")
    resolved_config = ImplicitDiffConfig() if config is None else config
    if not isinstance(resolved_config, ImplicitDiffConfig):
        raise TypeError("config must be ImplicitDiffConfig or None")
    resolved_kernel = L2() if default_kernel is None else default_kernel
    if not isinstance(resolved_kernel, RobustKernel):
        raise TypeError("default_kernel must implement rho(squared_norm) and weight(squared_norm)")
    _validate_route(problem, forward_linearization, resolved_config)

    detached_values = {name: value.detach() for name, value in terminal_values.items()}
    batch_shape = problem._validate_values(detached_values)
    snapshot = _snapshot_state(state)
    exemplar = detached_values[problem.vars[0].name]
    _validate_state_layout(
        snapshot,
        batch_shape=batch_shape,
        tangent_dim=problem.tangent_dim_total,
        exemplar=exemplar,
    )
    value_names = tuple(spec.name for spec in problem.vars)
    parameter_names = tuple(problem.parameters)
    differentiable_names = frozenset(problem.differentiable_external_parameters)
    nonfloating = tuple(
        name
        for name in parameter_names
        if name in differentiable_names and not problem.parameters[name].is_floating_point()
    )
    if nonfloating:
        raise TypeError(
            "implicit differentiable parameters must be floating tensors; "
            f"non-floating names {nonfloating}"
        )
    payload = _ImplicitPayload(
        problem=problem,
        value_names=value_names,
        parameter_names=parameter_names,
        differentiable_names=differentiable_names,
        batch_shape=batch_shape,
        default_kernel=resolved_kernel,
        config=resolved_config,
        snapshot=snapshot,
    )
    parameter_inputs = tuple(
        problem.parameters[name] if name in differentiable_names else problem.parameters[name].detach()
        for name in parameter_names
    )
    raw_outputs = _ImplicitValuesFunction.apply(
        *(detached_values[name] for name in value_names),
        *parameter_inputs,
        payload,
    )
    outputs = (raw_outputs,) if isinstance(raw_outputs, torch.Tensor) else raw_outputs
    return dict(zip(value_names, outputs, strict=True))


__all__ = [
    "ForwardLinearization",
    "ImplicitDiffConfig",
    "ImplicitDifferentiationError",
    "ImplicitTerminalState",
    "attach_implicit_gradients",
]
