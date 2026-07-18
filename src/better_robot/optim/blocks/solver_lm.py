"""Batched named-block Gauss--Newton and Levenberg--Marquardt solvers.

The update is pure, fixed-shape, and tensor-branching only. Private
``GraphExecutor`` CUDA tests cover capture/replay parity for fixed groups of
these updates; the public ``run`` driver remains eager.

Bounds use projected active-set LM with a projected-gradient safeguard.  The
normal system is restricted before solving, the gain ratio uses the tangent
step actually taken after projection, and terminal bound stalls are reported
only after a projected-gradient KKT check.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from functools import reduce
import math
from operator import mul
from typing import Literal, NamedTuple, TypeAlias, cast

import torch

from ..kernels import L2
from ..kernels.base import RobustKernel
from ..solvers import BandedCholesky, Cholesky, NormalCG
from ..solvers.base import LinearSolveResult, LinearSolveStatus, LinearSolver
from ..structure import (
    BlockBandedMatrix,
    LinearizationDecision,
    LinearizationMode,
    LinearizationReason,
    NormalOperator,
)
from ._solver_common import _batch_shape, _blend_values
from .implicit import ImplicitDiffConfig
from .manifolds import Euclidean, RobotConfig, _joint_coordinate_layout
from .problem import JacobianStrategy, Problem
from .variables import Values, detach_values

_ResolvedLinearSolver: TypeAlias = LinearSolver | BandedCholesky | NormalCG | Cholesky


class LMStatus(IntEnum):
    """Per-element terminal status stored as ``torch.int8``."""

    RUNNING = 0
    CONVERGED = 1
    STALLED_AT_BOUNDS = 2
    MAXITER = 3
    FAILED = 4


class LMState(NamedTuple):
    """Fixed-structure tensor state for one arbitrary leading batch shape.

    ``converged`` means that an element has satisfied either the unconstrained
    or bound-constrained KKT test.  Inspect ``status`` to distinguish those
    two successful terminal cases.  ``implicit_valid`` records structural
    terminal eligibility; implicit attachment separately validates and solves
    the backward system.
    """

    residual: torch.Tensor
    robust_weights: torch.Tensor
    cost: torch.Tensor
    mu: torch.Tensor
    increase_factor: torch.Tensor
    gain_ratio: torch.Tensor
    gradient: torch.Tensor
    grad_norm: torch.Tensor
    projected_gradient: torch.Tensor
    projected_grad_norm: torch.Tensor
    step_norm: torch.Tensor
    relative_decrease: torch.Tensor
    active_mask: torch.Tensor
    factorization_ok: torch.Tensor
    linear_solve_iterations: torch.Tensor
    linear_solve_residual_norm: torch.Tensor
    linear_solve_relative_residual: torch.Tensor
    linear_solve_status: torch.Tensor
    previous_linear_step: torch.Tensor
    converged: torch.Tensor
    implicit_valid: torch.Tensor
    status: torch.Tensor
    iterations: torch.Tensor
    scale: torch.Tensor
    bound_state_index: torch.Tensor
    bound_lower: torch.Tensor
    bound_upper: torch.Tensor
    bounded_mask: torch.Tensor

    @property
    def kkt_norm(self) -> torch.Tensor:
        """Alias for the projected-gradient infinity norm."""
        return self.projected_grad_norm


@dataclass(frozen=True)
class _JacobianOperators:
    """Evaluation-local physical-tangent Jacobian operations."""

    jvp_fn: Callable[[torch.Tensor], torch.Tensor]
    vjp_fn: Callable[[torch.Tensor], torch.Tensor]
    normal_diagonal_tensor: torch.Tensor

    def jvp(self, tangent: torch.Tensor) -> torch.Tensor:
        return self.jvp_fn(tangent)

    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor:
        return self.vjp_fn(cotangent)

    def normal_matvec(self, tangent: torch.Tensor) -> torch.Tensor:
        return self.vjp(self.jvp(tangent))

    def normal_diagonal(self) -> torch.Tensor:
        return self.normal_diagonal_tensor


class _LinearizedLeastSquares(NamedTuple):
    residual: torch.Tensor
    robust_weights: torch.Tensor
    cost: torch.Tensor
    gradient: torch.Tensor
    normal_diagonal: torch.Tensor
    operators: _JacobianOperators
    normal: torch.Tensor | BlockBandedMatrix | None
    grad_norm: torch.Tensor
    projected_gradient: torch.Tensor
    projected_grad_norm: torch.Tensor
    active_mask: torch.Tensor
    finite: torch.Tensor


def _max_abs(vector: torch.Tensor) -> torch.Tensor:
    return vector.abs().amax(dim=-1)


def _split_step(step: torch.Tensor, problem: Problem) -> Values:
    return {spec.name: step[..., problem.column_offsets[spec.name]] for spec in problem.vars}


def _difference_reduced(
    x0: Values,
    x1: Values,
    problem: Problem,
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    full = problem._difference_prevalidated(x0, x1, batch_shape=batch_shape)
    return torch.cat(
        tuple(spec._gather_tangent_prevalidated(full[spec.name]) for spec in problem.vars),
        dim=-1,
    )


def _retract_reduced(
    values: Values,
    step: torch.Tensor,
    problem: Problem,
    batch_shape: tuple[int, ...],
) -> Values:
    return problem._retract_prevalidated(
        values,
        _split_step(step, problem),
        batch_shape=batch_shape,
    )


def _limit_block_step_norms(
    step: torch.Tensor,
    problem: Problem,
    limits: tuple[tuple[str, float], ...],
) -> torch.Tensor:
    """Clamp configured physical tangent-block norms without tensor branching."""
    if not limits:
        return step
    by_name = dict(limits)
    blocks: list[torch.Tensor] = []
    for spec in problem.vars:
        block = step[..., problem.column_offsets[spec.name]]
        limit = by_name.get(spec.name)
        if limit is not None and spec.free_dim:
            norm = torch.linalg.vector_norm(block, dim=-1, keepdim=True)
            denominator = norm.clamp_min(torch.finfo(step.dtype).tiny)
            multiplier = (limit / denominator).clamp(max=1.0)
            block = block * multiplier
        blocks.append(block)
    return torch.cat(blocks, dim=-1)


def _validate_block_step_limits(limits: tuple[tuple[str, float], ...]) -> None:
    if not isinstance(limits, tuple):
        raise TypeError("block_step_limits must be a tuple of (block_name, max_norm) pairs")
    seen: set[str] = set()
    for entry in limits:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError("block_step_limits must be a tuple of (block_name, max_norm) pairs")
        block_name, max_norm = entry
        if not isinstance(block_name, str) or not block_name:
            raise TypeError("block_step_limits block names must be non-empty strings")
        if block_name in seen:
            raise ValueError(f"duplicate block_step_limits entry for {block_name!r}")
        seen.add(block_name)
        if (
            isinstance(max_norm, (bool, torch.Tensor))
            or not isinstance(max_norm, (int, float))
            or not math.isfinite(float(max_norm))
            or float(max_norm) <= 0.0
        ):
            raise ValueError(f"block_step_limits max norm for {block_name!r} must be a finite positive Python number")


def _detach_state(state: LMState) -> LMState:
    return LMState(*(tensor.detach() for tensor in state))


def _static_layout(  # noqa: PLR0912 - handles the finite supported variable layouts
    values: Values,
    problem: Problem,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build scale and state-box-to-tangent maps once at the public boundary."""
    exemplar = values[problem.vars[0].name]
    nt = problem.tangent_dim_total
    scale = exemplar.new_ones(nt)
    state_index = torch.full((nt,), -1, dtype=torch.long, device=exemplar.device)
    lower = exemplar.new_full((nt,), -torch.inf)
    upper = exemplar.new_full((nt,), torch.inf)
    state_offset = 0

    for spec in problem.vars:
        column = problem.column_offsets[spec.name]
        free_indices = spec.free_indices.to(device=exemplar.device)
        if spec.free_scale is not None:
            scale[column] = spec.free_scale.to(device=exemplar.device)

        full_mapping = torch.full(
            (spec.tangent_dim,),
            -1,
            dtype=torch.long,
            device=exemplar.device,
        )
        full_lower = exemplar.new_full((spec.tangent_dim,), -torch.inf)
        full_upper = exemplar.new_full((spec.tangent_dim,), torch.inf)
        if isinstance(spec.manifold, Euclidean):
            full_mapping = torch.arange(
                state_offset,
                state_offset + spec.tangent_dim,
                dtype=torch.long,
                device=exemplar.device,
            )
            if spec.bounds is not None:
                full_lower = spec.bounds.lower.reshape(-1)
                full_upper = spec.bounds.upper.reshape(-1)
        elif isinstance(spec.manifold, RobotConfig):
            model = spec.manifold.model
            q_for_v = [-1] * model.nv
            unsafe_q: list[int] = []
            for joint, nq_joint, nv_joint, iq, iv in zip(
                model.joint_models,
                model.nqs,
                model.nvs,
                model.idx_qs,
                model.idx_vs,
                strict=True,
            ):
                if nq_joint == 0 and nv_joint == 0:
                    continue
                layout = _joint_coordinate_layout(joint)
                for local_v, local_q in enumerate(layout.q_for_v):
                    if local_q >= 0:
                        q_for_v[iv + local_v] = iq + local_q
                unsafe_q.extend(iq + index for index in layout.unsafe_q)
            if spec.bounds is not None and unsafe_q:
                unsafe = torch.tensor(unsafe_q, dtype=torch.long, device=exemplar.device)
                unsafe_finite = torch.isfinite(spec.bounds.lower.index_select(0, unsafe)) | torch.isfinite(
                    spec.bounds.upper.index_select(0, unsafe)
                )
                if bool(unsafe_finite.any()):  # bench-ok: init-only static bounds validation
                    raise ValueError(
                        f"RobotConfig VarSpec {spec.name!r} has a finite free-flyer "
                        "translation bound. World-axis state boxes are not axis-aligned "
                        "in the right-local SE(3) tangent; defer this constraint or "
                        "express it as a residual."
                    )
            event_count = reduce(mul, spec.shape[:-1], 1)
            for event in range(event_count):
                for local_v, local_q in enumerate(q_for_v):
                    tangent_index = event * model.nv + local_v
                    if local_q < 0:
                        continue
                    full_mapping[tangent_index] = state_offset + event * model.nq + local_q
                    if spec.bounds is not None:
                        full_lower[tangent_index] = spec.bounds.lower[local_q]
                        full_upper[tangent_index] = spec.bounds.upper[local_q]

        state_index[column] = full_mapping.index_select(0, free_indices)
        lower[column] = full_lower.index_select(0, free_indices)
        upper[column] = full_upper.index_select(0, free_indices)
        state_offset += reduce(mul, spec.shape, 1)

    bounded = torch.isfinite(lower) | torch.isfinite(upper)
    return scale, state_index, lower, upper, bounded


def _state_coordinates(
    values: Values,
    problem: Problem,
    state_index: torch.Tensor,
) -> torch.Tensor:
    flat = torch.cat(
        tuple(values[spec.name].reshape(*_batch_shape(values, problem), -1) for spec in problem.vars),
        dim=-1,
    )
    mapped = state_index >= 0
    gathered = flat.index_select(-1, state_index.clamp(min=0))
    return torch.where(mapped, gathered, torch.zeros_like(gathered))


def _robustify(
    residual: torch.Tensor,
    problem: Problem,
    default_kernel: RobustKernel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    costs: list[torch.Tensor] = []
    row_weights: list[torch.Tensor] = []
    for item in problem.residuals:
        rows = residual[..., problem.row_offsets[item.name]]
        groups = rows.reshape(*rows.shape[:-1], item.residual.dim // item.group_size, item.group_size)
        squared_norm = groups.square().sum(dim=-1)
        kernel = item.kernel if item.kernel is not None else default_kernel
        costs.append(kernel.rho(squared_norm).sum(dim=-1))
        group_weight = kernel.weight(squared_norm)
        row_weights.append(group_weight.unsqueeze(-1).expand(*group_weight.shape, item.group_size).reshape(*rows.shape))
    cost = torch.stack(costs, dim=-1).sum(dim=-1)
    weight = torch.cat(row_weights, dim=-1)
    scale = torch.sqrt(weight.clamp(min=0.0))
    return cost, weight, scale


def _robust_decrease(
    current: torch.Tensor,
    candidate: torch.Tensor,
    problem: Problem,
    default_kernel: RobustKernel,
) -> torch.Tensor:
    """Accumulate per-group decreases without subtracting two large totals."""
    decreases: list[torch.Tensor] = []
    for item in problem.residuals:
        row_slice = problem.row_offsets[item.name]
        shape = (*current.shape[:-1], item.residual.dim // item.group_size, item.group_size)
        current_groups = current[..., row_slice].reshape(shape)
        candidate_groups = candidate[..., row_slice].reshape(shape)
        kernel = item.kernel if item.kernel is not None else default_kernel
        current_rho = kernel.rho(current_groups.square().sum(dim=-1))
        candidate_rho = kernel.rho(candidate_groups.square().sum(dim=-1))
        decreases.append((current_rho - candidate_rho).sum(dim=-1))
    return torch.stack(decreases, dim=-1).sum(dim=-1)


def _active_mask(
    values: Values,
    gradient: torch.Tensor,
    problem: Problem,
    state_index: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    bounded: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    coordinate = _state_coordinates(values, problem, state_index)
    at_lower = coordinate <= lower + tolerance
    at_upper = coordinate >= upper - tolerance
    fixed = bounded & (lower == upper)
    outward = (at_lower & (gradient > 0.0)) | (at_upper & (gradient < 0.0))
    return bounded & (outward | fixed)


def _projected_gradient(
    values: Values,
    gradient: torch.Tensor,
    problem: Problem,
    batch_shape: tuple[int, ...],
    bounded: torch.Tensor,
) -> torch.Tensor:
    projected = _retract_reduced(values, -gradient, problem, batch_shape)
    # P(x - g) - x is the projected *descent* direction; negate it so the
    # stored vector follows the ordinary-gradient sign on bounded axes. Group
    # manifolds are unbounded under VarSpec and can wrap a large tangent
    # through log(exp(.)); retain their raw gradient instead.
    projected_box_gradient = -_difference_reduced(values, projected, problem, batch_shape)
    return torch.where(bounded, projected_box_gradient, gradient)


def _linearize_model(
    values: Values,
    problem: Problem,
    *,
    decision: LinearizationDecision,
    kernel: RobustKernel,
    jacobian_strategy: JacobianStrategy,
    state_index: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    bounded: torch.Tensor,
    bound_tolerance: float,
    create_graph: bool = False,
) -> _LinearizedLeastSquares:
    batch_shape = _batch_shape(values, problem)
    residual = problem._residual_prevalidated(values, batch_shape=batch_shape)
    cost, robust_weights, row_scale = _robustify(residual, problem, kernel)
    common_finite = (
        torch.isfinite(residual).all(dim=-1)
        & torch.isfinite(cost)
        & torch.isfinite(robust_weights).all(dim=-1)
        & (robust_weights >= 0.0).all(dim=-1)
        & (robust_weights <= 1.0).all(dim=-1)
    )

    if decision.used == "dense":
        jacobian_raw = problem._dense_jacobian_prevalidated(
            values,
            batch_shape=batch_shape,
            strategy=jacobian_strategy,
            create_graph=create_graph,
        )
        jacobian = jacobian_raw * row_scale.unsqueeze(-1)
        weighted_residual = residual * row_scale
        gradient = (jacobian.mT @ weighted_residual.unsqueeze(-1)).squeeze(-1)
        normal = jacobian.mT @ jacobian
        normal_diagonal = normal.diagonal(dim1=-2, dim2=-1)
        operators = _JacobianOperators(
            jvp_fn=lambda tangent: (jacobian @ tangent.unsqueeze(-1)).squeeze(-1),
            vjp_fn=lambda cotangent: (jacobian.mT @ cotangent.unsqueeze(-1)).squeeze(-1),
            normal_diagonal_tensor=normal_diagonal,
        )
        representation: torch.Tensor | BlockBandedMatrix | None = normal
        linearization_finite = torch.isfinite(jacobian).all(dim=(-2, -1))
    elif decision.used == "banded" or problem.temporal_analysis.direct_eligible:
        structured = problem._structured_normal_prevalidated(
            values,
            batch_shape=batch_shape,
            row_scale=row_scale,
            residual=residual,
            create_graph=create_graph,
        )
        gradient = structured.gradient
        normal_diagonal = structured.normal_diagonal
        operators = _JacobianOperators(
            jvp_fn=structured.jvp,
            vjp_fn=structured.vjp,
            normal_diagonal_tensor=normal_diagonal,
        )
        representation = structured.normal if decision.used == "banded" else None
        linearization_finite = structured.finite
    else:
        zero = residual.new_zeros(*batch_shape, problem.tangent_dim_total)

        def tangent_residual(delta: torch.Tensor) -> torch.Tensor:
            perturbed = _retract_reduced(values, delta, problem, batch_shape)
            raw = problem._residual_prevalidated(perturbed, batch_shape=batch_shape)
            return raw * row_scale

        _weighted_at_zero, pullback = torch.func.vjp(tangent_residual, zero)  # type: ignore[misc]

        def jvp(tangent: torch.Tensor) -> torch.Tensor:
            return torch.func.jvp(tangent_residual, (zero,), (tangent,))[1]

        def vjp(cotangent: torch.Tensor) -> torch.Tensor:
            return pullback(cotangent)[0]

        weighted_residual = residual * row_scale
        gradient = vjp(weighted_residual)
        basis = torch.eye(
            problem.tangent_dim_total,
            dtype=zero.dtype,
            device=zero.device,
        ).reshape(problem.tangent_dim_total, *([1] * len(batch_shape)), problem.tangent_dim_total)
        basis = basis.expand(problem.tangent_dim_total, *batch_shape, problem.tangent_dim_total)
        columns = torch.vmap(jvp)(basis).movedim(0, -1)
        normal_diagonal = columns.square().sum(dim=-2)
        operators = _JacobianOperators(
            jvp_fn=jvp,
            vjp_fn=vjp,
            normal_diagonal_tensor=normal_diagonal,
        )
        representation = None
        linearization_finite = torch.isfinite(normal_diagonal).all(dim=-1)

    grad_norm = _max_abs(gradient)
    projected_gradient = _projected_gradient(values, gradient, problem, batch_shape, bounded)
    projected_grad_norm = _max_abs(projected_gradient)
    active = _active_mask(
        values,
        gradient,
        problem,
        state_index,
        lower,
        upper,
        bounded,
        bound_tolerance,
    )
    finite = common_finite & linearization_finite & torch.isfinite(gradient).all(dim=-1)
    return _LinearizedLeastSquares(
        residual,
        robust_weights,
        cost,
        gradient,
        normal_diagonal,
        operators,
        representation,
        grad_norm,
        projected_gradient,
        projected_grad_norm,
        active,
        finite,
    )


def _terminal_status(model: _LinearizedLeastSquares, gtol: float) -> torch.Tensor:
    active_kkt = model.active_mask.any(dim=-1) & (model.grad_norm > gtol)
    satisfies_kkt = model.finite & (model.projected_grad_norm <= gtol)
    running = torch.full_like(model.projected_grad_norm, LMStatus.RUNNING.value, dtype=torch.int8)
    converged = torch.full_like(running, LMStatus.CONVERGED.value)
    stalled = torch.full_like(running, LMStatus.STALLED_AT_BOUNDS.value)
    failed = torch.full_like(running, LMStatus.FAILED.value)
    success = torch.where(active_kkt, stalled, converged)
    return torch.where(~model.finite, failed, torch.where(satisfies_kkt, success, running))


@dataclass(frozen=True)
class LevenbergMarquardt:
    """Batched projected active-set LM over named-block ``Problem``/``Values``.

    Hyperparameters are frozen.  All mutable per-element quantities live in
    :class:`LMState`.  ``update`` preserves a possible explicit unrolled
    oracle; ``run`` is the detached default driver and performs at most one
    host-side terminal check per iteration. It is capture-ready by
    construction; private ``GraphExecutor`` CUDA tests cover fixed update
    groups, while the public driver remains eager. ``block_step_limits`` optionally caps the
    physical tangent norm of named variable blocks before every retraction;
    state-space bounds remain the responsibility of :class:`VarSpec`.
    """

    max_iter: int = 50
    gtol: float = 1e-6
    xtol: float = 1e-9
    ftol: float = 1e-9
    damping_parameter: float = 1e-4
    mu_min: float = 1e-12
    mu_max: float = float(2**32)
    increase_factor_max: float = float(2**32)
    bound_tolerance: float = 1e-7
    linear_solver: LinearSolver | None = None
    linearization: LinearizationMode = "auto"
    kernel: RobustKernel = field(default_factory=L2)
    jacobian_strategy: JacobianStrategy = "auto"
    fixed_damping: bool = False
    block_step_limits: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("max_iter must be a non-negative int")
        if not isinstance(self.fixed_damping, bool):
            raise TypeError("fixed_damping must be a static bool")
        _validate_block_step_limits(self.block_step_limits)
        for name in (
            "gtol",
            "xtol",
            "ftol",
            "damping_parameter",
            "mu_min",
            "mu_max",
            "increase_factor_max",
            "bound_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                raise TypeError(f"{name} is a frozen static hyperparameter, not a tensor")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite Python number")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.mu_min <= 0.0 or self.mu_min > self.mu_max:
            raise ValueError("require 0 < mu_min <= mu_max")
        if self.increase_factor_max < 2.0:
            raise ValueError("increase_factor_max must be >= 2")
        if self.jacobian_strategy not in {
            "auto",
            "analytic",
            "jacrev",
            "jacfwd",
            "finite_difference",
        }:
            raise ValueError("solver jacobian_strategy must be auto/analytic/jacrev/jacfwd/finite_difference")
        if self.linearization not in {"auto", "dense", "structured", "matrix_free"}:
            raise ValueError("linearization must be auto/dense/structured/matrix_free")
        if self.linear_solver is not None and not isinstance(self.linear_solver, LinearSolver):
            raise TypeError("linear_solver must implement solve(A, b, ridge=None)")
        if not isinstance(self.kernel, RobustKernel):
            raise TypeError("kernel must implement rho(squared_norm) and weight(squared_norm)")

    def resolve_linearization(  # noqa: PLR0911, PLR0912 - explicit routing matrix
        self,
        problem: Problem,
    ) -> LinearizationDecision:
        """Resolve one static dense/banded/operator route for ``problem``."""
        analysis = problem.temporal_analysis
        solver = self.linear_solver
        supported = frozenset({"dense"}) if solver is None else getattr(
            solver,
            "supported_systems",
            frozenset({"dense"}),
        )

        def decision(used: Literal["dense", "banded", "matrix_free"], reason, detail: str):
            return LinearizationDecision(
                requested=self.linearization,
                used=used,
                reason=reason,
                detail=detail,
            )

        if self.linearization == "dense":
            if solver is not None and "dense" not in supported:
                raise ValueError("incompatible_solver: forced dense linearization requires a dense linear solver")
            return decision(
                "dense",
                LinearizationReason.FORCED_DENSE,
                "dense linearization was requested explicitly",
            )

        if self.linearization == "structured":
            if not analysis.direct_eligible:
                raise ValueError(f"{analysis.reason.value}: {analysis.detail}")
            if solver is not None and "banded" not in supported:
                raise ValueError("incompatible_solver: structured linearization requires a banded linear solver")
            return decision(
                "banded",
                LinearizationReason.ELIGIBLE_BANDED,
                "validated temporal blocks use the banded route",
            )

        if self.linearization == "matrix_free":
            if not analysis.operator_eligible:
                raise ValueError(f"{analysis.reason.value}: {analysis.detail}")
            if solver is not None and "operator" not in supported:
                raise ValueError("incompatible_solver: matrix_free linearization requires an operator linear solver")
            return decision(
                "matrix_free",
                LinearizationReason.EXPLICIT_MATRIX_FREE,
                "matrix-free linearization was requested",
            )

        if solver is None:
            if analysis.direct_eligible:
                return decision(
                    "banded",
                    LinearizationReason.ELIGIBLE_BANDED,
                    "validated temporal blocks use the automatic banded route",
                )
            return decision("dense", analysis.reason, analysis.detail)

        if analysis.direct_eligible and "banded" in supported:
            return decision(
                "banded",
                LinearizationReason.ELIGIBLE_BANDED,
                "explicit solver accepts the validated banded system",
            )
        if "dense" in supported:
            reason = LinearizationReason.EXPLICIT_DENSE_SOLVER if analysis.direct_eligible else analysis.reason
            detail = (
                "explicit solver is dense-only; retaining the dense correctness route"
                if analysis.direct_eligible
                else analysis.detail
            )
            return decision("dense", reason, detail)
        if analysis.operator_eligible and "operator" in supported:
            return decision(
                "matrix_free",
                LinearizationReason.EXPLICIT_MATRIX_FREE,
                "explicit solver accepts the normal operator",
            )
        raise ValueError(
            "incompatible_solver: explicit linear solver supports none of the eligible "
            f"systems {sorted(supported)}"
        )

    def _resolved_linear_solver(self, decision: LinearizationDecision) -> _ResolvedLinearSolver:
        if self.linear_solver is not None:
            return self.linear_solver
        if decision.used == "banded":
            return BandedCholesky()
        if decision.used == "matrix_free":
            return NormalCG()
        return Cholesky()

    def init_state(
        self,
        values: Values,
        problem: Problem,
        *,
        create_graph: bool = False,
    ) -> LMState:
        """Validate a solve boundary and evaluate its initial linearization."""
        if not isinstance(create_graph, bool):
            raise TypeError("create_graph must be a static bool")
        problem.require_least_squares(type(self).__name__)
        if not problem.residuals:
            raise ValueError(f"{type(self).__name__} requires at least one residual vector")
        limited_names = {name for name, _limit in self.block_step_limits}
        unknown_step_limits = limited_names - {spec.name for spec in problem.vars}
        if unknown_step_limits:
            raise ValueError(f"block_step_limits contain unknown variable names {sorted(unknown_step_limits)}")
        fixed_step_limits = {spec.name for spec in problem.vars if spec.name in limited_names and spec.free_dim == 0}
        if fixed_step_limits:
            raise ValueError(
                "block_step_limits require blocks with at least one free tangent "
                f"coordinate; fully fixed {sorted(fixed_step_limits)}"
            )
        if problem.tangent_dim_total <= 0:
            raise ValueError(f"{type(self).__name__} requires at least one free tangent coordinate")
        batch_shape = problem._validate_values(values)
        decision = self.resolve_linearization(problem)
        scale, state_index, lower, upper, bounded = _static_layout(values, problem)
        model = _linearize_model(
            values,
            problem,
            decision=decision,
            kernel=self.kernel,
            jacobian_strategy=self.jacobian_strategy,
            state_index=state_index,
            lower=lower,
            upper=upper,
            bounded=bounded,
            bound_tolerance=self.bound_tolerance,
            create_graph=create_graph,
        )
        diagonal_max = (model.normal_diagonal * scale.square()).amax(dim=-1)
        mu = (self.damping_parameter * diagonal_max).clamp(min=self.mu_min, max=self.mu_max)
        status = _terminal_status(model, self.gtol)
        converged = (status == LMStatus.CONVERGED.value) | (status == LMStatus.STALLED_AT_BOUNDS.value)
        zeros = model.cost.new_zeros(batch_shape)
        return LMState(
            residual=model.residual,
            robust_weights=model.robust_weights,
            cost=model.cost,
            mu=mu,
            increase_factor=model.cost.new_full(batch_shape, 2.0),
            gain_ratio=zeros,
            gradient=model.gradient,
            grad_norm=model.grad_norm,
            projected_gradient=model.projected_gradient,
            projected_grad_norm=model.projected_grad_norm,
            step_norm=zeros,
            relative_decrease=zeros,
            active_mask=model.active_mask,
            factorization_ok=torch.ones_like(model.cost, dtype=torch.bool),
            linear_solve_iterations=torch.zeros_like(status, dtype=torch.int64),
            linear_solve_residual_norm=zeros,
            linear_solve_relative_residual=zeros,
            linear_solve_status=torch.full_like(status, LinearSolveStatus.SUCCESS.value),
            previous_linear_step=model.gradient.new_zeros(*batch_shape, problem.tangent_dim_total),
            converged=converged,
            implicit_valid=(status == LMStatus.CONVERGED.value) & model.finite,
            status=status,
            iterations=torch.zeros_like(status, dtype=torch.int64),
            scale=scale,
            bound_state_index=state_index,
            bound_lower=lower,
            bound_upper=upper,
            bounded_mask=bounded,
        )

    def _solve_step(
        self,
        model: _LinearizedLeastSquares,
        state: LMState,
        problem: Problem,
        decision: LinearizationDecision,
    ) -> tuple[torch.Tensor, LinearSolveResult]:
        scaled_gradient = model.gradient * state.scale
        movable = (~model.active_mask).to(dtype=model.gradient.dtype)
        diagonal = state.mu.unsqueeze(-1) * movable + (1.0 - movable)
        rhs = -scaled_gradient * movable

        if decision.used == "dense":
            if not isinstance(model.normal, torch.Tensor):
                raise RuntimeError("dense linearization did not produce a dense normal matrix")
            scaled_normal = model.normal * state.scale.unsqueeze(-1) * state.scale.unsqueeze(-2)
            restricted = scaled_normal * movable.unsqueeze(-1) * movable.unsqueeze(-2)
            dense_system = restricted.clone()
            dense_system.diagonal(dim1=-2, dim2=-1).add_(diagonal)
            system: torch.Tensor | BlockBandedMatrix | NormalOperator = dense_system
        elif decision.used == "banded":
            if not isinstance(model.normal, BlockBandedMatrix):
                raise RuntimeError("banded linearization did not produce block-banded normal storage")
            system = model.normal.scaled_restricted(
                state.scale,
                movable,
                diagonal,
            )
        else:
            coordinate = state.scale * movable
            approximate_diagonal = model.normal_diagonal * coordinate.square() + diagonal
            safe_diagonal = approximate_diagonal.clamp_min(torch.finfo(model.gradient.dtype).tiny)
            time_length = problem.temporal_analysis.time_length
            reduced_width = problem.temporal_analysis.reduced_width
            block_shape = (
                (time_length, reduced_width)
                if time_length is not None and reduced_width is not None
                else None
            )
            system = NormalOperator(
                size=problem.tangent_dim_total,
                matvec=lambda vector: coordinate * model.operators.normal_matvec(coordinate * vector)
                + diagonal * vector,
                preconditioner=lambda vector: vector / safe_diagonal,
                block_shape=block_shape,
            )

        solver = self._resolved_linear_solver(decision)
        informative = getattr(solver, "solve_with_info", None)
        if informative is not None:
            initial = state.previous_linear_step if getattr(solver, "supports_initial", False) else None
            result = informative(system, rhs, ridge=None, initial=initial)
        else:
            raw_step = cast(LinearSolver, solver).solve(system, rhs, ridge=None)
            finite = torch.isfinite(raw_step).all(dim=-1)
            if isinstance(system, torch.Tensor):
                linear_residual = (system @ raw_step.unsqueeze(-1)).squeeze(-1) - rhs
            elif isinstance(system, BlockBandedMatrix):
                linear_residual = system.matvec(raw_step) - rhs
            else:
                linear_residual = system(raw_step) - rhs
            residual_norm = torch.linalg.vector_norm(linear_residual, dim=-1)
            rhs_norm = torch.linalg.vector_norm(rhs, dim=-1)
            relative = residual_norm / rhs_norm.clamp_min(torch.finfo(rhs.dtype).tiny)
            finite = finite & torch.isfinite(residual_norm) & torch.isfinite(relative)
            status = torch.where(
                finite,
                torch.full_like(finite, LinearSolveStatus.SUCCESS.value, dtype=torch.int8),
                torch.full_like(finite, LinearSolveStatus.NONFINITE.value, dtype=torch.int8),
            )
            result = LinearSolveResult(
                solution=torch.where(finite.unsqueeze(-1), raw_step, torch.zeros_like(raw_step)),
                converged=finite,
                finite=finite,
                ok=finite,
                iterations=torch.zeros_like(finite, dtype=torch.int64),
                residual_norm=residual_norm,
                relative_residual=relative,
                status=status,
            )
        scaled_step = torch.where(
            result.ok.unsqueeze(-1),
            torch.nan_to_num(result.solution),
            torch.zeros_like(result.solution),
        )
        return state.scale * scaled_step * movable, result

    def update(  # noqa: PLR0915 - one fixed-work tensor program keeps acceptance auditable
        self,
        values: Values,
        state: LMState,
        problem: Problem,
        *,
        create_graph: bool = False,
    ) -> tuple[Values, LMState]:
        """Apply one pure, sync-free, fixed-shape batched LM update.

        ``create_graph=True`` is the explicit small-problem unrolled oracle;
        the default step and :meth:`run` retain no Jacobian graph.
        """
        batch_shape = _batch_shape(values, problem)
        decision = self.resolve_linearization(problem)
        model = _linearize_model(
            values,
            problem,
            decision=decision,
            kernel=self.kernel,
            jacobian_strategy=self.jacobian_strategy,
            state_index=state.bound_state_index,
            lower=state.bound_lower,
            upper=state.bound_upper,
            bounded=state.bounded_mask,
            bound_tolerance=self.bound_tolerance,
            create_graph=create_graph,
        )
        terminal_now = _terminal_status(model, self.gtol)
        was_running = state.status == LMStatus.RUNNING.value
        terminal_from_model = was_running & (terminal_now != LMStatus.RUNNING.value)
        current_status = torch.where(terminal_from_model, terminal_now, state.status)
        movable_element = current_status == LMStatus.RUNNING.value

        lm_step, linear_result = self._solve_step(model, state, problem, decision)
        factorization_ok = linear_result.ok
        lm_step = _limit_block_step_norms(lm_step, problem, self.block_step_limits)
        lm_values = _retract_reduced(values, lm_step, problem, batch_shape)
        lm_actual_step = _difference_reduced(values, lm_values, problem, batch_shape)
        lm_jp = model.operators.jvp(lm_actual_step)
        lm_prediction = -((model.gradient * lm_actual_step).sum(dim=-1) + 0.5 * lm_jp.square().sum(dim=-1))

        pg1_proposal = _limit_block_step_norms(
            -model.gradient,
            problem,
            self.block_step_limits,
        )
        pg1_values = _retract_reduced(values, pg1_proposal, problem, batch_shape)
        pg1_step = _difference_reduced(values, pg1_values, problem, batch_shape)
        pg1_h = model.operators.normal_matvec(pg1_step)
        pg_denominator = (pg1_step * pg1_h).sum(dim=-1).clamp(min=torch.finfo(model.cost.dtype).eps)
        pg_beta = (-(model.gradient * pg1_step).sum(dim=-1) / pg_denominator).clamp(min=0.0, max=1.0)
        pg_proposal = _limit_block_step_norms(
            pg_beta.unsqueeze(-1) * pg1_step,
            problem,
            self.block_step_limits,
        )
        pg_values = _retract_reduced(values, pg_proposal, problem, batch_shape)
        pg_actual_step = _difference_reduced(values, pg_values, problem, batch_shape)
        pg_jp = model.operators.jvp(pg_actual_step)
        pg_prediction = -((model.gradient * pg_actual_step).sum(dim=-1) + 0.5 * pg_jp.square().sum(dim=-1))

        pg_better = torch.isfinite(pg_prediction) & (pg_prediction > lm_prediction) & (pg_prediction > 0.0)
        selected_values = _blend_values(pg_better, pg_values, lm_values)
        selected_step = torch.where(pg_better.unsqueeze(-1), pg_actual_step, lm_actual_step)
        prediction = torch.where(pg_better, pg_prediction, lm_prediction)
        valid_prediction = torch.isfinite(prediction) & (prediction > 0.0)
        selected_values = _blend_values(valid_prediction & factorization_ok, selected_values, values)
        selected_step = torch.where(
            (valid_prediction & factorization_ok).unsqueeze(-1),
            selected_step,
            torch.zeros_like(selected_step),
        )

        candidate_residual = problem._residual_prevalidated(selected_values, batch_shape=batch_shape)
        candidate_cost, candidate_weights, _candidate_row_scale = _robustify(
            candidate_residual,
            problem,
            self.kernel,
        )
        actual_decrease = _robust_decrease(
            model.residual,
            candidate_residual,
            problem,
            self.kernel,
        )
        gain_ratio = actual_decrease / prediction.clamp(min=torch.finfo(model.cost.dtype).eps)
        candidate_finite = (
            torch.isfinite(candidate_residual).all(dim=-1)
            & torch.isfinite(candidate_cost)
            & torch.isfinite(candidate_weights).all(dim=-1)
            & (candidate_weights >= 0.0).all(dim=-1)
            & (candidate_weights <= 1.0).all(dim=-1)
        )
        accept = (
            movable_element & model.finite & candidate_finite & factorization_ok & valid_prediction & (gain_ratio > 0.0)
        )
        next_values = _blend_values(accept, selected_values, values)
        next_residual = torch.where(accept.unsqueeze(-1), candidate_residual, model.residual)
        next_weights = torch.where(accept.unsqueeze(-1), candidate_weights, model.robust_weights)
        next_cost = torch.where(accept, candidate_cost, model.cost)

        accepted_multiplier = torch.maximum(
            torch.full_like(gain_ratio, 1.0 / 3.0),
            1.0 - (2.0 * gain_ratio - 1.0).pow(3),
        )
        mu_accept = (state.mu * accepted_multiplier).clamp(min=self.mu_min, max=self.mu_max)
        mu_reject = (state.mu * state.increase_factor).clamp(
            min=self.mu_min,
            max=self.mu_max,
        )
        factor_accept = torch.full_like(state.increase_factor, 2.0)
        factor_reject = (2.0 * state.increase_factor).clamp(max=self.increase_factor_max)
        if self.fixed_damping:
            mu_updated = state.mu
            factor_updated = state.increase_factor
        else:
            mu_updated = torch.where(accept, mu_accept, mu_reject)
            factor_updated = torch.where(accept, factor_accept, factor_reject)
        next_mu = torch.where(movable_element, mu_updated, state.mu)
        next_factor = torch.where(movable_element, factor_updated, state.increase_factor)

        # A failure below the cap escalates *to* the cap and must get one
        # attempt there before becoming terminal.  Testing ``next_mu`` would
        # fail an element one iteration too early.
        failed_factor = movable_element & ~factorization_ok & (state.mu >= self.mu_max)
        failed_model = movable_element & ~model.finite
        failure_status = torch.where(
            failed_factor | failed_model,
            torch.full_like(current_status, LMStatus.FAILED.value),
            current_status,
        )
        taken_step = torch.where(accept.unsqueeze(-1), selected_step, torch.zeros_like(selected_step))
        step_norm = _max_abs(taken_step)
        relative_decrease = torch.where(
            accept,
            actual_decrease / model.cost.abs().clamp(min=torch.finfo(model.cost.dtype).eps),
            torch.zeros_like(actual_decrease),
        )
        # Step/decrease tolerances are valid numerical stops for unbounded
        # least squares.  A finite box must earn a successful status through
        # the projected-gradient KKT check, even when progress is tiny.
        tolerance_converged = (
            accept & ~state.bounded_mask.any() & ((step_norm <= self.xtol) | (relative_decrease <= self.ftol))
        )
        next_status = torch.where(
            tolerance_converged,
            torch.full_like(failure_status, LMStatus.CONVERGED.value),
            failure_status,
        )
        converged = (next_status == LMStatus.CONVERGED.value) | (next_status == LMStatus.STALLED_AT_BOUNDS.value)
        iterations = state.iterations + movable_element.to(dtype=state.iterations.dtype)
        next_state = state._replace(
            residual=next_residual,
            robust_weights=next_weights,
            cost=next_cost,
            mu=next_mu,
            increase_factor=next_factor,
            gain_ratio=torch.where(movable_element, gain_ratio, state.gain_ratio),
            gradient=model.gradient,
            grad_norm=model.grad_norm,
            projected_gradient=model.projected_gradient,
            projected_grad_norm=model.projected_grad_norm,
            step_norm=torch.where(movable_element, step_norm, state.step_norm),
            relative_decrease=torch.where(
                movable_element,
                relative_decrease,
                state.relative_decrease,
            ),
            active_mask=model.active_mask,
            factorization_ok=torch.where(movable_element, factorization_ok, state.factorization_ok),
            linear_solve_iterations=torch.where(
                movable_element,
                linear_result.iterations,
                state.linear_solve_iterations,
            ),
            linear_solve_residual_norm=torch.where(
                movable_element,
                linear_result.residual_norm,
                state.linear_solve_residual_norm,
            ),
            linear_solve_relative_residual=torch.where(
                movable_element,
                linear_result.relative_residual,
                state.linear_solve_relative_residual,
            ),
            linear_solve_status=torch.where(
                movable_element,
                linear_result.status,
                state.linear_solve_status,
            ),
            previous_linear_step=torch.where(
                (movable_element & factorization_ok).unsqueeze(-1),
                linear_result.solution,
                state.previous_linear_step,
            ),
            converged=converged,
            # Step/decrease termination is a valid numerical stop, but only
            # the final-point KKT evaluation may make it implicit-eligible.
            implicit_valid=(next_status == LMStatus.CONVERGED.value)
            & (terminal_now == LMStatus.CONVERGED.value)
            & model.finite,
            status=next_status,
            iterations=iterations,
        )
        return next_values, next_state

    def finalize(
        self,
        values: Values,
        state: LMState,
        problem: Problem,
        *,
        create_graph: bool = False,
    ) -> tuple[Values, LMState]:
        """Canonical final-point evaluation for consistent terminal artifacts."""
        problem.require_least_squares(type(self).__name__)
        problem._validate_values(values)
        decision = self.resolve_linearization(problem)
        model = _linearize_model(
            values,
            problem,
            decision=decision,
            kernel=self.kernel,
            jacobian_strategy=self.jacobian_strategy,
            state_index=state.bound_state_index,
            lower=state.bound_lower,
            upper=state.bound_upper,
            bounded=state.bounded_mask,
            bound_tolerance=self.bound_tolerance,
            create_graph=create_graph,
        )
        terminal = _terminal_status(model, self.gtol)
        preserve_terminal_failure = (state.status == LMStatus.FAILED.value) | (state.status == LMStatus.MAXITER.value)
        same_terminal_artifacts = (
            (model.residual == state.residual).all(dim=-1)
            & (model.robust_weights == state.robust_weights).all(dim=-1)
            & (model.cost == state.cost)
        )
        preserve_tolerance_convergence = (
            (state.status == LMStatus.CONVERGED.value) & ~state.implicit_valid & same_terminal_artifacts
        )
        status = torch.where(
            terminal != LMStatus.RUNNING.value,
            terminal,
            torch.where(
                preserve_terminal_failure | preserve_tolerance_convergence,
                state.status,
                torch.full_like(state.status, LMStatus.RUNNING.value),
            ),
        )
        converged = (status == LMStatus.CONVERGED.value) | (status == LMStatus.STALLED_AT_BOUNDS.value)
        return values, state._replace(
            residual=model.residual,
            robust_weights=model.robust_weights,
            cost=model.cost,
            gradient=model.gradient,
            grad_norm=model.grad_norm,
            projected_gradient=model.projected_gradient,
            projected_grad_norm=model.projected_grad_norm,
            active_mask=model.active_mask,
            converged=converged,
            implicit_valid=(terminal == LMStatus.CONVERGED.value) & model.finite,
            status=status,
        )

    def _warm_start(self, values: Values, state: LMState, problem: Problem) -> LMState:
        """Refresh changed targets while retaining the previous damping state."""
        fresh = self.init_state(values, problem)
        for name in ("mu", "increase_factor"):
            retained = getattr(state, name)
            expected = getattr(fresh, name)
            if retained.shape != expected.shape:
                raise ValueError("warm-start state batch shape does not match Values")
            if retained.dtype != expected.dtype or retained.device != expected.device:
                raise ValueError(
                    "warm-start damping tensors must share Values' dtype and device; "
                    f"{name} is ({retained.dtype}, {retained.device}) but expected "
                    f"({expected.dtype}, {expected.device})"
                )
        return fresh._replace(
            mu=state.mu.detach().clamp(min=self.mu_min, max=self.mu_max),
            increase_factor=state.increase_factor.detach().clamp(
                min=2.0,
                max=self.increase_factor_max,
            ),
        )

    def run(
        self,
        values: Values,
        problem: Problem,
        state: LMState | None = None,
    ) -> tuple[Values, LMState]:
        """Run the detached eager loop, optionally retaining warm-start damping."""
        problem.require_least_squares(type(self).__name__)
        current_values = detach_values(values)
        current_state = (
            self.init_state(current_values, problem)
            if state is None
            else self._warm_start(current_values, state, problem)
        )
        for _ in range(self.max_iter):
            if bool((current_state.status != LMStatus.RUNNING.value).all()):  # bench-ok: eager loop boundary sync
                break
            current_values, current_state = self.update(current_values, current_state, problem)
            current_values = detach_values(current_values)
            current_state = _detach_state(current_state)
        current_values, current_state = self.finalize(current_values, current_state, problem)
        running = current_state.status == LMStatus.RUNNING.value
        current_state = current_state._replace(
            status=torch.where(
                running,
                torch.full_like(current_state.status, LMStatus.MAXITER.value),
                current_state.status,
            ),
            converged=current_state.converged & ~running,
            implicit_valid=current_state.implicit_valid & ~running,
        )
        return detach_values(current_values), _detach_state(current_state)

    def solve(
        self,
        values: Values,
        problem: Problem,
        state: LMState | None = None,
        *,
        differentiate: Literal["detached", "implicit"] = "detached",
        implicit_config: ImplicitDiffConfig | None = None,
    ) -> tuple[Values, LMState]:
        """Solve with detached semantics by default or an implicit backward.

        ``differentiate="implicit"`` attaches a first-order custom autograd
        boundary to the detached terminal values. Gradients flow only to
        external tensors explicitly named by
        :attr:`Problem.differentiable_external_parameters`; the initial guess,
        warm-start state, bounds, masks, and solver hyperparameters never
        receive an implicit gradient. :meth:`run` remains the always-detached
        compatibility entry point.
        """
        if differentiate not in {"detached", "implicit"}:
            raise ValueError("differentiate must be 'detached' or 'implicit'")
        if differentiate == "detached" and implicit_config is not None:
            raise ValueError("implicit_config is only used with differentiate='implicit'")

        if differentiate == "implicit":
            from .implicit import validate_implicit_input_roles  # noqa: PLC0415

            validate_implicit_input_roles(values, problem)

        terminal_values, terminal_state = self.run(values, problem, state)
        if differentiate == "detached":
            return terminal_values, terminal_state

        from .implicit import attach_implicit_gradients  # noqa: PLC0415

        decision = self.resolve_linearization(problem)
        differentiable_values = attach_implicit_gradients(
            terminal_values,
            terminal_state,
            problem,
            default_kernel=self.kernel,
            forward_linearization=decision.used,
            config=implicit_config,
        )
        return differentiable_values, terminal_state


@dataclass(frozen=True)
class GaussNewton(LevenbergMarquardt):
    """Gauss--Newton preset using LM's guarded acceptance update."""

    damping_parameter: float = 1e-9
    fixed_damping: bool = True
