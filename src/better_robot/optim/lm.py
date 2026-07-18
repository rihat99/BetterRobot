"""Batched projected Gauss--Newton and Levenberg--Marquardt solvers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from functools import reduce
import math
from operator import mul
from typing import Literal, NamedTuple, TypeAlias, cast

import torch

from .kernels import L2, RobustKernel, _group_rows
from .solvers import (
    BandedCholesky,
    Cholesky,
    LinearSolver,
)
from ._solver_common import _blend_values, _state_coordinates
from .implicit import ImplicitDiffConfig
from .manifolds import Euclidean, RobotConfig, _joint_coordinate_layout
from .problem import JacobianStrategy, Problem
from .temporal import BlockBandedMatrix, LinearizationReason
from .variables import Values, detach_values

LinearizationMode: TypeAlias = Literal["auto", "dense", "structured"]


@dataclass(frozen=True)
class LinearizationDecision:
    """Resolved dense or block-banded linearization route."""

    requested: LinearizationMode
    used: Literal["dense", "banded"]
    reason: LinearizationReason
    detail: str


_ResolvedLinearSolver: TypeAlias = LinearSolver | BandedCholesky | Cholesky


class LMStatus(IntEnum):
    """Per-element terminal status stored as ``torch.int8``."""

    RUNNING = 0
    CONVERGED = 1
    STALLED_AT_BOUNDS = 2
    MAXITER = 3
    FAILED = 4


class LMState(NamedTuple):
    """Fixed-structure tensor state for arbitrary leading batch axes."""

    residual: torch.Tensor
    robust_weights: torch.Tensor
    cost: torch.Tensor
    mu: torch.Tensor
    increase_factor: torch.Tensor
    gain_ratio: torch.Tensor
    gradient: torch.Tensor
    grad_norm: torch.Tensor
    projected_grad_norm: torch.Tensor
    step_norm: torch.Tensor
    relative_decrease: torch.Tensor
    active_mask: torch.Tensor
    factorization_ok: torch.Tensor
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


class _JacobianOperators(NamedTuple):
    jvp: Callable[[torch.Tensor], torch.Tensor]
    normal_matvec: Callable[[torch.Tensor], torch.Tensor]


class _LinearizedLeastSquares(NamedTuple):
    residual: torch.Tensor
    robust_weights: torch.Tensor
    cost: torch.Tensor
    gradient: torch.Tensor
    normal_diagonal: torch.Tensor
    operators: _JacobianOperators
    normal: torch.Tensor | BlockBandedMatrix | None
    grad_norm: torch.Tensor
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
) -> torch.Tensor:
    full = problem.difference(x0, x1)
    return torch.cat(
        tuple(spec.gather_tangent(full[spec.name]) for spec in problem.vars),
        dim=-1,
    )


def _retract_reduced(
    values: Values,
    step: torch.Tensor,
    problem: Problem,
) -> Values:
    return problem.retract(values, _split_step(step, problem))


def _project_step(
    values: Values, step: torch.Tensor, problem: Problem, limits: tuple[tuple[str, float], ...]
) -> tuple[Values, torch.Tensor]:
    proposed = _retract_reduced(values, _limit_block_step_norms(step, problem, limits), problem)
    return proposed, _difference_reduced(values, proposed, problem)


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


def _robustify(
    residual: torch.Tensor,
    problem: Problem,
    default_kernel: RobustKernel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    costs: list[torch.Tensor] = []
    row_weights: list[torch.Tensor] = []
    for item in problem.residuals:
        rows = residual[..., problem.row_offsets[item.name]]
        groups = _group_rows(rows, item.group_size)
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
        current_groups = _group_rows(current[..., row_slice], item.group_size)
        candidate_groups = _group_rows(candidate[..., row_slice], item.group_size)
        kernel = item.kernel if item.kernel is not None else default_kernel
        current_rho = kernel.rho(current_groups.square().sum(dim=-1))
        candidate_rho = kernel.rho(candidate_groups.square().sum(dim=-1))
        decreases.append((current_rho - candidate_rho).sum(dim=-1))
    return torch.stack(decreases, dim=-1).sum(dim=-1)


def _robust_finite(residual: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (
        torch.isfinite(residual).all(dim=-1)
        & torch.isfinite(cost)
        & torch.isfinite(weights).all(dim=-1)
        & (weights >= 0.0).all(dim=-1)
        & (weights <= 1.0).all(dim=-1)
    )


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
    bounded: torch.Tensor,
) -> torch.Tensor:
    projected = _retract_reduced(values, -gradient, problem)
    # P(x - g) - x is the projected *descent* direction; negate it so the
    # stored vector follows the ordinary-gradient sign on bounded axes. Group
    # manifolds are unbounded under VarSpec and can wrap a large tangent
    # through log(exp(.)); retain their raw gradient instead.
    projected_box_gradient = -_difference_reduced(values, projected, problem)
    return torch.where(bounded, projected_box_gradient, gradient)


def _linearize_model(
    values: Values,
    problem: Problem,
    solver: LevenbergMarquardt,
    decision: LinearizationDecision,
    bounds: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    create_graph: bool = False,
) -> _LinearizedLeastSquares:
    state_index, lower, upper, bounded = bounds
    residual = problem.residual(values)
    cost, robust_weights, row_scale = _robustify(residual, problem, solver.kernel)
    common_finite = _robust_finite(residual, cost, robust_weights)

    if decision.used == "dense":
        jacobian_raw = problem.dense_jacobian(
            values,
            strategy=solver.jacobian_strategy,
            create_graph=create_graph,
        )
        jacobian = jacobian_raw * row_scale.unsqueeze(-1)
        weighted_residual = residual * row_scale
        gradient = (jacobian.mT @ weighted_residual.unsqueeze(-1)).squeeze(-1)
        normal = jacobian.mT @ jacobian
        normal_diagonal = normal.diagonal(dim1=-2, dim2=-1)
        operators = _JacobianOperators(
            lambda tangent: (jacobian @ tangent.unsqueeze(-1)).squeeze(-1),
            lambda tangent: (normal @ tangent.unsqueeze(-1)).squeeze(-1),
        )
        representation: torch.Tensor | BlockBandedMatrix | None = normal
        linearization_finite = torch.isfinite(jacobian).all(dim=(-2, -1))
    else:
        structured = problem.structured_normal(
            values,
            row_scale=row_scale,
            residual=residual,
            create_graph=create_graph,
        )
        gradient = structured.gradient
        normal_diagonal = structured.normal_diagonal
        operators = _JacobianOperators(structured.jvp, structured.normal_matvec)
        representation = structured.normal
        linearization_finite = structured.finite

    grad_norm = _max_abs(gradient)
    projected_gradient = _projected_gradient(values, gradient, problem, bounded)
    projected_grad_norm = _max_abs(projected_gradient)
    active = _active_mask(
        values,
        gradient,
        problem,
        state_index,
        lower,
        upper,
        bounded,
        solver.bound_tolerance,
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
    """Batched projected active-set LM over named-block problems."""

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
        if self.linearization not in {"auto", "dense", "structured"}:
            raise ValueError("linearization must be auto/dense/structured")
        if self.linear_solver is not None and not isinstance(self.linear_solver, LinearSolver):
            raise TypeError("linear_solver must implement solve(A, b, ridge=None)")
        if not isinstance(self.kernel, RobustKernel):
            raise TypeError("kernel must implement rho(squared_norm) and weight(squared_norm)")

    def resolve_linearization(self, problem: Problem) -> LinearizationDecision:
        """Resolve one static dense or block-banded route for ``problem``."""
        analysis = problem.temporal_analysis
        solver = self.linear_solver
        supported = frozenset({"dense"}) if solver is None else getattr(solver, "supported_systems", {"dense"})

        def choice(used: Literal["dense", "banded"], reason: LinearizationReason, detail: str):
            return LinearizationDecision(self.linearization, used, reason, detail)

        if self.linearization == "dense":
            if solver is not None and "dense" not in supported:
                raise ValueError("incompatible_solver: forced dense linearization requires a dense linear solver")
            return choice("dense", LinearizationReason.FORCED_DENSE, "dense linearization was requested explicitly")
        if self.linearization == "structured":
            if not analysis.direct_eligible:
                raise ValueError(f"{analysis.reason.value}: {analysis.detail}")
            if solver is not None and "banded" not in supported:
                raise ValueError("incompatible_solver: structured linearization requires a banded linear solver")
            return choice(
                "banded", LinearizationReason.ELIGIBLE_BANDED, "validated temporal blocks use the banded route"
            )
        if solver is None:
            if analysis.direct_eligible:
                return choice(
                    "banded",
                    LinearizationReason.ELIGIBLE_BANDED,
                    "validated temporal blocks use the automatic banded route",
                )
            return choice("dense", analysis.reason, analysis.detail)
        if analysis.direct_eligible and "banded" in supported:
            return choice(
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
            return choice("dense", reason, detail)
        raise ValueError(
            f"incompatible_solver: explicit linear solver supports none of the eligible systems {sorted(supported)}"
        )

    def _resolved_linear_solver(self, decision: LinearizationDecision) -> _ResolvedLinearSolver:
        return self.linear_solver or (BandedCholesky() if decision.used == "banded" else Cholesky())

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
        model = _linearize_model(values, problem, self, decision, (state_index, lower, upper, bounded), create_graph)
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
            projected_grad_norm=model.projected_grad_norm,
            step_norm=zeros,
            relative_decrease=zeros,
            active_mask=model.active_mask,
            factorization_ok=torch.ones_like(model.cost, dtype=torch.bool),
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
        decision: LinearizationDecision,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            system: torch.Tensor | BlockBandedMatrix = dense_system
        else:
            if not isinstance(model.normal, BlockBandedMatrix):
                raise RuntimeError("banded linearization did not produce block-banded normal storage")
            system = model.normal.scaled_restricted(
                state.scale,
                movable,
                diagonal,
            )
        solver = self._resolved_linear_solver(decision)
        informative = getattr(solver, "solve_with_info", None)
        if informative is not None:
            result = informative(system, rhs, ridge=None, initial=None)
            raw_step, ok = result.solution, result.ok
        else:
            raw_step = cast(LinearSolver, solver).solve(system, rhs, ridge=None)
            ok = torch.isfinite(raw_step).all(dim=-1)
        safe_step = torch.where(ok.unsqueeze(-1), torch.nan_to_num(raw_step), torch.zeros_like(raw_step))
        return state.scale * safe_step * movable, ok

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
        decision = self.resolve_linearization(problem)
        bounds = (state.bound_state_index, state.bound_lower, state.bound_upper, state.bounded_mask)
        model = _linearize_model(values, problem, self, decision, bounds, create_graph)
        terminal_now = _terminal_status(model, self.gtol)
        was_running = state.status == LMStatus.RUNNING.value
        terminal_from_model = was_running & (terminal_now != LMStatus.RUNNING.value)
        current_status = torch.where(terminal_from_model, terminal_now, state.status)
        movable_element = current_status == LMStatus.RUNNING.value

        lm_step, factorization_ok = self._solve_step(model, state, decision)
        lm_values, lm_actual_step = _project_step(values, lm_step, problem, self.block_step_limits)
        lm_jp = model.operators.jvp(lm_actual_step)
        lm_prediction = -((model.gradient * lm_actual_step).sum(dim=-1) + 0.5 * lm_jp.square().sum(dim=-1))

        _pg1_values, pg1_step = _project_step(values, -model.gradient, problem, self.block_step_limits)
        pg1_h = model.operators.normal_matvec(pg1_step)
        pg_denominator = (pg1_step * pg1_h).sum(dim=-1).clamp(min=torch.finfo(model.cost.dtype).eps)
        pg_beta = (-(model.gradient * pg1_step).sum(dim=-1) / pg_denominator).clamp(min=0.0, max=1.0)
        pg_values, pg_actual_step = _project_step(
            values, pg_beta.unsqueeze(-1) * pg1_step, problem, self.block_step_limits
        )
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

        candidate_residual = problem.residual(selected_values)
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
        candidate_finite = _robust_finite(candidate_residual, candidate_cost, candidate_weights)
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
            projected_grad_norm=model.projected_grad_norm,
            step_norm=torch.where(movable_element, step_norm, state.step_norm),
            relative_decrease=torch.where(
                movable_element,
                relative_decrease,
                state.relative_decrease,
            ),
            active_mask=model.active_mask,
            factorization_ok=torch.where(movable_element, factorization_ok, state.factorization_ok),
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
        problem._validate_values(values)
        decision = self.resolve_linearization(problem)
        bounds = (state.bound_state_index, state.bound_lower, state.bound_upper, state.bounded_mask)
        model = _linearize_model(values, problem, self, decision, bounds, create_graph)
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
        """Solve detached by default, or attach the guarded implicit backward."""
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
