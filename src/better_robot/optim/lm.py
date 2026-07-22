"""Batched projected Gauss--Newton and Levenberg--Marquardt solvers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
import math
from operator import mul
from typing import Literal, NamedTuple, TypeAlias, cast

import torch

from ..data_model.joint_models import JointComposite, JointModel
from .kernels import _group_rows
from .solvers import (
    BandedCholesky,
    Cholesky,
    LinearSolver,
)
from .implicit import ImplicitDiffConfig
from .optimizers import Optimizer, OptimizerInfo, OptimizerStatus
from .problem import JacobianStrategy, Problem, _EvaluationBundle, _check_strategy, _detach
from .temporal import BlockBandedMatrix, LinearizationReason, _warn_missing_temporal_blocks
from .variables import RobotVariable, _joint_layout
from .utils import _blend_values, _state_coordinates

LinearizationMode: TypeAlias = Literal["auto", "dense", "structured"]
_TensorValues: TypeAlias = dict[str, torch.Tensor]
# State records


@dataclass(frozen=True)
class LinearizationDecision:
    """Resolved dense or block-banded linearization route."""

    requested: LinearizationMode
    used: Literal["dense", "banded"]
    reason: LinearizationReason
    detail: str


_ResolvedLinearSolver: TypeAlias = LinearSolver | BandedCholesky | Cholesky


class _LMIterationState(NamedTuple):
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
    bound_state_index: torch.Tensor
    bound_lower: torch.Tensor
    bound_upper: torch.Tensor
    bounded_mask: torch.Tensor


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
    evaluation: _EvaluationBundle


# Tangent layout and projection


def _max_abs(vector: torch.Tensor) -> torch.Tensor:
    return vector.abs().amax(dim=-1)


def _unsafe_translation_v_indices(joint: JointModel) -> tuple[int, ...]:
    if joint.kind == "free_flyer":
        return (0, 1, 2)
    if not isinstance(joint, JointComposite):
        return ()
    indices: list[int] = []
    offset = 0
    for child in joint.sub_joints:
        indices.extend(offset + index for index in _unsafe_translation_v_indices(child))
        offset += child.nv
    return tuple(indices)


def _split_step(step: torch.Tensor, problem: Problem) -> _TensorValues:
    return {spec.name: step[..., problem.column_offsets[spec.name]] for spec in problem.vars}


def _difference_tangent(
    x0: _TensorValues,
    x1: _TensorValues,
    problem: Problem,
) -> torch.Tensor:
    difference = problem.difference(x0, x1)
    return torch.cat(tuple(difference[variable.name] for variable in problem.vars), dim=-1)


def _retract_tangent(
    values: _TensorValues,
    step: torch.Tensor,
    problem: Problem,
) -> _TensorValues:
    return problem.retract(values, _split_step(step, problem))


def _project_step(values: _TensorValues, step: torch.Tensor, problem: Problem) -> tuple[_TensorValues, torch.Tensor]:
    proposed = _retract_tangent(values, step, problem)
    return proposed, _difference_tangent(values, proposed, problem)


def _detach_state(state: _LMIterationState) -> _LMIterationState:
    return _LMIterationState(*(tensor.detach() for tensor in state))


def _static_layout(  # noqa: PLR0912, PLR0915 - handles the finite supported variable layouts
    values: _TensorValues,
    problem: Problem,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build state-box-to-tangent maps once at the public boundary."""
    exemplar = values[problem.vars[0].name]
    nt = problem.tangent_dim_total
    state_index = torch.full((nt,), -1, dtype=torch.long, device=exemplar.device)
    lower = exemplar.new_full((nt,), -torch.inf)
    upper = exemplar.new_full((nt,), torch.inf)
    state_offset = 0

    for spec in problem.vars:
        column = problem.column_offsets[spec.name]
        tangent_dim = spec.tangent_dim()
        full_mapping = torch.full(
            (tangent_dim,),
            -1,
            dtype=torch.long,
            device=exemplar.device,
        )
        full_lower = exemplar.new_full((tangent_dim,), -torch.inf)
        full_upper = exemplar.new_full((tangent_dim,), torch.inf)
        unsafe_tangent = torch.zeros(tangent_dim, dtype=torch.bool, device=exemplar.device)
        if not isinstance(spec, RobotVariable):
            full_mapping = torch.arange(
                state_offset,
                state_offset + tangent_dim,
                dtype=torch.long,
                device=exemplar.device,
            )
            if spec.bounds is not None:
                full_lower = spec.bounds.lower.reshape(-1)
                full_upper = spec.bounds.upper.reshape(-1)
        elif isinstance(spec, RobotVariable):
            model = spec.model
            q_for_v = [-1] * model.nv
            unsafe_v: list[int] = []
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
                layout = _joint_layout(joint)
                for local_v, local_q in enumerate(layout.q_for_v):
                    if local_q >= 0:
                        q_for_v[iv + local_v] = iq + local_q
                unsafe_v.extend(iv + index for index in _unsafe_translation_v_indices(joint))
                unsafe_q.extend(iq + index for index in layout.unsafe_q)
            event_count = reduce(mul, spec.shape[:-1], 1)
            unsafe_per_knot = torch.zeros(model.nv, dtype=torch.bool, device=exemplar.device)
            unsafe_indices = torch.tensor(unsafe_v, dtype=torch.long, device=exemplar.device)
            unsafe_q_indices = torch.tensor(unsafe_q, dtype=torch.long, device=exemplar.device)
            unsafe_per_knot[unsafe_indices] = True
            unsafe_tangent = unsafe_per_knot.repeat(event_count)
            for event in range(event_count):
                for local_v, local_q in enumerate(q_for_v):
                    tangent_index = event * model.nv + local_v
                    if local_q < 0:
                        continue
                    full_mapping[tangent_index] = state_offset + event * model.nq + local_q
                    if spec.bounds is not None:
                        full_lower[tangent_index] = spec.bounds.lower[local_q]
                        full_upper[tangent_index] = spec.bounds.upper[local_q]
                if spec.bounds is not None:
                    full_lower[event * model.nv + unsafe_indices] = spec.bounds.lower[unsafe_q_indices]
                    full_upper[event * model.nv + unsafe_indices] = spec.bounds.upper[unsafe_q_indices]

        free_mapping = spec.gather_tangent(full_mapping)
        free_lower = spec.gather_tangent(full_lower)
        free_upper = spec.gather_tangent(full_upper)
        unsafe_finite = spec.gather_tangent(unsafe_tangent) & (torch.isfinite(free_lower) | torch.isfinite(free_upper))
        if bool(unsafe_finite.any()):  # bench-ok: init-only static bounds validation
            raise ValueError(
                f"RobotVariable {spec.name!r} has a finite free-flyer "
                "translation bound. World-axis state boxes are not axis-aligned "
                "in the right-local SE(3) tangent; defer this constraint or "
                "express it as a residual."
            )
        state_index[column] = free_mapping
        lower[column] = free_lower
        upper[column] = free_upper
        state_offset += reduce(mul, spec.shape, 1)

    bounded = torch.isfinite(lower) | torch.isfinite(upper)
    return state_index, lower, upper, bounded


# Robust objective and IRLS


def _robustify(
    evaluation: _EvaluationBundle,
    problem: Problem,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_weights: list[torch.Tensor] = []
    row_scales: list[torch.Tensor] = []
    for index, item in enumerate(problem.residuals):
        rows = evaluation.rows[..., problem.row_offsets[item.name]]
        groups = _group_rows(rows, item.group_size)
        squared_norm = groups.square().sum(dim=-1)
        if item.is_inactive():
            group_weight = torch.ones_like(squared_norm)
        else:
            kernel = problem._kernel(item, detach_tensors=False)
            group_weight = kernel.weight(squared_norm)
        outer = evaluation.active_groups[index].to(rows.dtype) * evaluation.coefficients[index]
        group_scale = torch.sqrt(group_weight.clamp(min=0.0) * outer)
        row_weights.append(group_weight.unsqueeze(-1).expand(*group_weight.shape, item.group_size).reshape(*rows.shape))
        row_scales.append(group_scale.unsqueeze(-1).expand(*group_scale.shape, item.group_size).reshape(*rows.shape))
    weight = torch.cat(row_weights, dim=-1)
    scale = torch.cat(row_scales, dim=-1)
    return evaluation.cost, weight, scale


def _robust_decrease(
    current: _EvaluationBundle,
    candidate: _EvaluationBundle,
) -> torch.Tensor:
    """Accumulate per-group decreases without subtracting two large totals."""
    decreases = [
        (current_cost - candidate_cost).sum(dim=-1)
        for current_cost, candidate_cost in zip(current.group_costs, candidate.group_costs, strict=True)
    ]
    return torch.stack(decreases, dim=-1).sum(dim=-1)


def _robust_finite(residual: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (
        torch.isfinite(residual).all(dim=-1)
        & torch.isfinite(cost)
        & torch.isfinite(weights).all(dim=-1)
        & (weights >= 0.0).all(dim=-1)
        & (weights <= 1.0).all(dim=-1)
    )


# Linearization and routing


def _active_mask(
    values: _TensorValues,
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
    values: _TensorValues,
    gradient: torch.Tensor,
    problem: Problem,
    bounded: torch.Tensor,
) -> torch.Tensor:
    projected = _retract_tangent(values, -gradient, problem)
    # P(x - g) - x is the projected *descent* direction; negate it so the
    # stored vector follows the ordinary-gradient sign on bounded axes. Group
    # group variables are unbounded and can wrap a large tangent
    # through log(exp(.)); retain their raw gradient instead.
    projected_box_gradient = -_difference_tangent(values, projected, problem)
    return torch.where(bounded, projected_box_gradient, gradient)


def _linearize_model(
    values: _TensorValues,
    problem: Problem,
    solver: LevenbergMarquardt,
    decision: LinearizationDecision,
    bounds: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    create_graph: bool = False,
) -> _LinearizedLeastSquares:
    state_index, lower, upper, bounded = bounds
    evaluation = problem._evaluate_at(values)
    residual = evaluation.rows
    cost, robust_weights, row_scale = _robustify(evaluation, problem)
    common_finite = _robust_finite(residual, cost, robust_weights)

    if decision.used == "dense":
        jacobian_raw = problem._dense_jacobian_at(
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
        evaluation,
    )


def _terminal_status(model: _LinearizedLeastSquares, gtol: float) -> torch.Tensor:
    active_kkt = model.active_mask.any(dim=-1) & (model.grad_norm > gtol)
    satisfies_kkt = model.finite & (model.projected_grad_norm <= gtol)
    running = torch.full_like(model.projected_grad_norm, OptimizerStatus.RUNNING.value, dtype=torch.int8)
    converged = torch.full_like(running, OptimizerStatus.CONVERGED.value)
    stalled = torch.full_like(running, OptimizerStatus.STALLED_AT_BOUNDS.value)
    failed = torch.full_like(running, OptimizerStatus.FAILED.value)
    success = torch.where(active_kkt, stalled, converged)
    return torch.where(~model.finite, failed, torch.where(satisfies_kkt, success, running))


# Solver drivers


class LevenbergMarquardt(Optimizer):
    """Batched projected active-set LM that owns one problem."""

    def __init__(
        self,
        problem: Problem,
        *,
        solver: Literal["auto"] | LinearSolver = "auto",
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        step_tolerance: float = 1e-9,
        relative_tolerance: float = 1e-9,
        damping: float = 1e-4,
        mu_min: float = 1e-12,
        mu_max: float = float(2**32),
        increase_factor_max: float = float(2**32),
        bound_tolerance: float = 1e-7,
        linearization: LinearizationMode = "auto",
        jacobian_strategy: JacobianStrategy = "auto",
        fixed_damping: bool = False,
    ) -> None:
        super().__init__(problem, max_iterations=max_iterations, tolerance=tolerance)
        if solver == "auto":
            linear_solver = None
        elif isinstance(solver, LinearSolver):
            linear_solver = solver
        else:
            raise TypeError("solver must be 'auto' or implement solve(A, b, ridge=None)")
        self.gtol = self.tolerance
        self.xtol = step_tolerance
        self.ftol = relative_tolerance
        self.damping_parameter = damping
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.increase_factor_max = increase_factor_max
        self.bound_tolerance = bound_tolerance
        self.linear_solver = linear_solver
        self.linearization = linearization
        self.jacobian_strategy = jacobian_strategy
        self.fixed_damping = fixed_damping
        self._state: _LMIterationState | None = None
        self._seen_update_serial = problem._update_serial
        if not isinstance(self.fixed_damping, bool):
            raise TypeError("fixed_damping must be a static bool")
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
        _check_strategy(self.jacobian_strategy, False)
        if self.linearization not in {"auto", "dense", "structured"}:
            raise ValueError("linearization must be auto/dense/structured")
        problem._freeze()

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
            if analysis.reason is LinearizationReason.MISSING_TEMPORAL_BLOCKS:
                _warn_missing_temporal_blocks(problem)
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
            if reason is LinearizationReason.MISSING_TEMPORAL_BLOCKS:
                _warn_missing_temporal_blocks(problem)
            return choice("dense", reason, detail)
        raise ValueError(
            f"incompatible_solver: explicit linear solver supports none of the eligible systems {sorted(supported)}"
        )

    def _resolved_linear_solver(self, decision: LinearizationDecision) -> _ResolvedLinearSolver:
        return self.linear_solver or (BandedCholesky() if decision.used == "banded" else Cholesky())

    def _init_state(
        self,
        values: _TensorValues,
        problem: Problem,
        *,
        create_graph: bool = False,
    ) -> _LMIterationState:
        """Validate a solve boundary and evaluate its initial linearization."""
        if not isinstance(create_graph, bool):
            raise TypeError("create_graph must be a static bool")
        if not problem.residuals:
            raise ValueError(f"{type(self).__name__} requires at least one residual vector")
        if problem.tangent_dim_total <= 0:
            raise ValueError(f"{type(self).__name__} requires at least one free tangent coordinate")
        batch_shape = problem._validate_trainable_values(values)
        self._has_bounds = any(variable.bounds is not None for variable in problem.vars)
        decision = self.resolve_linearization(problem)
        state_index, lower, upper, bounded = _static_layout(values, problem)
        model = _linearize_model(values, problem, self, decision, (state_index, lower, upper, bounded), create_graph)
        diagonal_max = model.normal_diagonal.amax(dim=-1)
        mu = (self.damping_parameter * diagonal_max).clamp(min=self.mu_min, max=self.mu_max)
        status = _terminal_status(model, self.gtol)
        converged = (status == OptimizerStatus.CONVERGED.value) | (status == OptimizerStatus.STALLED_AT_BOUNDS.value)
        zeros = model.cost.new_zeros(batch_shape)
        return _LMIterationState(
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
            implicit_valid=(status == OptimizerStatus.CONVERGED.value) & model.finite,
            status=status,
            iterations=torch.zeros_like(status, dtype=torch.int64),
            bound_state_index=state_index,
            bound_lower=lower,
            bound_upper=upper,
            bounded_mask=bounded,
        )

    def _solve_step(
        self,
        model: _LinearizedLeastSquares,
        state: _LMIterationState,
        decision: LinearizationDecision,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        movable = (~model.active_mask).to(dtype=model.gradient.dtype)
        diagonal = state.mu.unsqueeze(-1) * movable + (1.0 - movable)
        rhs = -model.gradient * movable

        if decision.used == "dense":
            if not isinstance(model.normal, torch.Tensor):
                raise RuntimeError("dense linearization did not produce a dense normal matrix")
            restricted = model.normal * movable.unsqueeze(-1) * movable.unsqueeze(-2)
            dense_system = restricted.clone()
            dense_system.diagonal(dim1=-2, dim2=-1).add_(diagonal)
            system: torch.Tensor | BlockBandedMatrix = dense_system
        else:
            if not isinstance(model.normal, BlockBandedMatrix):
                raise RuntimeError("banded linearization did not produce block-banded normal storage")
            system = model.normal.restricted(movable, diagonal)
        solver = self._resolved_linear_solver(decision)
        informative = getattr(solver, "solve_with_info", None)
        if informative is not None:
            result = informative(system, rhs, ridge=None, initial=None)
            raw_step, ok = result.solution, result.ok
        else:
            raw_step = cast(LinearSolver, solver).solve(system, rhs, ridge=None)
            ok = torch.isfinite(raw_step).all(dim=-1)
        safe_step = torch.where(ok.unsqueeze(-1), torch.nan_to_num(raw_step), torch.zeros_like(raw_step))
        return safe_step * movable, ok

    def _update(  # noqa: PLR0915 - one fixed-work tensor program keeps acceptance auditable
        self,
        values: _TensorValues,
        state: _LMIterationState,
        problem: Problem,
        *,
        create_graph: bool = False,
    ) -> tuple[_TensorValues, _LMIterationState]:
        """Apply one pure, sync-free, fixed-shape batched LM _update.

        ``create_graph=True`` is the explicit small-problem unrolled oracle;
        the default step and :meth:`run` retain no Jacobian graph.
        """
        decision = self.resolve_linearization(problem)
        bounds = (state.bound_state_index, state.bound_lower, state.bound_upper, state.bounded_mask)
        model = _linearize_model(values, problem, self, decision, bounds, create_graph)
        terminal_now = _terminal_status(model, self.gtol)
        was_running = state.status == OptimizerStatus.RUNNING.value
        terminal_from_model = was_running & (terminal_now != OptimizerStatus.RUNNING.value)
        current_status = torch.where(terminal_from_model, terminal_now, state.status)
        movable_element = current_status == OptimizerStatus.RUNNING.value

        lm_step, factorization_ok = self._solve_step(model, state, decision)
        lm_values, lm_actual_step = _project_step(values, lm_step, problem)
        lm_jp = model.operators.jvp(lm_actual_step)
        lm_prediction = -((model.gradient * lm_actual_step).sum(dim=-1) + 0.5 * lm_jp.square().sum(dim=-1))

        if self._has_bounds:
            _pg1_values, pg1_step = _project_step(values, -model.gradient, problem)
            pg1_h = model.operators.normal_matvec(pg1_step)
            pg_denominator = (pg1_step * pg1_h).sum(dim=-1).clamp(min=torch.finfo(model.cost.dtype).eps)
            pg_beta = (-(model.gradient * pg1_step).sum(dim=-1) / pg_denominator).clamp(min=0.0, max=1.0)
            pg_values, pg_actual_step = _project_step(values, pg_beta.unsqueeze(-1) * pg1_step, problem)
            pg_jp = model.operators.jvp(pg_actual_step)
            pg_prediction = -((model.gradient * pg_actual_step).sum(dim=-1) + 0.5 * pg_jp.square().sum(dim=-1))
            pg_better = torch.isfinite(pg_prediction) & (pg_prediction > lm_prediction) & (pg_prediction > 0.0)
            selected_values = _blend_values(pg_better, pg_values, lm_values)
            selected_step = torch.where(pg_better.unsqueeze(-1), pg_actual_step, lm_actual_step)
            prediction = torch.where(pg_better, pg_prediction, lm_prediction)
        else:
            selected_values, selected_step, prediction = lm_values, lm_actual_step, lm_prediction
        valid_prediction = torch.isfinite(prediction) & (prediction > 0.0)
        selected_values = _blend_values(valid_prediction & factorization_ok, selected_values, values)
        selected_step = torch.where(
            (valid_prediction & factorization_ok).unsqueeze(-1),
            selected_step,
            torch.zeros_like(selected_step),
        )

        candidate_evaluation = problem._evaluate_at(selected_values)
        candidate_residual = candidate_evaluation.rows
        candidate_cost, candidate_weights, _candidate_row_scale = _robustify(candidate_evaluation, problem)
        actual_decrease = _robust_decrease(model.evaluation, candidate_evaluation)
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
            torch.full_like(current_status, OptimizerStatus.FAILED.value),
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
            torch.full_like(failure_status, OptimizerStatus.CONVERGED.value),
            failure_status,
        )
        converged = (next_status == OptimizerStatus.CONVERGED.value) | (
            next_status == OptimizerStatus.STALLED_AT_BOUNDS.value
        )
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
            implicit_valid=(next_status == OptimizerStatus.CONVERGED.value)
            & (terminal_now == OptimizerStatus.CONVERGED.value)
            & model.finite,
            status=next_status,
            iterations=iterations,
        )
        return next_values, next_state

    def _finalize(
        self,
        values: _TensorValues,
        state: _LMIterationState,
        problem: Problem,
        *,
        create_graph: bool = False,
    ) -> tuple[_TensorValues, _LMIterationState]:
        """Canonical final-point evaluation for consistent terminal artifacts."""
        decision = self.resolve_linearization(problem)
        bounds = (state.bound_state_index, state.bound_lower, state.bound_upper, state.bounded_mask)
        model = _linearize_model(values, problem, self, decision, bounds, create_graph)
        terminal = _terminal_status(model, self.gtol)
        preserve_terminal_failure = (state.status == OptimizerStatus.FAILED.value) | (
            state.status == OptimizerStatus.MAXITER.value
        )
        same_terminal_artifacts = (
            (model.residual == state.residual).all(dim=-1)
            & (model.robust_weights == state.robust_weights).all(dim=-1)
            & (model.cost == state.cost)
        )
        preserve_tolerance_convergence = (
            (state.status == OptimizerStatus.CONVERGED.value) & ~state.implicit_valid & same_terminal_artifacts
        )
        status = torch.where(
            terminal != OptimizerStatus.RUNNING.value,
            terminal,
            torch.where(
                preserve_terminal_failure | preserve_tolerance_convergence,
                state.status,
                torch.full_like(state.status, OptimizerStatus.RUNNING.value),
            ),
        )
        converged = (status == OptimizerStatus.CONVERGED.value) | (status == OptimizerStatus.STALLED_AT_BOUNDS.value)
        return values, state._replace(
            residual=model.residual,
            robust_weights=model.robust_weights,
            cost=model.cost,
            gradient=model.gradient,
            grad_norm=model.grad_norm,
            projected_grad_norm=model.projected_grad_norm,
            active_mask=model.active_mask,
            converged=converged,
            implicit_valid=(terminal == OptimizerStatus.CONVERGED.value) & model.finite,
            status=status,
        )

    def _warm_start(self, values: _TensorValues, state: _LMIterationState, problem: Problem) -> _LMIterationState:
        """Refresh changed targets while retaining the previous damping state."""
        fresh = self._init_state(values, problem)
        for name in ("mu", "increase_factor"):
            retained = getattr(state, name)
            expected = getattr(fresh, name)
            if retained.shape != expected.shape:
                raise ValueError("warm-start state batch shape does not match _TensorValues")
            if retained.dtype != expected.dtype or retained.device != expected.device:
                raise ValueError(
                    "warm-start damping tensors must share _TensorValues' dtype and device; "
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

    def reset(self) -> None:
        self._state = None
        self._seen_update_serial = self.problem._update_serial

    def _ensure_state(self) -> _LMIterationState:
        values = _detach(self.problem._trainable_values())
        if self._state is None:
            self._state = self._init_state(values, self.problem)
        elif self._seen_update_serial != self.problem._update_serial:
            try:
                self._state = self._warm_start(values, self._state, self.problem)
            except ValueError:
                self._state = self._init_state(values, self.problem)
        self._seen_update_serial = self.problem._update_serial
        return self._state

    @staticmethod
    def _info(state: _LMIterationState) -> OptimizerInfo:
        return OptimizerInfo(state.status.detach(), state.iterations.detach(), state.cost.detach())

    def _initial_info(self) -> OptimizerInfo:
        return self._info(self._ensure_state())

    def step(self) -> OptimizerInfo:
        state = self._ensure_state()
        if bool((state.status != OptimizerStatus.RUNNING.value).all()):  # bench-ok: eager step boundary
            return self._info(state)
        values = _detach(self.problem._trainable_values())
        values, state = self._update(values, state, self.problem)
        values = _detach(values)
        state = _detach_state(state)
        self.problem._set_trainable(values, detach=True)
        self._state = state
        return self._info(state)

    def optimize(
        self,
        *,
        verbose: bool = False,
        differentiate: Literal["implicit"] | None = None,
        implicit_config: ImplicitDiffConfig | None = None,
    ) -> OptimizerInfo:
        if differentiate not in {None, "implicit"}:
            raise ValueError("differentiate must be None or 'implicit'")
        if differentiate is None and implicit_config is not None:
            raise ValueError("implicit_config must only be used with differentiate='implicit'")

        state = self._ensure_state()
        for _ in range(self.max_iterations):
            if bool((state.status != OptimizerStatus.RUNNING.value).all()):  # bench-ok: eager boundary sync
                break
            info = self.step()
            assert self._state is not None
            state = self._state
            if verbose:
                converged = int(info.converged.sum().detach().cpu())  # bench-ok: eager verbose reporting
                print(
                    f"iteration={int(info.iterations.max().detach().cpu())} "  # bench-ok: eager verbose reporting
                    f"cost={float(info.cost.sum().detach().cpu()):.6g} "  # bench-ok: eager verbose reporting
                    f"converged={converged}/{info.converged.numel()}"
                )

        values = _detach(self.problem._trainable_values())
        values, state = self._finalize(values, state, self.problem)
        running = state.status == OptimizerStatus.RUNNING.value
        state = state._replace(
            status=torch.where(
                running,
                torch.full_like(state.status, OptimizerStatus.MAXITER.value),
                state.status,
            ),
            converged=state.converged & ~running,
            implicit_valid=state.implicit_valid & ~running,
        )
        self._state = _detach_state(state)
        self.problem._set_trainable(_detach(values), detach=True)

        if differentiate == "implicit":
            from .implicit import _attach_implicit_gradients  # noqa: PLC0415

            decision = self.resolve_linearization(self.problem)
            differentiable = _attach_implicit_gradients(
                _detach(values),
                state,
                self.problem,
                forward_linearization=decision.used,
                config=implicit_config,
            )
            self.problem._set_trainable(differentiable)
        return self._info(self._state)


class GaussNewton(LevenbergMarquardt):
    """Gauss--Newton preset using LM's guarded acceptance step."""

    def __init__(self, problem: Problem, **kwargs) -> None:
        kwargs.setdefault("damping", 1e-9)
        kwargs.setdefault("fixed_damping", True)
        super().__init__(problem, **kwargs)
