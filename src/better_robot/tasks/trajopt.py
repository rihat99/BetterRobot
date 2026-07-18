"""Kinematic trajectory optimization as a named-block problem preset.

Residual items are adapted to one temporal ``q`` block. Solver ownership is
otherwise entirely in :mod:`better_robot.optim`: this module contains no
optimization loop or dense/structured dispatch of its own.

Only :class:`~better_robot.tasks.parameterization.KnotTrajectory` is a safe
robot parameterization today.  The numerical ``BSplineTrajectory`` utility is
still rejected because component-wise interpolation is not a robot-manifold
map and does not preserve bounds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, replace
import math
from typing import Any, Literal

import torch

from ..data_model.model import Model
from ..optim import (
    Bounds,
    JacobianStrategy,
    LevenbergMarquardt,
    LinearizationMode,
    LinearizationReason,
    Problem,
    ResidualItem,
    RobotConfig,
    RobotStateProvider,
    VarSpec,
)
from ..residuals.regularization import ReferenceTrajectoryResidual
from ..residuals.smoothness import AccelerationResidual, VelocityResidual
from ..residuals.temporal import TimeIndexedResidual
from .parameterization import KnotTrajectory
from .trajectory import Trajectory


class _ResidualAdapter:
    """Give one trajectory residual the item identity expected by ``Problem``.

    Optional analytic and temporal hooks are exposed by normal attribute
    delegation, so their absence remains structurally meaningful to
    ``Problem``.
    """

    def __init__(self, name: str, residual: Any) -> None:
        self.name = name
        self._residual = residual
        dim = getattr(residual, "dim", None)
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                f"Active cost item {name!r} must have a positive static residual dim; "
                "use a static-horizon named-block residual"
            )
        self.dim = dim
        reads = getattr(residual, "reads", None)
        if not isinstance(reads, tuple) or any(not isinstance(read, str) or not read for read in reads):
            raise TypeError(f"Residual item {name!r} must declare reads as tuple[str, ...]")
        self.reads = reads

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return self._residual(ctx)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._residual, name)


@dataclass
class TrajOptResult:
    """Result of an unbatched or independently batched trajectory solve.

    Historical unbatched calls retain ``trajectory.q.shape == (1, T, nq)``.
    Batched calls preserve every leading batch axis and return tensor-valued
    per-element iteration and convergence diagnostics. Route metadata records
    a requested ``auto``/``dense``/``structured`` mode and the selected
    ``dense`` or ``banded`` implementation.
    """

    trajectory: Trajectory
    residual: torch.Tensor
    iters: int | torch.Tensor
    converged: bool | torch.Tensor
    status: int | torch.Tensor
    model: Model
    linearization_requested: LinearizationMode
    linearization_used: Literal["dense", "banded"]
    linearization_reason: LinearizationReason
    linearization_detail: str


def _configuration_bounds(
    model: Model,
    exemplar: torch.Tensor,
    manifold: RobotConfig,
    lower: torch.Tensor | None,
    upper: torch.Tensor | None,
) -> Bounds | None:
    """Build optional state bounds while removing manifold coordinates."""
    if lower is None and upper is None:
        return None
    for name, limit in (("lower", lower), ("upper", upper)):
        if limit is not None:
            if not isinstance(limit, torch.Tensor) or not limit.is_floating_point():
                raise TypeError(f"{name} must be a floating torch.Tensor or None")
            if tuple(limit.shape) != (model.nq,):
                raise ValueError(f"{name} must have shape ({model.nq},), got {tuple(limit.shape)}")
    raw_lower = (
        torch.full((model.nq,), -torch.inf, dtype=exemplar.dtype, device=exemplar.device)
        if lower is None
        else lower.to(dtype=exemplar.dtype, device=exemplar.device)
    )
    raw_upper = (
        torch.full((model.nq,), torch.inf, dtype=exemplar.dtype, device=exemplar.device)
        if upper is None
        else upper.to(dtype=exemplar.dtype, device=exemplar.device)
    )
    box = manifold.box_mask.to(device=exemplar.device)
    return Bounds(
        lower=torch.where(box, raw_lower, torch.full_like(raw_lower, -torch.inf)),
        upper=torch.where(box, raw_upper, torch.full_like(raw_upper, torch.inf)),
    )


def _hemisphere_align_robot_trajectory(
    q: torch.Tensor,
    manifold: RobotConfig,
) -> torch.Tensor:
    """Clone and align free-flyer/spherical quaternion representatives."""
    aligned = q.clone()
    for coordinate_slice in manifold.unit_coordinate_slices:
        start = 0 if coordinate_slice.start is None else coordinate_slice.start
        stop = manifold.model.nq if coordinate_slice.stop is None else coordinate_slice.stop
        if stop - start != 4:
            continue
        quaternion = aligned[..., coordinate_slice]
        adjacent_dot = (quaternion[..., 1:, :] * quaternion[..., :-1, :]).sum(dim=-1)
        step_sign = torch.where(
            adjacent_dot < 0.0,
            -torch.ones_like(adjacent_dot),
            torch.ones_like(adjacent_dot),
        )
        first = torch.ones_like(quaternion[..., :1, 0])
        signs = torch.cat((first, step_sign), dim=-1).cumprod(dim=-1).unsqueeze(-1)
        aligned[..., coordinate_slice] = quaternion * signs
    return aligned


def _prepare_cost_residual(
    residual: Any,
    *,
    model: Model,
    horizon: int,
    item_name: str,
    manifold: RobotConfig,
) -> Any:
    """Clone one residual and fill static-horizon metadata."""
    prepared = copy(residual)
    declared_horizon = getattr(prepared, "horizon", None)
    if declared_horizon is not None and declared_horizon != horizon:
        raise ValueError(
            f"Residual item {item_name!r} declares horizon={declared_horizon}, but solve_trajopt horizon={horizon}"
        )

    if isinstance(prepared, (VelocityResidual, AccelerationResidual)):
        if horizon < 3:
            raise ValueError(f"Residual item {item_name!r} requires horizon >= 3, got {horizon}")
        prepared.horizon = horizon
        prepared.dim = (horizon - 2) * model.nv
    elif isinstance(prepared, TimeIndexedResidual):
        if not 0 <= prepared.t_idx < horizon:
            raise ValueError(f"Residual item {item_name!r} has t_idx={prepared.t_idx} outside horizon={horizon}")
        prepared.horizon = horizon
        prepared.inner = copy(prepared.inner)
        if getattr(prepared.inner, "model", False) is None and hasattr(prepared.inner, "model"):
            prepared.inner.model = model

    if getattr(prepared, "model", False) is None and hasattr(prepared, "model"):
        prepared.model = model

    if isinstance(prepared, ReferenceTrajectoryResidual):
        reference_spec = VarSpec(
            "q_reference",
            (horizon, model.nq),
            manifold=manifold,
            time_axis=0,
        )
        reference_spec.validate_value(prepared.q_ref)
        prepared.q_ref = _hemisphere_align_robot_trajectory(prepared.q_ref, manifold)
    return prepared


def _prepare_residuals(
    items: Sequence[ResidualItem],
    *,
    model: Model,
    horizon: int,
    manifold: RobotConfig,
) -> tuple[ResidualItem, ...]:
    residuals: list[ResidualItem] = []
    for item in items:
        prepared = _prepare_cost_residual(
            item.residual,
            model=model,
            horizon=horizon,
            item_name=item.name,
            manifold=manifold,
        )
        residuals.append(
            replace(
                item,
                residual=_ResidualAdapter(item.name, prepared),
            )
        )
    if not residuals:
        raise ValueError("residuals must contain at least one residual")
    return tuple(residuals)


def _public_diagnostics(
    state: Any,
) -> tuple[int | torch.Tensor, bool | torch.Tensor, int | torch.Tensor]:
    iterations = state.iterations
    converged = state.converged
    status = state.status
    if iterations.ndim == 0:
        return int(iterations), bool(converged), int(status)
    return iterations, converged, status


def solve_trajopt(  # noqa: PLR0913, PLR0915 - explicit facade policy stays auditable
    model: Model,
    *,
    horizon: int,
    dt: float,
    initial_q_traj: torch.Tensor,
    residuals: Sequence[ResidualItem],
    optimizer: LevenbergMarquardt | None = None,
    max_iter: int = 50,
    jacobian_strategy: JacobianStrategy = "auto",
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    parameterization: KnotTrajectory | None = None,
) -> TrajOptResult:
    """Solve one or an arbitrary leading batch of knot trajectories.

    ``initial_q_traj`` is shaped ``(B..., T, nq)``.  ``optimizer=None``
    constructs route-aware named-block LM with ``linearization="auto"``;
    pass ``LevenbergMarquardt(linearization="dense")`` for the dense oracle.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive int")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError("dt must be a finite positive Python number")
    if not isinstance(initial_q_traj, torch.Tensor) or not initial_q_traj.is_floating_point():
        raise TypeError("initial_q_traj must be a floating torch.Tensor")
    if initial_q_traj.ndim < 2:
        raise ValueError(f"initial_q_traj must end in (T, nq); got {tuple(initial_q_traj.shape)}")
    T, nq = initial_q_traj.shape[-2:]
    if T != horizon:
        raise ValueError(f"horizon={horizon} inconsistent with initial_q_traj T={T}")
    if nq != model.nq:
        raise ValueError(f"initial_q_traj.shape[-1]={nq} != model.nq={model.nq}")

    parameterization = KnotTrajectory() if parameterization is None else parameterization
    if not isinstance(parameterization, KnotTrajectory):
        raise NotImplementedError(
            "solve_trajopt supports only KnotTrajectory. The current "
            f"{type(parameterization).__name__} performs component-space "
            "Euclidean interpolation, which is not a manifold-safe robot "
            "trajectory map and cannot preserve state bounds. Robot B-spline "
            "support remains explicitly deferred; use KnotTrajectory."
        )
    if jacobian_strategy not in {"auto", "analytic", "jacrev", "jacfwd", "finite_difference"}:
        raise ValueError(
            f"Unknown jacobian_strategy {jacobian_strategy!r}; expected auto, analytic, "
            "jacrev, jacfwd, or finite_difference"
        )

    manifold = RobotConfig(model)
    bounds = _configuration_bounds(model, initial_q_traj, manifold, lower, upper)
    spec = VarSpec(
        "q",
        (horizon, model.nq),
        manifold=manifold,
        bounds=bounds,
        time_axis=0,
    )
    start = initial_q_traj.detach().clone()
    spec.validate_value(start)
    start = _hemisphere_align_robot_trajectory(start, manifold)
    start = manifold.project(start, bounds)

    prepared_residuals = _prepare_residuals(
        residuals,
        model=model,
        horizon=horizon,
        manifold=manifold,
    )
    problem = Problem(
        vars=(spec,),
        residuals=prepared_residuals,
        providers=(RobotStateProvider(model),),
    )

    solver = (
        LevenbergMarquardt(
            max_iter=max_iter,
            linearization="auto",
            jacobian_strategy=jacobian_strategy,
        )
        if optimizer is None
        else optimizer
    )
    if optimizer is not None and jacobian_strategy != "auto":
        solver = replace(
            solver,
            jacobian_strategy=jacobian_strategy,
        )
    decision = solver.resolve_linearization(problem)
    values, state = solver.run({"q": start}, problem)
    residual = problem.residual(values)
    iterations, converged, status = _public_diagnostics(state)

    q_result = _hemisphere_align_robot_trajectory(values["q"], manifold)
    batch_shape = tuple(q_result.shape[:-2])
    t_axis = torch.arange(horizon, dtype=q_result.dtype, device=q_result.device) * float(dt)
    if batch_shape:
        trajectory_q = q_result
        trajectory_t = t_axis.expand(*batch_shape, horizon)
    else:
        trajectory_q = q_result.unsqueeze(0)
        trajectory_t = t_axis.unsqueeze(0)
    trajectory = Trajectory(
        t=trajectory_t,
        q=trajectory_q,
        model_id=getattr(model, "id", -1),
    )
    return TrajOptResult(
        trajectory=trajectory,
        residual=residual,
        iters=iterations,
        converged=converged,
        status=status,
        model=model,
        linearization_requested=decision.requested,
        linearization_used=decision.used,
        linearization_reason=decision.reason,
        linearization_detail=decision.detail,
    )


__all__ = ["TrajOptResult", "solve_trajopt"]
