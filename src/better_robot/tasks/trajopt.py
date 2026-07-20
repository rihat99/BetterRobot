"""Kinematic trajectory optimization over an object-referenced problem."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Literal

import torch

from ..data_model.model import Model
from ..optim import (
    Bounds,
    JacobianStrategy,
    LevenbergMarquardt,
    LinearizationDecision,
    LinearizationMode,
    LinearizationReason,
    Optimizer,
    Problem,
    Residual,
    RobotVariable,
)
from .trajectory import Trajectory
from .utils import _hemisphere_align, _public_diagnostics


@dataclass
class TrajOptResult:
    """Solved trajectory, optimizer diagnostics, and linearization route."""

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


ResidualFactory = Callable[[RobotVariable], Residual]
OptimizerFactory = Callable[[Problem], Optimizer]


def _configuration_bounds(
    model: Model,
    exemplar: torch.Tensor,
    lower: torch.Tensor | None,
    upper: torch.Tensor | None,
) -> Bounds | None:
    if lower is None and upper is None:
        return None
    for name, limit in (("lower", lower), ("upper", upper)):
        if limit is not None:
            if not isinstance(limit, torch.Tensor) or not limit.is_floating_point():
                raise TypeError(f"{name} must be a floating torch.Tensor or None")
            if tuple(limit.shape) != (model.nq,):
                raise ValueError(f"{name} must have shape ({model.nq},), got {tuple(limit.shape)}")
    raw_lower = (
        exemplar.new_full((model.nq,), -torch.inf)
        if lower is None
        else lower.to(dtype=exemplar.dtype, device=exemplar.device)
    )
    raw_upper = (
        exemplar.new_full((model.nq,), torch.inf)
        if upper is None
        else upper.to(dtype=exemplar.dtype, device=exemplar.device)
    )
    layout = RobotVariable(model, exemplar[..., 0, :], trainable=False)
    box = layout.box_mask.to(device=exemplar.device)
    return Bounds(
        torch.where(box, raw_lower, torch.full_like(raw_lower, -torch.inf)),
        torch.where(box, raw_upper, torch.full_like(raw_upper, torch.inf)),
    )


def _trajectory_variable(
    model: Model,
    initial: torch.Tensor | RobotVariable,
    lower: torch.Tensor | None,
    upper: torch.Tensor | None,
) -> RobotVariable:
    if isinstance(initial, RobotVariable):
        if initial.model is not model:
            raise ValueError("initial_q_traj RobotVariable must reference the supplied model")
        if initial.time_axis != 0:
            raise ValueError("initial_q_traj RobotVariable must declare time_axis=0")
        if lower is not None or upper is not None:
            raise ValueError("lower/upper belong on a supplied RobotVariable; do not pass both")
        variable = initial
    else:
        if not isinstance(initial, torch.Tensor) or not initial.is_floating_point():
            raise TypeError("initial_q_traj must be a floating Tensor or RobotVariable")
        if initial.ndim < 2 or initial.shape[-1] != model.nq or initial.shape[-2] <= 0:
            raise ValueError(f"initial_q_traj must end in (T, {model.nq}) with T > 0; got {tuple(initial.shape)}")
        bounds = _configuration_bounds(model, initial, lower, upper)
        variable = RobotVariable(model, initial.detach().clone(), name="q", bounds=bounds, time_axis=0)
    variable.tensor = _hemisphere_align(variable.tensor, variable.unit_coordinate_slices)
    zeros = variable.tensor.new_zeros(*variable.batch_shape, variable.free_dim)
    variable.tensor = variable.retract(zeros)
    return variable


def _prepare_residuals(
    residuals: Sequence[Residual | ResidualFactory],
    q: RobotVariable,
) -> tuple[Residual, ...]:
    prepared: list[Residual] = []
    for declaration in residuals:
        item = declaration(q) if callable(declaration) and not isinstance(declaration, Residual) else declaration
        if not isinstance(item, Residual):
            raise TypeError(
                f"trajectory residual declarations must produce Residual objects, got {type(item).__name__}"
            )
        reachable = (*item.variables, *(variable for node in getattr(item, "nodes", ()) for variable in node.variables))
        if all(variable is not q for variable in reachable):
            raise ValueError(f"Residual {item.name!r} does not reference the trajectory RobotVariable")
        prepared.append(item)
    if not prepared:
        raise ValueError("residuals must contain at least one residual")
    return tuple(prepared)


def _non_lm_decision() -> LinearizationDecision:
    return LinearizationDecision(
        "dense",
        "dense",
        LinearizationReason.FORCED_DENSE,
        "the selected optimizer does not assemble a block-banded normal system",
    )


def solve_trajopt(  # noqa: PLR0913
    model: Model,
    *,
    dt: float,
    initial_q_traj: torch.Tensor | RobotVariable,
    residuals: Sequence[Residual | ResidualFactory],
    optimizer: OptimizerFactory | None = None,
    max_iter: int = 50,
    jacobian_strategy: JacobianStrategy = "auto",
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
) -> TrajOptResult:
    """Solve a knot trajectory using residual objects or ``q -> residual`` factories.

    A tensor input lets the facade construct ``RobotVariable(...,
    time_axis=0)``; passing that variable directly lets callers own the graph.
    ``optimizer`` is a factory because the optimizer must be created after the
    residual graph has produced its :class:`Problem`.
    """
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError("dt must be a finite positive Python number")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer")
    if jacobian_strategy not in {"auto", "analytic", "jacrev", "jacfwd", "finite_difference"}:
        raise ValueError(f"Unknown jacobian_strategy {jacobian_strategy!r}")
    q = _trajectory_variable(model, initial_q_traj, lower, upper)
    problem = Problem(_prepare_residuals(residuals, q))
    if optimizer is None:
        driver: Optimizer = LevenbergMarquardt(
            problem,
            max_iterations=max_iter,
            jacobian_strategy=jacobian_strategy,
        )
    else:
        if not callable(optimizer):
            raise TypeError(f"optimizer must be a callable factory, got {type(optimizer).__name__}")
        driver = optimizer(problem)
        if not isinstance(driver, Optimizer):
            raise TypeError(f"optimizer factory must return Optimizer, got {type(driver).__name__}")
        if driver.problem is not problem:
            raise ValueError("optimizer factory must return an Optimizer owning the supplied Problem")
    decision = driver.resolve_linearization(problem) if isinstance(driver, LevenbergMarquardt) else _non_lm_decision()
    info = driver.optimize()
    residual = problem.error()
    iterations, converged, status = _public_diagnostics((info,))

    q_result = _hemisphere_align(q.tensor, q.unit_coordinate_slices)
    batch_shape = tuple(q_result.shape[:-2])
    horizon = q.time_length
    t_axis = torch.arange(horizon, dtype=q_result.dtype, device=q_result.device) * float(dt)
    trajectory = Trajectory(
        t=t_axis.expand(*batch_shape, horizon) if batch_shape else t_axis.unsqueeze(0),
        q=q_result if batch_shape else q_result.unsqueeze(0),
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


__all__ = ["OptimizerFactory", "ResidualFactory", "TrajOptResult", "solve_trajopt"]
