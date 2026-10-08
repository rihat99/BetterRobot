"""Inverse kinematics as an open recipe: build the problem, add terms if needed, solve."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..data_model.model import Model
from ..optim import LevenbergMarquardt, Problem, Result, Solver, solve
from ..optim.solvers import select
from .robot import RobotManifold, frame_pose, joint_limits, kinematic_state, rest


def ik_problem(
    model: Model,
    targets: Mapping[str, torch.Tensor],
    *,
    q0: torch.Tensor,
    limit_weight: float = 0.0,
    rest_weight: float = 0.0,
    q_rest: torch.Tensor | None = None,
) -> Problem:
    """Problem that moves frames to target poses.

    ``targets`` maps frame names to poses ``(B, 7)``; ``q0: (B, nq)`` is the start. Position
    limits are hard bounds. ``limit_weight`` and ``rest_weight`` add soft terms; both are off by
    default so a reachable target is reached exactly. Terms are named ``pose/<frame>``,
    ``limits`` and ``rest``; the returned problem accepts more terms.
    """
    problem = Problem(
        {"q": q0},
        manifolds={"q": RobotManifold(model)},
        bounds={"q": (model.lower_pos_limit, model.upper_pos_limit)},
        data={f"target/{frame}": pose for frame, pose in targets.items()},
    )
    state = problem.cached(kinematic_state(model), name="state")
    for frame in targets:
        problem.residual(f"pose/{frame}", frame_pose(model, frame=frame, target=f"target/{frame}", state=state))
    if limit_weight:
        problem.residual("limits", joint_limits(model), weight=limit_weight)
    if rest_weight:
        problem.residual("rest", rest(model, q_rest if q_rest is not None else model.q_neutral), weight=rest_weight)
    return problem


def random_start(model: Model, q0: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Copy of ``q0: (B, nq)`` with every vector-space joint that has finite limits sampled uniformly."""
    index = model.structure.manifold_euclidean_q_indices
    lower, upper = model.lower_pos_limit[index], model.upper_pos_limit[index]
    sample = torch.rand(q0.shape[0], index.numel(), generator=generator, dtype=q0.dtype).to(q0.device)
    sample = lower + (upper - lower) * sample
    out = q0.clone()
    out[:, index] = torch.where(torch.isfinite(sample), sample, q0[:, index])
    return out


def solve_ik(
    model: Model,
    targets: Mapping[str, torch.Tensor],
    *,
    q0: torch.Tensor,
    solver: Solver | None = None,
    iters: int = 100,
    restarts: int = 0,
    generator: torch.Generator | None = None,
) -> Result:
    """Solve ``ik_problem`` with LM (or ``solver``). The configuration is ``result.x["q"]``.

    ``restarts`` extra attempts start from random configurations; each batch element keeps
    its best attempt. This is what gets past the local minima of a single start.
    """
    problem = ik_problem(model, targets, q0=q0)
    solver = solver or LevenbergMarquardt()
    best = solve(problem, solver, iters=iters)
    for _ in range(restarts):
        attempt = solve(problem, solver, iters=iters, x={"q": random_start(model, q0, generator=generator)})
        better = attempt.loss < best.loss
        best = Result(
            x=select(better, attempt.x, best.x),
            loss=torch.where(better, attempt.loss, best.loss),
            done=torch.where(better, attempt.done, best.done),
            iters=best.iters + attempt.iters,
        )
    return best


__all__ = ["ik_problem", "random_start", "solve_ik"]
