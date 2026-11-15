"""Inverse kinematics as an open recipe: build the problem, add terms if needed, solve."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from ..exceptions import ArgumentError, ShapeError
from ..kinematics import euclidean_columns
from ..lie._checks import check_broadcast, check_tensor
from ..model import Model
from ..optim import LevenbergMarquardt, Phase, Problem, Result, Solver, State, run
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
    default so a reachable target is reached exactly. The rest term pulls toward ``q_rest: (nq,)`` or
    ``(B, nq)``, ``model.q_neutral`` when omitted. Terms are named ``pose/<frame>``,
    ``limits`` and ``rest``; the returned problem accepts more terms.

    ``q0`` and each target are checked here, once, in the model's dtype on its device (a target batch broadcasts
    with ``B``); the solver's iterations trust them.
    """
    _check(model, targets, q0)
    return _problem(model, targets, q0, limit_weight, rest_weight, q_rest)


def _check(model: Model, targets: Mapping[str, torch.Tensor], q0: torch.Tensor) -> None:
    """The boundary checks of ``ik_problem`` and ``solve_ik``: ``q0: (B, nq)`` and targets ``(..., 7)`` broadcasting
    with ``B``, in the model's dtype on its device."""
    check_tensor("q0", q0, shape=(model.nq,), dtype=model.dtype, device=model.device)
    if q0.ndim != 2:
        raise ShapeError(f"q0 must be (B, {model.nq}), got {tuple(q0.shape)}")
    for frame, pose in targets.items():
        name = f"targets[{frame!r}]"
        check_tensor(name, pose, shape=(7,), dtype=model.dtype, device=model.device)
        check_broadcast(q0=tuple(q0.shape[:-1]), **{name: tuple(pose.shape[:-1])})


def _problem(
    model: Model,
    targets: Mapping[str, torch.Tensor],
    q0: torch.Tensor,
    limit_weight: float,
    rest_weight: float,
    q_rest: torch.Tensor | None,
) -> Problem:
    """``ik_problem`` of checked inputs."""
    problem = Problem(
        {"q": q0},
        manifolds={"q": RobotManifold(model)},
        bounds={"q": (model.lower_position_limit, model.upper_position_limit)},
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


def random_start(model: Model, q0: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """Copy ``(B, nq)`` of ``q0: (B, nq)`` with every ``EUCLIDEAN`` coordinate that has finite limits sampled uniformly.

    The draw happens on ``q0``'s device, so ``generator`` must live there.
    """
    if generator is None:
        raise ArgumentError("random_start draws with a generator; pass one on q0's device")
    if q0.ndim != 2 or q0.shape[-1] != model.nq:
        raise ShapeError(f"q0 must be (B, {model.nq}), got {tuple(q0.shape)}")
    index, _ = euclidean_columns(model)
    lower, upper = model.lower_position_limit[index], model.upper_position_limit[index]
    sample = torch.rand(q0.shape[0], index.numel(), generator=generator, dtype=q0.dtype, device=q0.device)
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
    iterations: int = 50,
    seeds: int = 1,
    generator: torch.Generator | None = None,
    step: Callable[[Problem, State], State] | None = None,
) -> Result:
    """Solve ``ik_problem`` from ``q0: (B, nq)`` with LM (or ``solver``) for ``iterations`` steps. The configuration
    is ``result.x["q"]``.

    ``seeds`` starts per target are solved as one batch of ``seeds * B`` problems: the first is ``q0``, the others
    are random configurations (``random_start``) drawn with ``generator``, required when ``seeds > 1``; each target
    must then broadcast to one pose per row of ``q0`` (``ShapeError`` otherwise). Each target keeps the start that
    reached the lowest loss; that is what gets past the local minima of a single start. The solve runs exactly
    ``iterations`` steps and reads nothing back from the device on the way (``Phase`` with ``stop_every=0``).
    ``step`` replaces ``solver.step``, for a compiled one (``Phase``), and is the step of the ``solver`` passed here:
    ``kinematics.compiled(lambda problem, state: solver.step(problem, state))`` with ``solver=solver`` (with no
    ``solver``, the step of a ``LevenbergMarquardt()``). Every step here, the solver's own too, is handed
    ``state.iteration == 0`` as a ``Phase.step`` hook is (``optim.solve`` hands a solver's own step the count). With
    ``seeds > 1`` the result's ``history`` and ``diagnostics`` are those of the kept start and ``state`` is ``None``.
    """
    if seeds < 1:
        raise ArgumentError(f"seeds must be at least 1, got {seeds}")
    if seeds > 1 and generator is None:
        raise ArgumentError("seeds after the first start at random configurations; pass a generator on q0's device")
    _check(model, targets, q0)
    batch = q0.shape[0]
    if seeds > 1:
        for frame, pose in targets.items():  # the fold pairs each row of q0 with one pose of each target
            axes = torch.broadcast_shapes(pose.shape[:-1], (batch,))
            if axes.numel() != batch:
                raise ShapeError(
                    f"with seeds > 1, each row of q0 (batch ({batch},)) takes one pose of targets[{frame!r}], whose "
                    f"batch axes {tuple(pose.shape[:-1])} hold more"
                )
        draws = random_start(model, q0.repeat(seeds - 1, 1), generator=generator)
        q0 = torch.cat([q0, draws])
        targets = {frame: pose.reshape(-1, 7).expand(batch, 7).repeat(seeds, 1) for frame, pose in targets.items()}
    # contiguous, as the seeds' fold makes them: a compiled step then sees one stride for every call of a shape
    targets, q0 = {frame: pose.contiguous() for frame, pose in targets.items()}, q0.contiguous()
    problem = _problem(model, targets, q0, 0.0, 0.0, None)
    solver = solver or LevenbergMarquardt()
    # the solver's own step goes as the hook too, so a torch.compile of the whole call does not recompile it on the count
    result = run(problem, [Phase(solver, iterations, stop_every=0, step=step or solver.step)])
    if seeds == 1:
        return result
    # the flat index of each target's best start, seed-major: start s of target b is row s * B + b
    best = result.loss.view(seeds, batch).argmin(0) * batch + torch.arange(batch, device=q0.device)
    return Result(
        x={"q": result.x["q"][best]},
        loss=result.loss[best],
        done=result.done[best],
        iterations=result.iterations,
        history={phase: losses[:, best] for phase, losses in result.history.items()},
        diagnostics={
            phase: {key: value[:, best] for key, value in steps.items()} for phase, steps in result.diagnostics.items()
        },
    )


__all__ = ["ik_problem", "random_start", "solve_ik"]
