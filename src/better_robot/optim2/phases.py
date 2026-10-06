"""Phases: run solvers one after another on the same problem with different weights and masks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from .problem import Problem
from .solvers import Solver


@dataclass
class Phase:
    """One stage of a solve.

    ``weights`` overrides term weights by name (0 switches a term off). ``free`` freezes
    tangent coordinates (see :meth:`Problem.with_`). ``on_start(problem, x)`` runs before
    the first step and may return replacement variables. Every ``stop_every`` iterations
    the driver stops early if all problems are done; 0 runs exactly ``iters`` steps.
    """

    solver: Solver
    iters: int
    name: str = "phase"
    weights: Mapping[str, float] | None = None
    free: Mapping[str, torch.Tensor] | None = None
    on_start: Callable[[Problem, dict[str, torch.Tensor]], dict[str, torch.Tensor] | None] | None = None
    stop_every: int = 1


@dataclass
class Result:
    """Outcome of :func:`run`."""

    x: dict[str, torch.Tensor]
    loss: torch.Tensor  #: ``(B,)`` final loss under the last phase's weights.
    done: torch.Tensor  #: ``(B,)`` convergence flag of the last phase.
    iters: int  #: total steps taken.
    history: dict[str, torch.Tensor] = field(default_factory=dict)  #: ``{phase: (steps, B)}`` losses.


def run(problem: Problem, phases: Sequence[Phase], x: Mapping[str, torch.Tensor] | None = None) -> Result:
    """Run ``phases`` in order, each starting from the previous result."""
    current = {name: value.detach() for name, value in (x if x is not None else problem.variables).items()}
    history: dict[str, torch.Tensor] = {}
    total, view, state = 0, problem, None
    for phase in phases:
        view = problem.with_(weights=phase.weights, free=phase.free)
        if phase.on_start is not None:
            current = phase.on_start(view, current) or current
        state = phase.solver.init(view, current)
        losses = []
        for iteration in range(phase.iters):
            state = phase.solver.step(view, state)
            losses.append(state.loss)
            if phase.stop_every and (iteration + 1) % phase.stop_every == 0 and bool(state.done.all()):
                break
        total += state.it
        current = {name: value.detach() for name, value in state.x.items()}
        if losses:
            history[phase.name] = torch.stack(losses)
    with torch.no_grad():
        loss = view.loss(current)
    done = state.done if state is not None else torch.zeros_like(loss, dtype=torch.bool)
    return Result(x=current, loss=loss, done=done, iters=total, history=history)


def solve(
    problem: Problem, solver: Solver, *, iters: int = 50, x: Mapping[str, torch.Tensor] | None = None
) -> Result:
    """Run one solver for at most ``iters`` steps."""
    return run(problem, [Phase(solver=solver, iters=iters)], x)


__all__ = ["Phase", "Result", "run", "solve"]
