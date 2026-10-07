"""Phases: run solvers one after another on the same problem with different weights and masks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from .problem import Problem
from .solvers import Solver, State


@dataclass
class Phase:
    """One stage of a solve.

    ``weights`` overrides term weights by name (0 switches a term off). ``free`` freezes
    tangent coordinates (see ``Problem.with_``). ``on_start(problem, x)`` runs before
    the first step; it may return replacement variables and may write ``problem.data`` (the
    phase's own copy) to refresh what terms read, such as contact labels.
    ``on_step(problem, state)`` runs after every step, for logging; ``problem.losses(state.x)``
    gives the loss of every term. Every ``stop_every`` iterations the driver stops early if
    all problems are done; 0 runs exactly ``iters`` steps.
    """

    solver: Solver
    iters: int
    name: str = "phase"
    weights: Mapping[str, float] | None = None
    free: Mapping[str, torch.Tensor] | None = None
    on_start: Callable[[Problem, dict[str, torch.Tensor]], dict[str, torch.Tensor] | None] | None = None
    on_step: Callable[[Problem, State], None] | None = None
    stop_every: int = 1


@dataclass
class Result:
    """Outcome of ``run``."""

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
            if phase.on_step is not None:
                phase.on_step(view, state)
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


def solve(problem: Problem, solver: Solver, *, iters: int = 50, x: Mapping[str, torch.Tensor] | None = None) -> Result:
    """Run one solver for at most ``iters`` steps."""
    return run(problem, [Phase(solver=solver, iters=iters)], x)


def solve_layer(
    problem: Problem,
    solver: Solver,
    *,
    iters: int = 50,
    x: Mapping[str, torch.Tensor] | None = None,
    damping: float = 1e-6,
) -> Result:
    """Solve, then attach the solution to the autograd graph of the problem's data tensors.

    The solve itself runs without a graph. The returned value is exactly the solver's ``x*``;
    its derivative is that of one Newton step ``-H⁻¹ g(x*, data)`` with a constant Gauss-Newton
    ``H``, which at a converged solution is the implicit-function derivative of the solution
    w.r.t. the data. Valid when the solve has converged; coordinates on an active bound get
    zero sensitivity; with large residuals at the optimum the Gauss-Newton ``H`` is approximate.
    ``damping`` keeps the step finite along directions the residuals do not constrain (a
    redundant arm), where the sensitivity is zero. Problems with scalar terms are rejected:
    their curvature is not in ``H``, so the derivative would be wrong.
    """
    scalars = [term.name for term in problem._active() if term.kind == "scalar"]
    if scalars:
        raise ValueError(f"solve_layer supports residual terms only; active scalar terms: {scalars}")
    result = solve(problem, solver, iters=iters, x=x)
    normal, _ = problem.linearize(result.x)
    _, g = problem.gradient(result.x, create_graph=True)
    bounds = problem.step_bounds(result.x)
    if bounds is not None:
        lower, upper = bounds
        blocked = ((lower >= -1e-9) & (normal.g > 0)) | ((upper <= 1e-9) & (normal.g < 0))
        normal, g = normal.masked(~blocked), g * ~blocked
    step = normal.solve(damping, g=g)
    attached = problem._retract(result.x, step - step.detach())
    return Result(x=attached, loss=result.loss, done=result.done, iters=result.iters, history=result.history)


__all__ = ["Phase", "Result", "run", "solve", "solve_layer"]
