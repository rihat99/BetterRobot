"""Phases: run solvers one after another on the same problem with different weights and masks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import torch

from .problem import Problem
from .solvers import DIAGNOSTICS, Solver, State


@dataclass(frozen=True)
class Phase:
    """One stage of a solve.

    ``weights`` overrides term weights by name (0 switches a term off). ``free`` freezes
    tangent coordinates (see ``Problem.with_``). ``on_start(problem, x)`` runs before
    the first step; it may return replacement variables and may write ``problem.data`` (the
    phase's own copy) to refresh what terms read, such as contact labels.
    ``on_step(problem, state)`` runs after every step, for logging; ``state.extra["terms"]`` holds
    the loss of every term (and, under ``TorchOptim``, ``state.extra["info"]`` the terms' info dicts)
    from the step's own evaluation, so logging costs no second forward. Every ``stop_every``
    iterations the driver stops early if all problems are done; 0 runs exactly ``iters`` steps.
    """

    solver: Solver
    iters: int
    name: str = "phase"
    weights: Mapping[str, float] | None = None
    free: Mapping[str, torch.Tensor] | None = None
    on_start: Callable[[Problem, dict[str, torch.Tensor]], dict[str, torch.Tensor] | None] | None = None
    on_step: Callable[[Problem, State], None] | None = None
    stop_every: int = 1


@dataclass(frozen=True)
class Result:
    """Outcome of ``run``."""

    x: dict[str, torch.Tensor]
    loss: torch.Tensor  #: ``(B,)`` final loss under the last phase's weights.
    done: torch.Tensor  #: ``(B,)`` convergence flag of the last phase.
    iters: int  #: total steps taken.
    history: dict[str, torch.Tensor] = field(default_factory=dict)  #: ``{phase: (steps, B)}`` losses.
    #: ``{phase: {key: (steps, B)}}`` per-step solver diagnostics (key set: ``State`` docstring).
    diagnostics: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    state: State | None = None  #: final solver state of the last phase; pass as ``state=`` to warm-start.

    def __getitem__(self, name: str) -> torch.Tensor:
        """Final value of variable ``name``; shorthand for ``result.x[name]``."""
        return self.x[name]


def run(
    problem: Problem,
    phases: Sequence[Phase],
    *,
    x: Mapping[str, torch.Tensor] | None = None,
    state: State | None = None,
) -> Result:
    """Run ``phases`` in order, each starting from the previous result.

    ``state`` (a previous ``Result.state``) warm-starts the first phase: its solver keeps the
    ``warm_keys`` entries of ``state.extra`` (LM damping, L-BFGS memory) and recomputes the rest
    at the start point, which is ``x``, else ``state.x``, else the problem's variables. The batch
    size must match.
    """
    if state is not None and not isinstance(state, State):
        raise TypeError(f"state must be a solver State (e.g. Result.state), got {type(state).__name__}")
    start = x if x is not None else state.x if state is not None else problem.variables
    current = {name: value.detach() for name, value in start.items()}
    history: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    total, view, warm, state = 0, problem, state, None
    for phase in phases:
        view = problem.with_(weights=phase.weights, free=phase.free)
        if phase.on_start is not None:
            current = phase.on_start(view, current) or current
        state = phase.solver.init(view, current)
        if warm is not None:
            # optim imports only torch, so a plain ValueError stands in for ShapeError here.
            if warm.done.shape != state.done.shape:
                raise ValueError(f"warm-start state has batch {tuple(warm.done.shape)}, problem has {tuple(state.done.shape)}")
            kept = {key: warm.extra[key] for key in getattr(phase.solver, "warm_keys", ()) if key in warm.extra}
            state, warm = replace(state, extra={**state.extra, **kept}), None
        losses, steps = [], []
        for iteration in range(phase.iters):
            state = phase.solver.step(view, state)
            losses.append(state.loss)
            steps.append({key: state.extra[key] for key in DIAGNOSTICS if key in state.extra})
            if phase.on_step is not None:
                phase.on_step(view, state)
            if phase.stop_every and (iteration + 1) % phase.stop_every == 0 and bool(state.done.all()):
                break
        total += state.it
        current = {name: value.detach() for name, value in state.x.items()}
        if losses:
            history[phase.name] = torch.stack(losses)
            diagnostics[phase.name] = {key: torch.stack([step[key] for step in steps]) for key in steps[0]}
    with torch.no_grad():
        loss = view.loss(current)
    done = state.done if state is not None else torch.zeros_like(loss, dtype=torch.bool)
    return Result(x=current, loss=loss, done=done, iters=total, history=history, diagnostics=diagnostics, state=state)


def solve(
    problem: Problem,
    solver: Solver,
    *,
    iters: int = 50,
    x: Mapping[str, torch.Tensor] | None = None,
    state: State | None = None,
) -> Result:
    """Run one solver for at most ``iters`` steps; ``state`` warm-starts it (see ``run``)."""
    return run(problem, [Phase(solver=solver, iters=iters)], x=x, state=state)


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
    return replace(result, x=attached)


__all__ = ["Phase", "Result", "run", "solve", "solve_layer"]
