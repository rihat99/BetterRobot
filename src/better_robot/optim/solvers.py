"""Solvers: ``init(problem, x) -> State`` and ``step(problem, state) -> State``.

Every tensor in ``State`` is batch-first. A step computes all batch elements and keeps
the old value where an element is done or its step was rejected, so shapes never change.
``step`` itself reads no tensor back to Python, except the ``LBFGS`` line search: one flag
per trial, to stop once every problem has its step (``stop_search=False`` runs every trial instead).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field, fields, replace
from typing import Any, ClassVar, final, override

import torch

from ..exceptions import ArgumentError, ShapeError
from .evaluation import CONSTRAINTS, active_terms
from .linear import ArrowNormal, BandedNormal, DenseSink, Normal, blocked
from .linearization import Rows, linearize
from .problem import Problem
from .qp import QPSolution, solve_qp


@final
@dataclass(frozen=True)
class State:
    """Optimizer state of a batch of problems.

    After a ``step``, ``extra`` holds the diagnostic keys of ``DIAGNOSTICS`` that the solver
    computes, each ``(B,)``: ``grad_norm`` (largest gradient entry before the step), ``step_norm``
    (largest entry of the step taken, 0 where none was taken), ``accepted`` (bool, the step
    moved the variables), ``damping`` (LM and ``SequentialQP``, damping for the next step) and ``violation``
    (``SequentialQP`` only: the largest distance of a constraint term's output from its bounds at ``x``, in that
    term's unit, 0 where every constraint holds). LM, Gauss-Newton and L-BFGS write the first three;
    ``TorchOptim`` writes none. ``run`` stacks them into
    ``Result.diagnostics``. The ``init`` of LM, Gauss-Newton and ``SequentialQP`` writes them too, for no step
    (``grad_norm`` at ``x``, ``step_norm`` 0, ``accepted`` false): a first state has the keys of a stepped one, so a
    compiled step serves both with one key.
    ``Solver.warm_keys`` names the ``extra`` entries a warm start keeps.

    Every solver also writes ``terms``, ``{name: (B,)}``: the detached loss of each active term from
    the evaluation that ``loss`` comes from, so logging needs no second forward. ``TorchOptim`` adds
    ``info``, the ``{name: info}`` dicts of terms returning ``(value, info)`` (see ``Problem.evaluate``).
    LM, Gauss-Newton and ``SequentialQP`` keep ``normal``, the normal equations at ``x``: a step linearizes once,
    at its candidate, and that serves the next step where the candidate is taken. ``SequentialQP`` steps an iterate
    that ``x`` follows only while it is within the solver's tolerance of the constraints: it keeps ``iterate`` (the
    iterate's variables and its loss), ``normal`` and ``rows`` there (the constraint terms linearized, ``None``
    without one), ``radius`` (the bound on the largest entry of its next step, infinite until a refused step sets
    it) and ``qp``, the ``QPSolution`` of its last quadratic program, which starts the next one; its ``accepted``
    and ``step_norm`` are the iterate's.
    ``solver`` is the class of the solver that made the state; a warm start needs a solver of that class.
    A state belongs to the problem ``init`` made it for: ``step`` it with that problem only. A loop that changes the
    problem's data (a new target each tick) calls ``init`` again, or ``run(..., state=)``, which does; stepping the old
    state would reuse the previous data's ``normal``.
    """

    x: dict[str, torch.Tensor] = field(repr=False)  #: variables, each ``(B, *event)``.
    loss: torch.Tensor = field(repr=False)  #: ``(B,)`` loss at ``x``.
    _: KW_ONLY
    done: torch.Tensor = field(repr=False)  #: ``(B,)`` bool, per-problem convergence.
    solver: type[Solver]  #: the solver kind that made this state.
    iteration: int = 0  #: steps taken so far; 0 inside a ``Phase.step`` hook, whose count ``run`` keeps.
    extra: dict[str, Any] = field(default_factory=dict, repr=False)  #: solver-owned (damping, moments, ...).


DIAGNOSTICS = ("grad_norm", "step_norm", "accepted", "damping", "violation")


class Solver(ABC):
    """A solver is ``init(problem, x) -> State`` and ``step(problem, state) -> State``; a new one subclasses this.

    ``solve`` and ``run`` call ``init`` once and then ``step`` up to the iteration budget, stopping
    early once every batch element is done (see ``Phase.stop_every``). ``warm_keys`` names the
    ``State.extra`` entries a warm start (``run(..., state=)``) keeps; ``init`` recomputes the rest. Every
    ``State`` a solver returns names it (``solver=type(self)``) and carries its ``warm_keys`` entries.
    """

    warm_keys: ClassVar[tuple[str, ...]] = ()

    @override
    def __repr__(self) -> str:
        # a setting left at None takes the dtype's default, so only the set ones are printed
        settings = ", ".join(f"{name}={value!r}" for name, value in vars(self).items() if value is not None)
        return f"{type(self).__name__}({settings})"

    @abstractmethod
    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        """Start state at variables ``x``: loss evaluated, nothing done, solver-owned ``extra`` set up."""

    @abstractmethod
    def step(self, problem: Problem, state: State) -> State:
        """One iteration for every batch element; done or rejected elements keep their values. ``problem`` is the one
        ``state`` was made for (see ``State``)."""


def _select(mask: torch.Tensor, new: dict[str, torch.Tensor], old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per batch element, take ``new`` where ``mask: (B,)`` is true, else ``old``."""
    return {
        name: torch.where(mask.reshape(-1, *([1] * (value.ndim - 1))), value, old[name]) for name, value in new.items()
    }


def _selected_normal[Fields: Normal | BandedNormal | ArrowNormal | Rows](
    mask: torch.Tensor, new: Fields, old: Fields
) -> Fields:
    """Normal equations, or constraint rows: ``new`` where ``mask: (B,)`` is true, else ``old``; the band of an arrow
    too, one layout."""
    changes: dict[str, Any] = {}
    for item in fields(new):
        value = getattr(new, item.name)
        if isinstance(value, torch.Tensor):
            keep = mask.reshape(-1, *([1] * (value.ndim - 1)))
            changes[item.name] = torch.where(keep, value, getattr(old, item.name))
        elif isinstance(value, BandedNormal):
            changes[item.name] = _selected_normal(mask, value, getattr(old, item.name))
    return replace(new, **changes)


def _selected_terms(
    mask: torch.Tensor, new: dict[str, torch.Tensor], old: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Per-term losses ``{name: (B,)}``: ``new`` where ``mask: (B,)`` is true, else ``old``."""
    return {name: torch.where(mask, value, old[name]) for name, value in new.items()}


def _cost_only(problem: Problem, solver: Solver) -> None:
    """Refuse a problem with active constraint terms (``ArgumentError``): ``solver`` minimizes the loss alone."""
    held = [term.name for term in active_terms(problem) if term.kind in CONSTRAINTS]
    if held:
        raise ArgumentError(
            f"{type(solver).__name__} minimizes the loss and holds no constraint term, and {held} are active; solve "
            "with SequentialQP, or switch them off with weights"
        )


def _tolerances(
    dtype: torch.dtype, gradient: float | None, loss: float | None, step: float | None
) -> tuple[float, float, float]:
    """Fill unset tolerances with values the dtype can resolve (gradient, relative loss, step)."""
    default_gradient, default_loss, default_step = (1e-8, 1e-12, 1e-10) if dtype == torch.float64 else (1e-7, 0.0, 0.0)
    return (
        default_gradient if gradient is None else gradient,
        default_loss if loss is None else loss,
        default_step if step is None else step,
    )


@final
class LevenbergMarquardt(Solver):
    """Levenberg-Marquardt with per-problem damping (Nielsen update) and Marquardt scaling.

    A problem is done when its gradient is below ``gradient_tolerance``, an accepted step changes the loss
    by less than ``loss_tolerance`` (relative) or the variables by less than ``step_tolerance``, or its
    damping has reached the maximum (no step is accepted any more). ``damping`` is the initial damping.
    ``gradient_tolerance`` bounds the largest gradient entry; a tolerance left at ``None`` takes a default the
    dtype can resolve (``1e-8, 1e-12, 1e-10`` in float64, ``1e-7, 0, 0`` otherwise).
    """

    warm_keys = ("damping", "growth")

    def __init__(
        self,
        *,
        damping: float = 1e-3,
        gradient_tolerance: float | None = None,
        loss_tolerance: float | None = None,
        step_tolerance: float | None = None,
    ) -> None:
        if damping < 0:
            raise ArgumentError(f"damping must be at least 0, got {damping}")
        self.damping, self.gradient_tolerance = damping, gradient_tolerance
        self.loss_tolerance, self.step_tolerance = loss_tolerance, step_tolerance

    @override
    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        _cost_only(problem, self)
        normal, loss, terms, _ = linearize(problem, x)
        extra: dict[str, Any] = {"damping": torch.full_like(loss, self.damping), "growth": torch.full_like(loss, 2.0)}
        extra.update(_no_step(normal.g, problem.step_bounds(x), loss), terms=terms, normal=normal)
        return State(
            x=dict(x), loss=loss, done=torch.zeros_like(loss, dtype=torch.bool), solver=type(self), extra=extra
        )

    @override
    def step(self, problem: Problem, state: State) -> State:
        damping, growth, normal = state.extra["damping"], state.extra["growth"], state.extra["normal"]
        bounds = problem.step_bounds(state.x)
        if bounds is not None:
            normal = normal.restrict(*bounds)
        delta = normal.solve(damping)
        with torch.no_grad():
            candidate = problem.retract(state.x, delta)
        normal_new, loss_new, terms_new, _ = linearize(problem, candidate)
        decrease = state.loss - loss_new
        accept = (decrease > 0) & torch.isfinite(loss_new) & ~state.done
        ratio = decrease / normal.predicted_decrease(delta).clamp_min(1e-30)
        shrink = (1.0 - (2.0 * ratio - 1.0) ** 3).clamp(min=1.0 / 3.0, max=1.0)
        new_damping = torch.where(accept, damping * shrink, damping * growth).clamp(1e-12, 1e12)
        new_growth = torch.where(accept, torch.full_like(growth, 2.0), growth * 2.0)
        gradient_tolerance, loss_tolerance, step_tolerance = _tolerances(
            state.loss.dtype, self.gradient_tolerance, self.loss_tolerance, self.step_tolerance
        )
        grad_norm, step_norm = normal.g.abs().amax(-1), delta.abs().amax(-1)
        small_gradient = grad_norm < gradient_tolerance
        small_change = accept & ((decrease <= loss_tolerance * state.loss.abs()) | (step_norm < step_tolerance))
        return State(
            x=_select(accept, candidate, state.x),
            loss=torch.where(accept, loss_new, state.loss),
            done=state.done | small_gradient | small_change | (new_damping >= 1e12),
            solver=type(self),
            iteration=state.iteration + 1,
            extra={
                "damping": new_damping,
                "growth": new_growth,
                "grad_norm": grad_norm,
                "step_norm": torch.where(accept, step_norm, torch.zeros_like(step_norm)),
                "accepted": accept,
                "terms": _selected_terms(accept, terms_new, state.extra["terms"]),
                "normal": _selected_normal(accept, normal_new, state.extra["normal"]),
            },
        )


@final
class GaussNewton(Solver):
    """Gauss-Newton: full step of the linearized problem, with optional fixed damping."""

    def __init__(
        self, *, damping: float = 0.0, gradient_tolerance: float | None = None, step_tolerance: float | None = None
    ) -> None:
        if damping < 0:
            raise ArgumentError(f"damping must be at least 0, got {damping}")
        self.damping, self.gradient_tolerance, self.step_tolerance = damping, gradient_tolerance, step_tolerance

    @override
    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        _cost_only(problem, self)
        normal, loss, terms, _ = linearize(problem, x)
        return State(
            x=dict(x),
            loss=loss,
            done=torch.zeros_like(loss, dtype=torch.bool),
            solver=type(self),
            extra={**_no_step(normal.g, problem.step_bounds(x), loss), "terms": terms, "normal": normal},
        )

    @override
    def step(self, problem: Problem, state: State) -> State:
        normal = state.extra["normal"]
        bounds = problem.step_bounds(state.x)
        if bounds is not None:
            normal = normal.restrict(*bounds)
        delta = normal.solve(self.damping)
        with torch.no_grad():
            candidate = problem.retract(state.x, delta)
        normal_new, loss_new, terms_new, _ = linearize(problem, candidate)
        move = torch.isfinite(loss_new) & ~state.done
        gradient_tolerance, _, step_tolerance = _tolerances(
            state.loss.dtype, self.gradient_tolerance, None, self.step_tolerance
        )
        grad_norm, step_norm = normal.g.abs().amax(-1), delta.abs().amax(-1)
        done = state.done | (grad_norm < gradient_tolerance) | (step_norm < step_tolerance)
        extra = {"grad_norm": grad_norm, "step_norm": torch.where(move, step_norm, torch.zeros_like(step_norm))}
        return State(
            x=_select(move, candidate, state.x),
            loss=torch.where(move, loss_new, state.loss),
            done=done,
            solver=type(self),
            iteration=state.iteration + 1,
            extra={
                **extra,
                "accepted": move,
                "terms": _selected_terms(move, terms_new, state.extra["terms"]),
                "normal": _selected_normal(move, normal_new, state.extra["normal"]),
            },
        )


@final
class SequentialQP(Solver):
    """Sequential quadratic programming: Levenberg-Marquardt whose step is a quadratic program with the problem's
    constraint terms and bounds as rows, solved by `solve_qp`.

    Each step minimizes the damped Gauss-Newton model of the loss subject to the constraint terms linearized at
    the iterate: a term held to ``lower <= value <= upper`` gives the rows ``gain · (lower - value) <= J · delta <=
    gain · (upper - value)``, so a step uses at most ``gain`` of a row's distance to its bound and repairs ``gain``
    of a violation (1 is the plain linearization, smaller keeps a reserve for the curvature the rows do not see);
    the problem's bounds are a box on the step. The quadratic program starts from the solution of the last one
    (``State.extra["qp"]``, also kept by a warm start), so the row count must stay fixed between steps.

    A step is taken where it lowers the merit: the loss plus the constraint terms' summed violation times a price
    (the step's largest multiplier, capped by two prices that redundant rows do not inflate). The damping follows the
    ratio of that decrease to the one the program predicted by Nielsen's rule, also on a step that is taken: it
    stands in for the curvature of the constraint terms, which the rows do not carry. The step is also scaled back
    to a radius on its largest entry, which a taken step doubles and a refused one shrinks once the damping has
    reached 1; there is none at first.

    The iterate may leave a curved constraint by what its rows do not see, and later steps bring it back.
    ``State.x`` is the last iterate that was within ``tolerance`` of every bound (in the terms' own units) or,
    before any was, the closest one: a solve started within ``tolerance`` returns a point within it after any
    number of steps. ``State.loss``, ``State.extra["violation"]`` and the ``violation`` diagnostic describe that
    point; the iterate and its loss are ``State.extra["iterate"]``, and ``normal`` and ``rows`` are linearized
    there. A problem is done once its iterate is within ``tolerance`` and either the gradient of the loss plus the
    rows' multipliers has no entry over ``gradient_tolerance`` or a step it took had none over ``step_tolerance``
    (LM's defaults; LM's test on the loss decrease has no place here, the merit being flat along a step that only
    repairs a row), or when its damping has reached the maximum or its radius the minimum.

    The normal equations must be dense (``Problem(dense=True)`` for a ``time=`` problem), and the problem needs a
    constraint term or bounds: without either, use `LevenbergMarquardt`.
    """

    warm_keys = ("damping", "growth", "radius", "qp")

    def __init__(
        self,
        *,
        damping: float = 1e-3,
        gain: float = 0.5,
        tolerance: float = 1e-4,
        gradient_tolerance: float | None = None,
        step_tolerance: float | None = None,
    ) -> None:
        if damping < 0:
            raise ArgumentError(f"damping must be at least 0, got {damping}")
        if not 0 < gain <= 1:
            raise ArgumentError(f"gain must be in (0, 1], got {gain}")
        if tolerance <= 0:
            raise ArgumentError(f"tolerance must be above 0, got {tolerance}")
        self.damping, self.gain, self.tolerance = damping, gain, tolerance
        self.gradient_tolerance, self.step_tolerance = gradient_tolerance, step_tolerance

    @override
    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        normal, loss, terms, rows = linearize(problem, x)
        if not isinstance(normal, Normal):
            raise ArgumentError("SequentialQP takes dense normal equations; build the problem with dense=True")
        if rows is None and not problem.bounds:
            raise ArgumentError(
                "SequentialQP holds constraint terms and bounds, and the problem has neither; use LevenbergMarquardt"
            )
        damping = torch.full_like(loss, self.damping)
        matrix, lower, upper = _program(problem, x, rows, self.gain, 1.0)
        # The first program is solved here, cold: a first state then has every key of a stepped one.
        solution = solve_qp(_damped(normal, damping), normal.g, rows=matrix, lower=lower, upper=upper)
        extra: dict[str, Any] = {
            "damping": damping,
            "growth": torch.full_like(loss, 2.0),
            "radius": torch.full_like(loss, torch.inf),
            "grad_norm": (normal.g + (solution.dual.unsqueeze(-2) @ matrix).squeeze(-2)).abs().amax(-1),
            "step_norm": torch.zeros_like(loss),
            "accepted": torch.zeros_like(loss, dtype=torch.bool),
            "violation": torch.zeros_like(loss) if rows is None else rows.violation,
            "terms": terms,
            "normal": normal,
            "rows": rows,
            "qp": solution,
            "iterate": (dict(x), loss),
        }
        return State(
            x=dict(x), loss=loss, done=torch.zeros_like(loss, dtype=torch.bool), solver=type(self), extra=extra
        )

    @override
    def step(self, problem: Problem, state: State) -> State:
        damping, growth, radius, normal, rows = (
            state.extra[key] for key in ("damping", "growth", "radius", "normal", "rows")
        )
        point, point_loss = state.extra["iterate"]
        matrix, lower, upper = _program(problem, point, rows, self.gain, 1.0)
        solution = solve_qp(
            _damped(normal, damping), normal.g, rows=matrix, lower=lower, upper=upper, warm=state.extra["qp"]
        )
        # The radius scales the program's step back and leaves the program alone: a shorter step along it still
        # descends the program's convex model, and a box of the radius in the program makes its solutions inexact.
        length = solution.x.abs().amax(-1)
        tiny = torch.finfo(length.dtype).tiny
        delta = solution.x * (radius / length.clamp_min(tiny)).clamp(max=1.0).unsqueeze(-1)
        step_norm = delta.abs().amax(-1)
        with torch.no_grad():
            candidate = problem.retract(point, delta)
        normal_new, loss_new, terms_new, rows_new = linearize(problem, candidate)
        decrease, predicted = point_loss - loss_new, normal.predicted_decrease(delta)
        violation = violation_new = torch.zeros_like(loss_new)
        if rows is not None and rows_new is not None:
            dual = solution.dual[:, : rows.value.shape[1]]
            excess = _excess(rows, rows.value)
            linear = rows.value + (rows.jacobian @ delta.unsqueeze(-1)).squeeze(-1)
            repair = excess - _excess(rows, linear)
            # Two prices of a unit of violation that multipliers drifting apart on redundant rows do not raise:
            # the share of the multipliers' pull along a row, and what the step's repair costs the loss model.
            pull = (rows.jacobian @ (dual.unsqueeze(-2) @ rows.jacobian).transpose(-1, -2)).squeeze(-1).abs()
            along = torch.where(dual != 0, pull / rows.jacobian.square().sum(-1).clamp_min(tiny), 0.0).amax(-1)
            steering = torch.where(repair > 0, (-predicted).clamp_min(0.0) / (0.9 * repair).clamp_min(tiny), 0.0)
            penalty = torch.minimum(dual.abs().amax(-1), torch.maximum(along, steering))
            decrease = decrease + penalty * (excess - _excess(rows_new, rows_new.value))
            predicted = predicted + penalty * repair
            violation, violation_new = rows.violation, rows_new.violation
        accept = (decrease > 0) & ~state.done
        ratio = decrease / predicted.clamp_min(1e-30)
        # Nielsen's factor whole: a taken step whose ratio is under a half raises the damping.
        factor = (1.0 - (2.0 * ratio - 1.0) ** 3).clamp(min=1.0 / 3.0, max=2.0)
        # A done problem keeps its damping, growth and radius, so a warm start from it does not begin at a limit.
        new_damping = torch.where(accept, damping * factor, damping * growth).clamp(1e-12, 1e12)
        new_damping = torch.where(state.done, damping, new_damping)
        new_growth = torch.where(accept, torch.full_like(growth, 2.0), torch.where(state.done, growth, growth * 2.0))
        # Below a damping of 1 a refusal is the damping's to answer, and the radius stays. Past it a refused step
        # leaves its length, and a quarter of it where the damping cannot shorten it: the step was already cut to
        # the radius, or it is no shorter than the last one refused.
        stiff = damping >= 1.0
        forced = (length > radius) | (stiff & (step_norm >= 0.9 * radius))
        refused = torch.where(forced, 0.25 * step_norm, torch.where(stiff, step_norm, radius))
        new_radius = torch.where(accept, 2.0 * radius, refused)
        new_radius = torch.where(state.done, radius, new_radius.clamp_min(1e-12))
        gradient_tolerance, _, step_tolerance = _tolerances(
            state.loss.dtype, self.gradient_tolerance, None, self.step_tolerance
        )
        grad_norm = (normal.g + (solution.dual.unsqueeze(-2) @ matrix).squeeze(-2)).abs().amax(-1)
        inside = torch.where(accept, violation_new, violation) <= self.tolerance
        settled = inside & ((grad_norm < gradient_tolerance) | (accept & (step_norm < step_tolerance)))
        reported = state.extra["violation"]
        # State.x follows the iterate while it is within the tolerance, or closer to the bounds than any before.
        report = accept & (violation_new <= reported.clamp_min(self.tolerance))
        return State(
            x=_select(report, candidate, state.x),
            loss=torch.where(report, loss_new, state.loss),
            done=state.done | settled | (new_damping >= 1e12) | (new_radius <= 1e-12),
            solver=type(self),
            iteration=state.iteration + 1,
            extra={
                "damping": new_damping,
                "growth": new_growth,
                "radius": new_radius,
                "grad_norm": grad_norm,
                "step_norm": torch.where(accept, step_norm, torch.zeros_like(step_norm)),
                "accepted": accept,
                "violation": torch.where(report, violation_new, reported),
                "terms": _selected_terms(report, terms_new, state.extra["terms"]),
                "normal": _selected_normal(accept, normal_new, normal),
                "rows": None if rows is None or rows_new is None else _selected_normal(accept, rows_new, rows),
                "qp": solution,
                "iterate": (_select(accept, candidate, point), torch.where(accept, loss_new, point_loss)),
            },
        )


def track(
    problem: Problem,
    x: dict[str, torch.Tensor],
    *,
    step: float | torch.Tensor,
    warm: QPSolution | None = None,
    gain: float = 0.5,
    damping: float = 1e-3,
    iterations: int = 20,
) -> tuple[dict[str, torch.Tensor], QPSolution]:
    """One step of differential inverse kinematics: ``x`` moved toward the problem's targets under its constraint
    terms and bounds, by one quadratic program. Returns the new point and the program's solution; pass that back as
    ``warm=`` on the next call (it is all the state a loop carries).

    ``x`` holds one ``(B, *event)`` tensor per variable. The step ``delta`` minimizes ``½ deltaᵀ(H + damping·I) delta
    + gᵀdelta``, the Gauss-Newton model of the cost terms at ``x`` (``H``, ``g`` of `Problem.linearize`) with
    ``damping`` on its diagonal, subject to the rows of `SequentialQP` at ``gain``: a constraint term held to ``lower
    <= value <= upper`` gives ``gain · (lower - value) <= J · delta <= gain · (upper - value)``, and the problem's
    bounds give the box ``gain · (lower - x) <= delta <= gain · (upper - x)``. So a row's distance to its bound
    shrinks by at most ``1 - gain`` per call (a discrete barrier: exact for rows linear in ``x``, such as the bounds,
    and up to the curvature the rows do not see for the others), and a violation is repaired at the same rate.
    ``step``, a positive number or ``(N,)`` of them, bounds every entry of ``delta`` (a robot's velocity limits times
    the time step); it has no default, because the curved rows hold only for small steps. The retraction clamps to
    the bounds, as in every solver: a start outside them is back inside after one call, whatever ``step``. The
    damping is added as a multiple of the identity, so ``H + damping·I`` stays positive definite where the targets'
    Jacobian loses rank and on coordinates no cost term reads; it shortens every step and moves no fixed point.

    Unlike `SequentialQP` nothing is accepted or refused: a step is taken every call, with its rows elastic as in
    `solve_qp`, and the program's step may not be its exact solution (``solution.exact`` unset). Two things keep
    such a step on the rows. The program holds each constraint row with a reserve: a tenth of what a step as long
    as the last program's (``warm.x``; none on a first call) can change that row by, never more than half the
    row's own width. A point that moves along a row keeps that reserve over ``gain`` short of it; one that comes to
    rest closes in on the row itself. And the step taken is the program's clamped to the box, moved the least
    distance that puts it back on the rows it leaves (among the coordinates the box leaves free), then scaled back
    until it leaves no row by more than 1e-6 in that row's unit and takes no violated row further out. So
    ``solution.x`` is the program's step and the new point carries the step taken. The scale-back shortens a step
    to nothing where the rows leave it no direction: rows in conflict, or a program still far from solved; the next
    calls go on solving it from ``warm``. ``iterations`` goes to `solve_qp`: more per call make more of its steps
    exact where many rows press at once.
    The targets move between calls through ``problem.with_(data=...)``; the rows must keep their count (``closest=``)
    so that ``warm`` fits. The problem needs dense normal equations. One evaluation, one `solve_qp` and one
    retraction per call, with no read back to the host.
    """
    if not 0 < gain <= 1:
        raise ArgumentError(f"gain must be in (0, 1], got {gain}")
    if damping < 0:
        raise ArgumentError(f"damping must be at least 0, got {damping}")
    if isinstance(step, torch.Tensor):
        if step.shape not in ((), (problem.tangent_dim,)):
            raise ShapeError(f"step must be a number or have shape ({problem.tangent_dim},), got {tuple(step.shape)}")
    elif step <= 0:  # bench-ok: a Python number on this branch; a tensor's values are not scanned
        raise ArgumentError(f"step must be positive, got {step}")
    normal, _, _, rows = linearize(problem, x)
    if not isinstance(normal, Normal):
        raise ArgumentError("track takes dense normal equations; build the problem with dense=True")
    matrix, lower, upper = _program(problem, x, rows, gain, gain, step)
    count = 0 if rows is None else rows.value.shape[1]
    travel = 0.0 if warm is None else warm.x.abs().unsqueeze(-2)
    reach = (matrix[:, :count].abs() * travel).sum(-1)
    reserve = torch.minimum(_RESERVE * reach, (upper[:, :count] - lower[:, :count]) / 2)
    diagonal = torch.full_like(normal.g, damping)
    solution = solve_qp(
        normal.H + torch.diag_embed(diagonal),
        normal.g,
        rows=matrix,
        lower=torch.cat((lower[:, :count] + reserve, lower[:, count:]), dim=1),
        upper=torch.cat((upper[:, :count] - reserve, upper[:, count:]), dim=1),
        iterations=iterations,
        warm=warm,
    )
    delta = _held(solution.x, matrix, lower, upper, count)
    with torch.no_grad():
        return problem.retract(x, delta), solution


def _held(
    delta: torch.Tensor, matrix: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, count: int
) -> torch.Tensor:
    """``delta: (B, N)`` clamped to the box rows (the rows after the first ``count``), moved back onto the first
    ``count`` rows it leaves and scaled back so that it leaves none of them by more than `_SLACK`, nor takes a
    violated one further out."""
    box_low, box_high = lower[:, count:], upper[:, count:]
    delta = torch.clamp(delta, box_low, box_high)
    if count:
        rows = matrix[:, :count]
        low, high = lower[:, :count].clamp(max=0.0), upper[:, :count].clamp(min=0.0)
        value = (rows @ delta.unsqueeze(-1)).squeeze(-1)
        excess = value - torch.clamp(value, low, high)
        # The least change of delta that takes the excess off the rows it leaves, among the coordinates that the
        # box does not hold against it: scaling alone would give up the whole step for one row left by a hair.
        left = rows * (excess != 0).unsqueeze(-1)
        columns = torch.arange(left.shape[-1], device=left.device)[None]
        sink = DenseSink(left, left.shape[0], left.shape[-1])
        sink.add(columns, columns, (left.mT @ left).unsqueeze(1), shared=(False, False))
        back = sink.finish((excess.unsqueeze(-2) @ left).squeeze(-2))
        delta = delta + back.restrict(box_low - delta, box_high - delta).solve()
        value = (rows @ delta.unsqueeze(-1)).squeeze(-1)
        low, high = low - _SLACK, high + _SLACK
        ratio = torch.where(value > high, high / value, torch.where(value < low, low / value, 1.0))
        delta = delta * ratio.amin(-1, keepdim=True)
    return delta


_SLACK = 1e-6  # how far the step of `track` may leave a row that held at its start, in the row's unit
_RESERVE = 0.1  # share of what the last step could change a row by that the program of `track` keeps short of its bound


def _program(
    problem: Problem,
    x: dict[str, torch.Tensor],
    rows: Rows | None,
    gain: float,
    box: float,
    step: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows ``(B, M, N)`` and bounds ``(B, M), (B, M)`` of a step from ``x``: the constraint terms at ``gain``, then
    the box of the problem's bounds at ``box``, clamped to ``±step`` (a number or ``(N,)``)."""
    matrices, lowers, uppers = [], [], []
    if rows is not None:
        matrices.append(rows.jacobian)
        lowers.append(gain * (rows.lower - rows.value))
        uppers.append(gain * (rows.upper - rows.value))
    bounds = problem.step_bounds(x)
    if bounds is None and step is not None:
        first = next(iter(x.values()))
        infinite = first.new_full((first.shape[0], problem.tangent_dim), torch.inf)
        bounds = (-infinite, infinite)
    if bounds is not None:
        low, high = bounds if box == 1.0 else (box * bounds[0], box * bounds[1])
        if step is not None:
            limit = torch.as_tensor(step, dtype=low.dtype, device=low.device)
            # both sides inside ±step, so a box that ``step`` cannot reach still has lower <= upper
            low, high = torch.clamp(low, -limit, limit), torch.clamp(high, -limit, limit)
        identity = torch.eye(low.shape[-1], dtype=low.dtype, device=low.device)
        matrices.append(identity.expand(low.shape[0], -1, -1))
        lowers.append(low)
        uppers.append(high)
    if not matrices:
        raise ArgumentError("the problem has no constraint term and no bounds, and no step limit was given")
    return torch.cat(matrices, dim=1), torch.cat(lowers, dim=1), torch.cat(uppers, dim=1)


def _excess(rows: Rows, value: torch.Tensor) -> torch.Tensor:
    """Summed distance of ``value: (B, M)`` from the bounds of ``rows``, ``(B,)``, in the rows' weighted units."""
    return torch.maximum(value - rows.upper, rows.lower - value).clamp_min(0.0).sum(-1)


def _damped(normal: Normal, damping: torch.Tensor) -> torch.Tensor:
    """``H + damping · diag(H)``, ``(B, N, N)``: the matrix `Normal.solve` factorizes, for `solve_qp`."""
    return normal.H + torch.diag_embed(damping.unsqueeze(-1) * normal.H.diagonal(dim1=-2, dim2=-1))


def _no_step(
    g: torch.Tensor, bounds: tuple[torch.Tensor, torch.Tensor] | None, loss: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The diagnostic keys of a state no step has made, from the gradient ``g: (B, n)`` at its ``x`` and the step
    bounds there: ``init`` writes them so that its state has the keys of a stepped one (one `kinematics.compiled` key
    for both)."""
    return {
        "grad_norm": _projected(g, bounds).abs().amax(-1),
        "step_norm": torch.zeros_like(loss),
        "accepted": torch.zeros_like(loss, dtype=torch.bool),
    }


def _free(g: torch.Tensor, bounds: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor | bool:
    """``(B, n)`` mask of the coordinates not held by an active step bound that the gradient ``g`` pushes against."""
    return True if bounds is None else ~blocked(g, *bounds)


def _projected(g: torch.Tensor, bounds: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
    """Gradient ``(B, n)`` with the entries that push against an active step bound zeroed."""
    return g * _free(g, bounds)


@final
class LBFGS(Solver):
    """Batched L-BFGS in tangent coordinates: per-problem history and backtracking line search.

    Every step tries the step sizes 1, ½, ¼, ... (at most ``line_search`` of them) and each problem
    takes the largest one that satisfies the Armijo condition on the step actually taken; the search
    stops as soon as every problem has its step (one host read per trial); with ``stop_search=False`` it runs all
    ``line_search`` trials and reads nothing, the same steps at more evaluations. The first trial is evaluated
    with its gradient: when every problem takes it, that is the step's only evaluation; otherwise the
    gradient is evaluated again at the new points. The plain gradient step (no history yet) has length
    ``1 / |g|_1``, a guess: a stiff term that adds curvature but little gradient (a residual near
    zero) can need 10 or more halvings of it. With bounds, a coordinate
    held by an active bound that the gradient pushes against does not move and every step is clipped
    to the bounds. A problem that finds no step keeps its variables and drops its history; when that
    happens on a plain gradient step (no history to drop) it is done, as every later step would repeat
    the same search. ``history`` is the number of stored curvature pairs, ``line_search`` (at least 1)
    the most trials per step, ``armijo`` the sufficient-decrease constant and ``gradient_tolerance`` bounds the largest
    gradient entry (``None`` takes the dtype default, as for LM).
    """

    def __init__(
        self,
        *,
        history: int = 10,
        line_search: int = 20,
        armijo: float = 1e-4,
        gradient_tolerance: float | None = None,
        stop_search: bool = True,
    ) -> None:
        if line_search < 1:
            raise ArgumentError(f"line_search must be at least 1, got {line_search}")
        self.history, self.line_search, self.armijo = history, line_search, armijo
        self.gradient_tolerance, self.stop_search = gradient_tolerance, stop_search

    warm_keys = ("s", "y", "rho")

    @override
    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        _cost_only(problem, self)
        loss, g, terms = problem.gradient(x)
        memory = g.new_zeros(g.shape[0], self.history, g.shape[1])
        extra = {"g": g, "s": memory, "y": memory.clone(), "rho": g.new_zeros(g.shape[0], self.history), "terms": terms}
        return State(
            x=dict(x), loss=loss, done=torch.zeros_like(loss, dtype=torch.bool), solver=type(self), extra=extra
        )

    def _direction(self, g: torch.Tensor, s: torch.Tensor, y: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Two-loop recursion; slots with ``rho == 0`` are empty and drop out."""
        q, alphas = g.clone(), []
        for i in reversed(range(self.history)):
            alpha = rho[:, i] * (s[:, i] * q).sum(-1)
            q = q - alpha.unsqueeze(-1) * y[:, i]
            alphas.append(alpha)
        newest = self.history - 1
        curvature = (y[:, newest] * y[:, newest]).sum(-1)
        gamma = torch.where(rho[:, newest] > 0, (s[:, newest] * y[:, newest]).sum(-1) / curvature.clamp_min(1e-30), 1.0)
        r = gamma.unsqueeze(-1) * q
        for i, alpha in zip(range(self.history), reversed(alphas)):
            beta = rho[:, i] * (y[:, i] * r).sum(-1)
            r = r + (alpha - beta).unsqueeze(-1) * s[:, i]
        return -r

    @override
    def step(self, problem: Problem, state: State) -> State:
        g, s, y, rho = (state.extra[key] for key in ("g", "s", "y", "rho"))
        bounds = problem.step_bounds(state.x)
        free = _free(g, bounds)
        g = g * free
        if bounds is not None:
            lower, upper = bounds
        direction = self._direction(g, s, y, rho) * free  # coordinates held by a bound stay put
        slope = (g * direction).sum(-1)
        uphill = slope >= 0
        direction = torch.where(uphill.unsqueeze(-1), -g, direction)
        first_step = (rho[:, -1] == 0) | uphill  # no curvature yet: scale the gradient step
        scale = torch.where(first_step, 1.0 / g.abs().sum(-1).clamp_min(1.0), 1.0)
        best_x, best_loss, best_step, best_terms = state.x, state.loss, torch.zeros_like(g), state.extra["terms"]
        found = torch.zeros_like(state.done)
        with torch.no_grad():
            for trial in range(self.line_search):
                step = direction * (scale * 0.5**trial).unsqueeze(-1)
                if bounds is not None:
                    step = torch.minimum(torch.maximum(step, lower), upper)
                candidate = problem.retract(state.x, step)
                if trial == 0:  # usually accepted: its gradient is then the one at the new point
                    loss, g_first, terms = problem.gradient(candidate)
                else:
                    loss, terms, _ = problem.evaluate(candidate)
                # Armijo on the step actually taken: a bound may have shortened it, even into a direction whose linear
                # model rises (slope > 0); the clamp then asks for no increase at all.
                slope = (g * step).sum(-1).clamp_max(0.0)
                ok = (loss <= state.loss + self.armijo * slope) & torch.isfinite(loss) & ~found
                ok &= ~state.done
                best_x = _select(ok, candidate, best_x)
                best_loss = torch.where(ok, loss, best_loss)
                best_step = torch.where(ok.unsqueeze(-1), step, best_step)
                best_terms = _selected_terms(ok, terms, best_terms)
                found |= ok
                if self.stop_search and bool((found | state.done).all()):  # bench-ok: later trials change nothing
                    break
        if trial == 0:  # every problem kept its start point or took the first trial
            g_new = torch.where(found.unsqueeze(-1), g_first, state.extra["g"])
        else:
            _, g_new, _ = problem.gradient(best_x)
        change = g_new - state.extra["g"]
        curvature = (best_step * change).sum(-1)
        keep = found & (curvature > 1e-10 * (change * change).sum(-1))
        pushed = lambda memory, new: torch.cat([memory[:, 1:], new.unsqueeze(1)], dim=1)  # noqa: E731
        s_new = torch.where(keep[:, None, None], pushed(s, best_step), s)
        y_new = torch.where(keep[:, None, None], pushed(y, change), y)
        rho_new = torch.where(keep[:, None], pushed(rho, 1.0 / curvature.clamp_min(1e-30)), rho)
        rho_new = torch.where(found[:, None], rho_new, torch.zeros_like(rho_new))  # failed search: forget history
        gradient_tolerance, _, _ = _tolerances(g.dtype, self.gradient_tolerance, None, None)
        grad_norm = _projected(g_new, problem.step_bounds(best_x)).abs().amax(-1)  # at the new point
        done = state.done | (grad_norm < gradient_tolerance) | (first_step & ~found)
        extra = {"g": g_new, "s": s_new, "y": y_new, "rho": rho_new, "grad_norm": grad_norm}
        extra.update(step_norm=best_step.abs().amax(-1), accepted=found, terms=best_terms)
        return State(x=best_x, loss=best_loss, done=done, solver=type(self), iteration=state.iteration + 1, extra=extra)


@final
class TorchOptim(Solver):
    """Run any ``torch.optim`` optimizer on the problem's loss.

    The optimizer steps the ambient tensors; afterwards they are projected onto their
    manifolds and bounds. ``State.loss`` is the loss before the step (no extra evaluation);
    ``extra["terms"]`` and ``extra["info"]`` come from the same, last closure evaluation.
    Adam-like optimizers keep batch elements independent; ``torch.optim.LBFGS`` couples them
    through its line search and treats the batch as one problem.
    ``factory(params)`` builds the optimizer over the list of variable tensors, e.g.
    ``TorchOptim(lambda params: torch.optim.Adam(params, lr=1e-2))``. ``scheduler(optimizer)`` may
    return a ``torch.optim.lr_scheduler`` object; its ``step()`` is called without arguments once per
    solver step.
    """

    def __init__(
        self,
        factory: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
        *,
        scheduler: Callable[[torch.optim.Optimizer], Any] | None = None,
    ) -> None:
        self.factory, self.scheduler = factory, scheduler

    @override
    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        _cost_only(problem, self)
        params = {name: value.detach().clone().requires_grad_(True) for name, value in x.items()}
        optimizer = self.factory(list(params.values()))
        loss = next(iter(x.values())).new_full((next(iter(x.values())).shape[0],), float("inf"))
        extra = {"params": params, "optimizer": optimizer}
        if self.scheduler is not None:
            extra["scheduler"] = self.scheduler(optimizer)
        return State(x=params, loss=loss, done=torch.zeros_like(loss, dtype=torch.bool), solver=type(self), extra=extra)

    @override
    def step(self, problem: Problem, state: State) -> State:
        params, optimizer = state.extra["params"], state.extra["optimizer"]
        last = {}

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss, terms, info = problem.evaluate(params)
            total = loss.sum()
            total.backward()
            self._mask_gradients(problem, params)
            last.update(loss=loss.detach(), terms={name: value.detach() for name, value in terms.items()}, info=info)
            return total

        optimizer.step(closure)
        if self.scheduler is not None:
            state.extra["scheduler"].step()
        with torch.no_grad():
            for name, value in problem.project(params).items():
                params[name].copy_(value)
        current = {name: value.detach().clone() for name, value in params.items()}
        extra = {**state.extra, "terms": last["terms"], "info": last["info"]}
        return State(
            x=current, loss=last["loss"], done=state.done, solver=type(self), iteration=state.iteration + 1, extra=extra
        )

    @staticmethod
    def _mask_gradients(problem: Problem, params: dict[str, torch.Tensor]) -> None:
        for name, mask in (problem.free or {}).items():
            gradient = params[name].grad
            if gradient is None:
                continue
            if gradient.shape[1:].numel() != mask.shape[-1]:
                raise ShapeError(
                    f"free mask of {name!r} is in tangent coordinates ({mask.shape[-1]}) but the variable has "
                    f"{gradient.shape[1:].numel()} ambient coordinates; use a second-order solver for this phase"
                )
            gradient.mul_(
                mask.to(gradient.dtype).reshape(-1, *gradient.shape[1:])
                if mask.ndim == 2
                else mask.to(gradient.dtype).reshape(gradient.shape[1:])
            )


__all__ = ["LBFGS", "GaussNewton", "LevenbergMarquardt", "SequentialQP", "Solver", "State", "TorchOptim", "track"]
