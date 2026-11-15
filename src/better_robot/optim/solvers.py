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
from .linear import ArrowNormal, BandedNormal, Normal
from .linearization import Rows, linearize, linearize_rows
from .problem import Problem
from .qp import solve_qp


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
    without one), and ``qp``, the ``QPSolution`` of its last quadratic program, which starts the next one; its
    ``accepted`` and ``step_norm`` are the iterate's.
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


def _selected_normal[Fields: (Normal, BandedNormal, ArrowNormal, Rows)](
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
        normal, loss, terms = linearize(problem, x)
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
        normal_new, loss_new, terms_new = linearize(problem, candidate)
        decrease = state.loss - loss_new
        accept = (decrease > 0) & torch.isfinite(loss_new) & ~state.done
        gain = decrease / normal.predicted_decrease(delta).clamp_min(1e-30)
        shrink = (1.0 - (2.0 * gain - 1.0) ** 3).clamp(min=1.0 / 3.0, max=1.0)
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
        normal, loss, terms = linearize(problem, x)
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
        normal_new, loss_new, terms_new = linearize(problem, candidate)
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

    A step is taken where it lowers the merit: the loss plus the constraint terms' summed violation, weighted by
    the largest multiplier of the step's program (what a unit of violation is worth in loss there). The damping
    follows the ratio of that decrease to the one the program predicted, by Nielsen's rule: a good ratio lowers
    it, a poor one raises it, also on a step that is taken. That is what stands in for the curvature of the
    constraint terms, which the rows do not carry: along a curved constraint that presses on the solution, the
    damping settles where the steps stop overshooting.

    The iterate may leave a curved constraint by what its rows do not see, and later steps bring it back.
    ``State.x`` is the last iterate that was within ``tolerance`` of every bound (in the terms' own units) or,
    before any was, the closest one: a solve started within ``tolerance`` returns a point within it after any
    number of steps. ``State.loss``, ``State.extra["violation"]`` and the ``violation`` diagnostic describe that
    point; the iterate and its loss are ``State.extra["iterate"]``, and ``normal`` and ``rows`` are linearized
    there. A problem is done once its iterate is within ``tolerance`` and either the gradient of the loss plus the
    rows' multipliers has no entry over ``gradient_tolerance`` or a step it took had none over ``step_tolerance``
    (LM's defaults; LM's test on the loss decrease has no place here, the merit being flat along a step that only
    repairs a row), or when its damping has reached the maximum.

    The normal equations must be dense (``Problem(dense=True)`` for a ``time=`` problem), and the problem needs a
    constraint term or bounds: without either, use `LevenbergMarquardt`.
    """

    warm_keys = ("damping", "growth", "qp")

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
        if tolerance < 0:
            raise ArgumentError(f"tolerance must be at least 0, got {tolerance}")
        self.damping, self.gain, self.tolerance = damping, gain, tolerance
        self.gradient_tolerance, self.step_tolerance = gradient_tolerance, step_tolerance

    def _program(
        self, problem: Problem, x: dict[str, torch.Tensor], rows: Rows | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rows ``(B, M, N)`` and bounds ``(B, M), (B, M)`` of the step from ``x``: the constraint terms at the
        gain, then the box of the problem's bounds."""
        matrices, lowers, uppers = [], [], []
        if rows is not None:
            matrices.append(rows.jacobian)
            lowers.append(self.gain * (rows.lower - rows.value))
            uppers.append(self.gain * (rows.upper - rows.value))
        bounds = problem.step_bounds(x)
        if bounds is not None:
            identity = torch.eye(bounds[0].shape[-1], dtype=bounds[0].dtype, device=bounds[0].device)
            matrices.append(identity.expand(bounds[0].shape[0], -1, -1))
            lowers.append(bounds[0])
            uppers.append(bounds[1])
        return torch.cat(matrices, dim=1), torch.cat(lowers, dim=1), torch.cat(uppers, dim=1)

    @override
    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        normal, loss, terms, rows = linearize_rows(problem, x)
        if not isinstance(normal, Normal):
            raise ArgumentError("SequentialQP takes dense normal equations; build the problem with dense=True")
        if rows is None and not problem.bounds:
            raise ArgumentError(
                "SequentialQP holds constraint terms and bounds, and the problem has neither; use LevenbergMarquardt"
            )
        damping = torch.full_like(loss, self.damping)
        matrix, lower, upper = self._program(problem, x, rows)
        # The first program is solved here, cold: a first state then has every key of a stepped one.
        solution = solve_qp(_damped(normal, damping), normal.g, rows=matrix, lower=lower, upper=upper)
        extra: dict[str, Any] = {
            "damping": damping,
            "growth": torch.full_like(loss, 2.0),
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
        damping, growth, normal, rows = (state.extra[key] for key in ("damping", "growth", "normal", "rows"))
        point, point_loss = state.extra["iterate"]
        matrix, lower, upper = self._program(problem, point, rows)
        solution = solve_qp(
            _damped(normal, damping), normal.g, rows=matrix, lower=lower, upper=upper, warm=state.extra["qp"]
        )
        delta = solution.x
        with torch.no_grad():
            candidate = problem.retract(point, delta)
        normal_new, loss_new, terms_new, rows_new = linearize_rows(problem, candidate)
        decrease, predicted = point_loss - loss_new, normal.predicted_decrease(delta)
        violation = violation_new = torch.zeros_like(loss_new)
        if rows is not None and rows_new is not None:
            penalty = solution.dual[:, : rows.value.shape[1]].abs().amax(-1)
            excess = _excess(rows, rows.value)
            linear = rows.value + (rows.jacobian @ delta.unsqueeze(-1)).squeeze(-1)
            decrease = decrease + penalty * (excess - _excess(rows_new, rows_new.value))
            predicted = predicted + penalty * (excess - _excess(rows, linear))
            violation, violation_new = rows.violation, rows_new.violation
        accept = (decrease > 0) & ~state.done
        ratio = decrease / predicted.clamp_min(1e-30)
        # Nielsen's factor whole: a taken step whose ratio is under a half raises the damping.
        factor = (1.0 - (2.0 * ratio - 1.0) ** 3).clamp(min=1.0 / 3.0, max=2.0)
        new_damping = torch.where(accept, damping * factor, damping * growth).clamp(1e-12, 1e12)
        gradient_tolerance, _, step_tolerance = _tolerances(
            state.loss.dtype, self.gradient_tolerance, None, self.step_tolerance
        )
        grad_norm = (normal.g + (solution.dual.unsqueeze(-2) @ matrix).squeeze(-2)).abs().amax(-1)
        step_norm = delta.abs().amax(-1)
        inside = torch.where(accept, violation_new, violation) <= self.tolerance
        settled = inside & ((grad_norm < gradient_tolerance) | (accept & (step_norm < step_tolerance)))
        reported = state.extra["violation"]
        # State.x follows the iterate while it is within the tolerance, or closer to the bounds than any before.
        report = accept & (violation_new <= reported.clamp_min(self.tolerance))
        return State(
            x=_select(report, candidate, state.x),
            loss=torch.where(report, loss_new, state.loss),
            done=state.done | settled | (new_damping >= 1e12),
            solver=type(self),
            iteration=state.iteration + 1,
            extra={
                "damping": new_damping,
                "growth": torch.where(accept, torch.full_like(growth, 2.0), growth * 2.0),
                "grad_norm": grad_norm,
                "step_norm": torch.where(accept, step_norm, torch.zeros_like(step_norm)),
                "accepted": accept,
                "violation": torch.where(report, violation_new, reported),
                "terms": _selected_terms(report, terms_new, state.extra["terms"]),
                "normal": _selected_normal(accept, normal_new, normal),
                "rows": None if rows is None else _selected_normal(accept, rows_new, rows),
                "qp": solution,
                "iterate": (_select(accept, candidate, point), torch.where(accept, loss_new, point_loss)),
            },
        )


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
    if bounds is None:
        return True
    lower, upper = bounds
    return ~(((lower >= -1e-9) & (g > 0)) | ((upper <= 1e-9) & (g < 0)))


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


__all__ = ["LBFGS", "GaussNewton", "LevenbergMarquardt", "SequentialQP", "Solver", "State", "TorchOptim"]
