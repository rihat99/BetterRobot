"""Solvers: ``init(problem, x) -> State`` and ``step(problem, state) -> State``.

Every tensor in ``State`` is batch-first. A step computes all batch elements and keeps
the old value where an element is done or its step was rejected, so shapes never change.
``step`` itself reads no tensor back to Python, except the ``LBFGS`` line search: one flag
per trial, to stop once every problem has its step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from .problem import Problem


@dataclass
class State:
    """Optimizer state of a batch of problems.

    After a ``step``, ``extra`` holds the diagnostic keys of ``DIAGNOSTICS`` that the solver
    computes, each ``(B,)``: ``grad_norm`` (largest gradient entry before the step), ``step_norm``
    (largest entry of the step taken, 0 where none was taken), ``accepted`` (bool, the step
    moved the variables) and ``damping`` (LM only, damping for the next step). LM, Gauss-Newton
    and L-BFGS write the first three; ``TorchOptim`` writes none. ``run`` stacks them into
    ``Result.diagnostics``. ``warm_keys`` on a solver names the ``extra`` entries a warm start keeps.

    Every solver also writes ``terms``, ``{name: (B,)}``: the detached loss of each active term from
    the evaluation that ``loss`` comes from, so logging needs no second forward. ``TorchOptim`` adds
    ``info``, the ``{name: info}`` dicts of terms returning ``(value, info)`` (see ``Problem.evaluate``).
    """

    x: dict[str, torch.Tensor]  #: variables, each ``(B, *event)``.
    loss: torch.Tensor  #: ``(B,)`` loss at ``x``.
    done: torch.Tensor  #: ``(B,)`` bool, per-problem convergence.
    it: int = 0  #: steps taken so far.
    extra: dict[str, Any] = field(default_factory=dict)  #: solver-owned (damping, moments, ...).


DIAGNOSTICS = ("grad_norm", "step_norm", "accepted", "damping")


class Solver(Protocol):
    """Anything with ``init`` and ``step`` is a solver.

    ``solve`` and ``run`` call ``init`` once and then ``step`` up to the iteration budget, stopping
    early once every batch element is done (see ``Phase.stop_every``).
    """

    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        """Start state at variables ``x``: loss evaluated, nothing done, solver-owned ``extra`` set up."""

    def step(self, problem: Problem, state: State) -> State:
        """One iteration for every batch element; done or rejected elements keep their values."""


def select(mask: torch.Tensor, new: dict[str, torch.Tensor], old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per batch element, take ``new`` where ``mask: (B,)`` is true, else ``old``."""
    return {
        name: torch.where(mask.reshape(-1, *([1] * (value.ndim - 1))), value, old[name]) for name, value in new.items()
    }


def _selected_terms(
    mask: torch.Tensor, new: dict[str, torch.Tensor], old: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Per-term losses ``{name: (B,)}``: ``new`` where ``mask: (B,)`` is true, else ``old``."""
    return {name: torch.where(mask, value, old[name]) for name, value in new.items()}


def _tolerances(
    dtype: torch.dtype, gtol: float | None, ftol: float | None, xtol: float | None
) -> tuple[float, float, float]:
    """Fill unset tolerances with values the dtype can resolve (gradient, relative loss, step)."""
    defaults = (1e-8, 1e-12, 1e-10) if dtype == torch.float64 else (1e-7, 0.0, 0.0)
    return tuple(default if value is None else value for value, default in zip((gtol, ftol, xtol), defaults))


class LevenbergMarquardt:
    """Levenberg-Marquardt with per-problem damping (Nielsen update) and Marquardt scaling.

    A problem is done when its gradient is below ``gtol``, an accepted step changes the loss
    by less than ``ftol`` (relative) or the variables by less than ``xtol``, or its damping
    has reached the maximum (no step is accepted any more). ``damping`` is the initial damping.
    ``gtol`` bounds the largest gradient entry; a tolerance left at ``None`` takes a default the
    dtype can resolve (``1e-8, 1e-12, 1e-10`` in float64, ``1e-7, 0, 0`` otherwise).
    """

    warm_keys = ("damping", "growth")

    def __init__(
        self, *, damping: float = 1e-3, gtol: float | None = None, ftol: float | None = None, xtol: float | None = None
    ) -> None:
        self.damping, self.gtol, self.ftol, self.xtol = damping, gtol, ftol, xtol

    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        with torch.no_grad():
            loss, terms, _ = problem.evaluate(x)
        extra = {"damping": torch.full_like(loss, self.damping), "growth": torch.full_like(loss, 2.0), "terms": terms}
        return State(x=dict(x), loss=loss, done=torch.zeros_like(loss, dtype=torch.bool), extra=extra)

    def step(self, problem: Problem, state: State) -> State:
        damping, growth = state.extra["damping"], state.extra["growth"]
        normal, _ = problem.linearize(state.x)
        bounds = problem.step_bounds(state.x)
        if bounds is not None:
            normal = normal.restrict(*bounds)
        delta = normal.solve(damping)
        with torch.no_grad():
            candidate = problem.retract(state.x, delta)
            loss_new, terms_new, _ = problem.evaluate(candidate)
        decrease = state.loss - loss_new
        accept = (decrease > 0) & torch.isfinite(loss_new) & ~state.done
        gain = decrease / normal.predicted_decrease(delta).clamp_min(1e-30)
        shrink = (1.0 - (2.0 * gain - 1.0) ** 3).clamp(min=1.0 / 3.0, max=1.0)
        new_damping = torch.where(accept, damping * shrink, damping * growth).clamp(1e-12, 1e12)
        new_growth = torch.where(accept, torch.full_like(growth, 2.0), growth * 2.0)
        gtol, ftol, xtol = _tolerances(state.loss.dtype, self.gtol, self.ftol, self.xtol)
        grad_norm, step_norm = normal.g.abs().amax(-1), delta.abs().amax(-1)
        small_gradient = grad_norm < gtol
        small_change = accept & ((decrease <= ftol * state.loss.abs()) | (step_norm < xtol))
        return State(
            x=select(accept, candidate, state.x),
            loss=torch.where(accept, loss_new, state.loss),
            done=state.done | small_gradient | small_change | (new_damping >= 1e12),
            it=state.it + 1,
            extra={
                "damping": new_damping,
                "growth": new_growth,
                "grad_norm": grad_norm,
                "step_norm": torch.where(accept, step_norm, torch.zeros_like(step_norm)),
                "accepted": accept,
                "terms": _selected_terms(accept, terms_new, state.extra["terms"]),
            },
        )


class GaussNewton:
    """Gauss-Newton: full step of the linearized problem, with optional fixed damping."""

    def __init__(self, *, damping: float = 0.0, gtol: float | None = None, xtol: float | None = None) -> None:
        self.damping, self.gtol, self.xtol = damping, gtol, xtol

    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        with torch.no_grad():
            loss, terms, _ = problem.evaluate(x)
        return State(x=dict(x), loss=loss, done=torch.zeros_like(loss, dtype=torch.bool), extra={"terms": terms})

    def step(self, problem: Problem, state: State) -> State:
        normal, _ = problem.linearize(state.x)
        bounds = problem.step_bounds(state.x)
        if bounds is not None:
            normal = normal.restrict(*bounds)
        delta = normal.solve(self.damping)
        with torch.no_grad():
            candidate = problem.retract(state.x, delta)
            loss_new, terms_new, _ = problem.evaluate(candidate)
        move = torch.isfinite(loss_new) & ~state.done
        gtol, _, xtol = _tolerances(state.loss.dtype, self.gtol, None, self.xtol)
        grad_norm, step_norm = normal.g.abs().amax(-1), delta.abs().amax(-1)
        done = state.done | (grad_norm < gtol) | (step_norm < xtol)
        extra = {"grad_norm": grad_norm, "step_norm": torch.where(move, step_norm, torch.zeros_like(step_norm))}
        return State(
            x=select(move, candidate, state.x),
            loss=torch.where(move, loss_new, state.loss),
            done=done,
            it=state.it + 1,
            extra={**extra, "accepted": move, "terms": _selected_terms(move, terms_new, state.extra["terms"])},
        )


def _free(g: torch.Tensor, bounds: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor | bool:
    """``(B, n)`` mask of the coordinates not held by an active step bound that the gradient ``g`` pushes against."""
    if bounds is None:
        return True
    lower, upper = bounds
    return ~(((lower >= -1e-9) & (g > 0)) | ((upper <= 1e-9) & (g < 0)))


def _projected(g: torch.Tensor, bounds: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
    """Gradient ``(B, n)`` with the entries that push against an active step bound zeroed."""
    return g * _free(g, bounds)


class LBFGS:
    """Batched L-BFGS in tangent coordinates: per-problem history and backtracking line search.

    Every step tries the step sizes 1, ½, ¼, ... (at most ``line_search`` of them) and each problem
    takes the largest one that satisfies the Armijo condition on the step actually taken; the search
    stops as soon as every problem has its step (one host read per trial). The plain gradient step
    (no history yet) has length ``1 / |g|_1``, a guess: a stiff term that adds curvature but little
    gradient (a residual near zero) can need 10 or more halvings of it. With bounds, a coordinate
    held by an active bound that the gradient pushes against does not move and every step is clipped
    to the bounds. A problem that finds no step keeps its variables and drops its history; when that
    happens on a plain gradient step (no history to drop) it is done, as every later step would repeat
    the same search. ``history`` is the number of stored curvature pairs, ``armijo`` the
    sufficient-decrease constant and ``gtol`` bounds the largest gradient entry (``None`` takes the
    dtype default, as for LM).
    """

    def __init__(
        self, *, history: int = 10, line_search: int = 20, armijo: float = 1e-4, gtol: float | None = None
    ) -> None:
        self.history, self.line_search, self.armijo, self.gtol = history, line_search, armijo, gtol

    warm_keys = ("s", "y", "rho")

    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        loss, g = problem.gradient(x)
        with torch.no_grad():
            _, terms, _ = problem.evaluate(x)
        memory = g.new_zeros(g.shape[0], self.history, g.shape[1])
        extra = {"g": g, "s": memory, "y": memory.clone(), "rho": g.new_zeros(g.shape[0], self.history), "terms": terms}
        return State(x=dict(x), loss=loss, done=torch.zeros_like(loss, dtype=torch.bool), extra=extra)

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
                loss, terms, _ = problem.evaluate(candidate)
                # Armijo on the step actually taken: a bound may have shortened it, even into a direction whose linear
                # model rises (slope > 0); the clamp then asks for no increase at all.
                slope = (g * step).sum(-1).clamp_max(0.0)
                ok = (loss <= state.loss + self.armijo * slope) & torch.isfinite(loss) & ~found
                ok &= ~state.done
                best_x = select(ok, candidate, best_x)
                best_loss = torch.where(ok, loss, best_loss)
                best_step = torch.where(ok.unsqueeze(-1), step, best_step)
                best_terms = _selected_terms(ok, terms, best_terms)
                found |= ok
                if bool((found | state.done).all()):  # bench-ok: later trials could change nothing
                    break
        _, g_new = problem.gradient(best_x)
        change = g_new - state.extra["g"]
        curvature = (best_step * change).sum(-1)
        keep = found & (curvature > 1e-10 * (change * change).sum(-1))
        pushed = lambda memory, new: torch.cat([memory[:, 1:], new.unsqueeze(1)], dim=1)  # noqa: E731
        s_new = torch.where(keep[:, None, None], pushed(s, best_step), s)
        y_new = torch.where(keep[:, None, None], pushed(y, change), y)
        rho_new = torch.where(keep[:, None], pushed(rho, 1.0 / curvature.clamp_min(1e-30)), rho)
        rho_new = torch.where(found[:, None], rho_new, torch.zeros_like(rho_new))  # failed search: forget history
        gtol, _, _ = _tolerances(g.dtype, self.gtol, None, None)
        grad_norm = _projected(g_new, problem.step_bounds(best_x)).abs().amax(-1)  # at the new point
        done = state.done | (grad_norm < gtol) | (first_step & ~found)
        extra = {"g": g_new, "s": s_new, "y": y_new, "rho": rho_new, "grad_norm": grad_norm}
        extra.update(step_norm=best_step.abs().amax(-1), accepted=found, terms=best_terms)
        return State(x=best_x, loss=best_loss, done=done, it=state.it + 1, extra=extra)


class TorchOptim:
    """Run any ``torch.optim`` optimizer on the problem's loss.

    The optimizer steps the ambient tensors; afterwards they are projected onto their
    manifolds and bounds. ``State.loss`` is the loss before the step (no extra evaluation);
    ``extra["terms"]`` and ``extra["info"]`` come from the same, last closure evaluation.
    Adam-like optimizers keep batch elements independent; ``torch.optim.LBFGS`` couples them
    through its line search and treats the batch as one problem.
    ``scheduler(optimizer)`` may return a ``torch.optim.lr_scheduler`` object; its ``step()``
    is called without arguments once per solver step. Extra keyword arguments go to
    ``optimizer_cls``, e.g. ``TorchOptim(torch.optim.Adam, lr=1e-2)``.
    """

    def __init__(
        self,
        optimizer_cls: type[torch.optim.Optimizer],
        *,
        scheduler: Callable[[torch.optim.Optimizer], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.optimizer_cls, self.scheduler, self.kwargs = optimizer_cls, scheduler, kwargs

    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        params = {name: value.detach().clone().requires_grad_(True) for name, value in x.items()}
        optimizer = self.optimizer_cls(list(params.values()), **self.kwargs)
        loss = next(iter(x.values())).new_full((next(iter(x.values())).shape[0],), float("inf"))
        extra = {"params": params, "optimizer": optimizer}
        if self.scheduler is not None:
            extra["scheduler"] = self.scheduler(optimizer)
        return State(x=params, loss=loss, done=torch.zeros_like(loss, dtype=torch.bool), extra=extra)

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
        if "scheduler" in state.extra:
            state.extra["scheduler"].step()
        with torch.no_grad():
            for name, value in problem.project(params).items():
                params[name].copy_(value)
        current = {name: value.detach().clone() for name, value in params.items()}
        extra = {**state.extra, "terms": last["terms"], "info": last["info"]}
        return State(x=current, loss=last["loss"], done=state.done, it=state.it + 1, extra=extra)

    @staticmethod
    def _mask_gradients(problem: Problem, params: dict[str, torch.Tensor]) -> None:
        for name, mask in (problem.free or {}).items():
            gradient = params[name].grad
            if gradient is None:
                continue
            if gradient.shape[1:].numel() != mask.shape[-1]:
                raise ValueError(
                    f"free mask of {name!r} is in tangent coordinates ({mask.shape[-1]}) but the variable has "
                    f"{gradient.shape[1:].numel()} ambient coordinates; use a second-order solver for this phase"
                )
            gradient.mul_(
                mask.to(gradient.dtype).reshape(-1, *gradient.shape[1:])
                if mask.ndim == 2
                else mask.to(gradient.dtype).reshape(gradient.shape[1:])
            )


__all__ = ["LBFGS", "GaussNewton", "LevenbergMarquardt", "Solver", "State", "TorchOptim", "select"]
