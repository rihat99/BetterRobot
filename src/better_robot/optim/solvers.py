"""Solvers: ``init(problem, x) -> State`` and ``step(problem, state) -> State``.

Every tensor in :class:`State` is batch-first. A step computes all batch elements and keeps
the old value where an element is done or its step was rejected, so shapes never change
and nothing inside ``step`` reads a tensor back to Python.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from .problem import Problem


@dataclass
class State:
    """Optimizer state of a batch of problems."""

    x: dict[str, torch.Tensor]
    loss: torch.Tensor  #: ``(B,)`` loss at ``x``.
    done: torch.Tensor  #: ``(B,)`` bool, per-problem convergence.
    it: int = 0
    extra: dict[str, Any] = field(default_factory=dict)  #: solver-owned (damping, moments, ...).


class Solver(Protocol):
    """Anything with ``init`` and ``step`` is a solver."""

    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State: ...

    def step(self, problem: Problem, state: State) -> State: ...


def select(mask: torch.Tensor, new: dict[str, torch.Tensor], old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per batch element, take ``new`` where ``mask: (B,)`` is true, else ``old``."""
    return {
        name: torch.where(mask.reshape(-1, *([1] * (value.ndim - 1))), value, old[name]) for name, value in new.items()
    }


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
    has reached the maximum (no step is accepted any more).
    """

    def __init__(
        self, *, damping: float = 1e-3, gtol: float | None = None, ftol: float | None = None, xtol: float | None = None
    ) -> None:
        self.damping, self.gtol, self.ftol, self.xtol = damping, gtol, ftol, xtol

    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        with torch.no_grad():
            loss = problem.loss(x)
        extra = {"damping": torch.full_like(loss, self.damping), "growth": torch.full_like(loss, 2.0)}
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
            loss_new = problem.loss(candidate)
        decrease = state.loss - loss_new
        accept = (decrease > 0) & torch.isfinite(loss_new) & ~state.done
        gain = decrease / normal.predicted_decrease(delta).clamp_min(1e-30)
        shrink = (1.0 - (2.0 * gain - 1.0) ** 3).clamp(min=1.0 / 3.0, max=1.0)
        new_damping = torch.where(accept, damping * shrink, damping * growth).clamp(1e-12, 1e12)
        new_growth = torch.where(accept, torch.full_like(growth, 2.0), growth * 2.0)
        gtol, ftol, xtol = _tolerances(state.loss.dtype, self.gtol, self.ftol, self.xtol)
        small_gradient = normal.g.abs().amax(-1) < gtol
        small_change = accept & ((decrease <= ftol * state.loss.abs()) | (delta.abs().amax(-1) < xtol))
        return State(
            x=select(accept, candidate, state.x),
            loss=torch.where(accept, loss_new, state.loss),
            done=state.done | small_gradient | small_change | (new_damping >= 1e12),
            it=state.it + 1,
            extra={"damping": new_damping, "growth": new_growth},
        )


class GaussNewton:
    """Gauss-Newton: full step of the linearized problem, with optional fixed damping."""

    def __init__(self, *, damping: float = 0.0, gtol: float | None = None, xtol: float | None = None) -> None:
        self.damping, self.gtol, self.xtol = damping, gtol, xtol

    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        with torch.no_grad():
            loss = problem.loss(x)
        return State(x=dict(x), loss=loss, done=torch.zeros_like(loss, dtype=torch.bool))

    def step(self, problem: Problem, state: State) -> State:
        normal, _ = problem.linearize(state.x)
        bounds = problem.step_bounds(state.x)
        if bounds is not None:
            normal = normal.restrict(*bounds)
        delta = normal.solve(self.damping)
        with torch.no_grad():
            candidate = problem.retract(state.x, delta)
            loss_new = problem.loss(candidate)
        move = torch.isfinite(loss_new) & ~state.done
        gtol, _, xtol = _tolerances(state.loss.dtype, self.gtol, None, self.xtol)
        done = state.done | (normal.g.abs().amax(-1) < gtol) | (delta.abs().amax(-1) < xtol)
        return State(
            x=select(move, candidate, state.x), loss=torch.where(move, loss_new, state.loss), done=done, it=state.it + 1
        )


class LBFGS:
    """Batched L-BFGS in tangent coordinates: per-problem history and backtracking line search.

    Every step evaluates the loss at ``line_search`` step sizes (1, ½, ¼, ...) for all
    problems and each problem takes the largest one that satisfies the Armijo condition.
    A problem that finds none keeps its variables and drops its history.
    """

    def __init__(
        self, *, history: int = 10, line_search: int = 8, armijo: float = 1e-4, gtol: float | None = None
    ) -> None:
        self.history, self.line_search, self.armijo, self.gtol = history, line_search, armijo, gtol

    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        loss, g = problem.gradient(x)
        memory = g.new_zeros(g.shape[0], self.history, g.shape[1])
        extra = {"g": g, "s": memory, "y": memory.clone(), "rho": g.new_zeros(g.shape[0], self.history)}
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
        if bounds is not None:
            lower, upper = bounds
            g = g * ~(((lower >= -1e-9) & (g > 0)) | ((upper <= 1e-9) & (g < 0)))
        direction = self._direction(g, s, y, rho)
        slope = (g * direction).sum(-1)
        uphill = slope >= 0
        direction = torch.where(uphill.unsqueeze(-1), -g, direction)
        slope = torch.where(uphill, -(g * g).sum(-1), slope)
        first_step = (rho[:, -1] == 0) | uphill  # no curvature yet: scale the gradient step
        scale = torch.where(first_step, 1.0 / g.abs().sum(-1).clamp_min(1.0), 1.0)
        best_x, best_loss, best_step = state.x, state.loss, torch.zeros_like(g)
        found = torch.zeros_like(state.done)
        with torch.no_grad():
            for trial in range(self.line_search):
                step = direction * (scale * 0.5**trial).unsqueeze(-1)
                if bounds is not None:
                    step = torch.minimum(torch.maximum(step, lower), upper)
                candidate = problem.retract(state.x, step)
                loss = problem.loss(candidate)
                ok = (loss <= state.loss + self.armijo * 0.5**trial * scale * slope) & torch.isfinite(loss) & ~found
                ok &= ~state.done
                best_x = select(ok, candidate, best_x)
                best_loss = torch.where(ok, loss, best_loss)
                best_step = torch.where(ok.unsqueeze(-1), step, best_step)
                found |= ok
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
        done = state.done | (g.abs().amax(-1) < gtol)
        extra = {"g": g_new, "s": s_new, "y": y_new, "rho": rho_new}
        return State(x=best_x, loss=best_loss, done=done, it=state.it + 1, extra=extra)


class TorchOptim:
    """Run any ``torch.optim`` optimizer on the problem's loss.

    The optimizer steps the ambient tensors; afterwards they are projected onto their
    manifolds and bounds. ``State.loss`` is the loss before the step (no extra evaluation).
    Adam-like optimizers keep batch elements independent; ``torch.optim.LBFGS`` couples them
    through its line search and treats the batch as one problem.
    ``scheduler(optimizer)`` may return a ``torch.optim.lr_scheduler`` object; its ``step()``
    is called without arguments once per solver step.
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
            loss = problem.loss(params)
            total = loss.sum()
            total.backward()
            self._mask_gradients(problem, params)
            last["loss"] = loss.detach()
            return total

        optimizer.step(closure)
        if "scheduler" in state.extra:
            state.extra["scheduler"].step()
        with torch.no_grad():
            for name, value in problem.project(params).items():
                params[name].copy_(value)
        current = {name: value.detach().clone() for name, value in params.items()}
        return State(x=current, loss=last["loss"], done=state.done, it=state.it + 1, extra=state.extra)

    @staticmethod
    def _mask_gradients(problem: Problem, params: dict[str, torch.Tensor]) -> None:
        for name, mask in (problem.free or {}).items():
            gradient = params[name].grad
            if gradient is None:
                continue
            if gradient[0].numel() != mask.shape[-1]:
                raise ValueError(
                    f"free mask of {name!r} is in tangent coordinates ({mask.shape[-1]}) but the variable has "
                    f"{gradient[0].numel()} ambient coordinates; use a second-order solver for this phase"
                )
            gradient.mul_(
                mask.to(gradient.dtype).reshape(-1, *gradient.shape[1:])
                if mask.ndim == 2
                else mask.to(gradient.dtype).reshape(gradient.shape[1:])
            )


__all__ = ["LBFGS", "GaussNewton", "LevenbergMarquardt", "Solver", "State", "TorchOptim", "select"]
