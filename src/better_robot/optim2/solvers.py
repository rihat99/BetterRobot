"""Solvers: ``init(problem, x) -> State`` and ``step(problem, state) -> State``.

Every tensor in :class:`State` is batch-first. A step computes all batch elements and keeps
the old value where an element is done or its step was rejected, so shapes never change
and nothing inside ``step`` reads a tensor back to Python.
"""

from __future__ import annotations

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

    A problem is done when its gradient is below ``gtol``, or an accepted step changes the
    loss by less than ``ftol`` (relative) or the variables by less than ``xtol``.
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
            done=state.done | small_gradient | small_change,
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


class TorchOptim:
    """Run any ``torch.optim`` optimizer on the problem's loss.

    The optimizer steps the ambient tensors; afterwards they are projected onto their
    manifolds and bounds. ``State.loss`` is the loss before the step (no extra evaluation).
    Adam-like optimizers keep batch elements independent; ``torch.optim.LBFGS`` couples them
    through its line search and treats the batch as one problem.
    """

    def __init__(self, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any) -> None:
        self.optimizer_cls, self.kwargs = optimizer_cls, kwargs

    def init(self, problem: Problem, x: dict[str, torch.Tensor]) -> State:
        params = {name: value.detach().clone().requires_grad_(True) for name, value in x.items()}
        optimizer = self.optimizer_cls(list(params.values()), **self.kwargs)
        loss = next(iter(x.values())).new_full((next(iter(x.values())).shape[0],), float("inf"))
        extra = {"params": params, "optimizer": optimizer}
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
        with torch.no_grad():
            for name, value in problem.project(params).items():
                params[name].copy_(value)
        return State(x=params, loss=last["loss"], done=state.done, it=state.it + 1, extra=state.extra)

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
            gradient.mul_(mask.to(gradient.dtype).reshape(-1, *gradient.shape[1:]) if mask.ndim == 2
                          else mask.to(gradient.dtype).reshape(gradient.shape[1:]))


__all__ = ["GaussNewton", "LevenbergMarquardt", "Solver", "State", "TorchOptim", "select"]
