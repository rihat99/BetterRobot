"""Matrix-free batched Adam for named product-manifold problems.

The update differentiates :meth:`Problem.objective` through reduced tangent
retractions and never assembles Jacobian blocks or a dense Jacobian. All
per-element decisions use tensor masks; only the eager :meth:`Adam.run` loop
performs a host-side all-terminal check. The default driver detaches values and
state between iterations.

Batched LBFGS is intentionally not implemented. Per-element histories, line
search, and curvature-validity/history-reset rules remain deferred; use this
first-order solver or named-block LM/GN instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Real
from typing import NamedTuple

import torch

from ._solver_common import _batch_shape, _blend_values
from .problem import Problem
from .variables import Values, detach_values


class AdamStatus(IntEnum):
    """Per-element terminal status stored as ``torch.int8``."""

    RUNNING = 0
    CONVERGED = 1
    MAXITER = 2
    FAILED = 3


class AdamState(NamedTuple):
    """Fixed tensor-pytree state for arbitrary leading batch axes.

    ``m`` and ``v`` are keyed exactly like ``Values`` and every tensor ends in
    that block's mask-reduced tangent dimension. ``step`` is retained across a
    compatible warm start because Adam's bias correction depends on it; cost,
    gradient diagnostics, and terminal status are refreshed at the new point.
    """

    m: Values
    v: Values
    step: torch.Tensor
    cost: torch.Tensor
    grad_norm: torch.Tensor
    converged: torch.Tensor
    status: torch.Tensor


def _gradient_stats(
    gradient: Values,
    problem: Problem,
    cost: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    norms = [gradient[spec.name].abs().amax(dim=-1) for spec in problem.vars if spec.free_dim]
    finite = [torch.isfinite(gradient[spec.name]).all(dim=-1) for spec in problem.vars]
    grad_norm = torch.stack(norms, dim=-1).amax(dim=-1)
    all_finite = torch.stack(finite, dim=-1).all(dim=-1) & torch.isfinite(cost)
    return grad_norm, all_finite


def _terminal_status(
    cost: torch.Tensor,
    grad_norm: torch.Tensor,
    finite: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    running = torch.full_like(cost, AdamStatus.RUNNING.value, dtype=torch.int8)
    converged = torch.full_like(running, AdamStatus.CONVERGED.value)
    failed = torch.full_like(running, AdamStatus.FAILED.value)
    return torch.where(~finite, failed, torch.where(grad_norm <= tolerance, converged, running))


def _values_finite(
    values: Values,
    problem: Problem,
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    finite = [torch.isfinite(values[spec.name].reshape(*batch_shape, -1)).all(dim=-1) for spec in problem.vars]
    return torch.stack(finite, dim=-1).all(dim=-1)


def _detach_state(state: AdamState) -> AdamState:
    return state._replace(
        m=detach_values(state.m),
        v=detach_values(state.v),
        step=state.step.detach(),
        cost=state.cost.detach(),
        grad_norm=state.grad_norm.detach(),
        converged=state.converged.detach(),
        status=state.status.detach(),
    )


@dataclass(frozen=True)
class Adam:
    """Batched matrix-free Adam over named manifold blocks.

    Hyperparameters are immutable, while all mutable quantities live in
    :class:`AdamState`. :meth:`update` uses only the prevalidated objective VJP
    and feasible manifold retraction; it does not call any Jacobian assembly
    API. :meth:`run` is the graph-free eager driver.
    """

    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    tol: float = 1e-6
    max_iter: int = 100

    def __post_init__(self) -> None:
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("max_iter must be a non-negative int")
        for name in ("lr", "beta1", "beta2", "eps", "tol"):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                raise TypeError(f"{name} is a frozen static hyperparameter, not a tensor")
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite Python number")
        if self.lr <= 0.0:
            raise ValueError("lr must be > 0")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta1 and beta2 must satisfy 0 <= beta < 1")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")
        if self.tol < 0.0:
            raise ValueError("tol must be >= 0")

    def _evaluate(
        self,
        values: Values,
        problem: Problem,
        batch_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, Values, torch.Tensor, torch.Tensor]:
        cost, gradient = problem._objective_gradient_prevalidated(
            values,
            batch_shape=batch_shape,
        )
        grad_norm, finite = _gradient_stats(gradient, problem, cost)
        return cost, gradient, grad_norm, finite

    def init_state(self, values: Values, problem: Problem) -> AdamState:
        """Validate one solve boundary and initialize reduced moments."""
        if not problem.residuals and not problem.objectives:
            raise ValueError("Adam requires at least one residual or scalar objective term")
        if problem.tangent_dim_total <= 0:
            raise ValueError("Adam requires at least one free tangent coordinate")
        batch_shape = problem._validate_values(values)
        problem._validate_runtime_weights(
            None,
            batch_shape=batch_shape,
            exemplar=values[problem.vars[0].name],
        )
        cost, gradient, grad_norm, finite = self._evaluate(values, problem, batch_shape)
        status = _terminal_status(cost, grad_norm, finite, self.tol)
        return AdamState(
            m={name: torch.zeros_like(value) for name, value in gradient.items()},
            v={name: torch.zeros_like(value) for name, value in gradient.items()},
            step=torch.zeros_like(cost, dtype=torch.int64),
            cost=cost,
            grad_norm=grad_norm,
            converged=status == AdamStatus.CONVERGED.value,
            status=status,
        )

    def update(
        self,
        values: Values,
        state: AdamState,
        problem: Problem,
    ) -> tuple[Values, AdamState]:
        """Apply one pure, host-sync-free matrix-free Adam update."""
        batch_shape = _batch_shape(values, problem)
        cost, gradient, grad_norm, finite = self._evaluate(values, problem, batch_shape)
        evaluated_status = _terminal_status(cost, grad_norm, finite, self.tol)
        was_running = state.status == AdamStatus.RUNNING.value
        current_status = torch.where(was_running, evaluated_status, state.status)
        movable = current_status == AdamStatus.RUNNING.value
        trial_step = state.step + movable.to(dtype=state.step.dtype)
        bias_step = trial_step.clamp(min=1).to(dtype=cost.dtype)
        bias1 = 1.0 - torch.pow(self.beta1, bias_step)
        bias2 = 1.0 - torch.pow(self.beta2, bias_step)

        trial_m: Values = {}
        trial_v: Values = {}
        deltas: Values = {}
        for spec in problem.vars:
            name = spec.name
            block_gradient = gradient[name]
            mask = movable.unsqueeze(-1)
            moment1 = self.beta1 * state.m[name] + (1.0 - self.beta1) * block_gradient
            moment2 = self.beta2 * state.v[name] + (1.0 - self.beta2) * block_gradient.square()
            moment1 = torch.where(mask, moment1, state.m[name])
            moment2 = torch.where(mask, moment2, state.v[name])
            corrected1 = moment1 / bias1.unsqueeze(-1)
            corrected2 = moment2 / bias2.unsqueeze(-1)
            delta = -self.lr * corrected1 / (torch.sqrt(corrected2) + self.eps)
            trial_m[name] = moment1
            trial_v[name] = moment2
            deltas[name] = torch.where(mask, delta, torch.zeros_like(delta))

        trial_values = problem._retract_prevalidated(
            values,
            deltas,
            batch_shape=batch_shape,
        )
        trial_cost, _trial_gradient, trial_grad_norm, trial_finite = self._evaluate(
            trial_values,
            problem,
            batch_shape,
        )
        trial_finite = trial_finite & _values_finite(trial_values, problem, batch_shape)
        accepted = movable & trial_finite
        next_values = _blend_values(accepted, trial_values, values)
        next_m = {name: torch.where(accepted.unsqueeze(-1), trial_m[name], state.m[name]) for name in trial_m}
        next_v = {name: torch.where(accepted.unsqueeze(-1), trial_v[name], state.v[name]) for name in trial_v}
        trial_status = _terminal_status(
            trial_cost,
            trial_grad_norm,
            trial_finite,
            self.tol,
        )
        next_status = torch.where(
            accepted,
            trial_status,
            torch.where(
                movable & ~trial_finite,
                torch.full_like(current_status, AdamStatus.FAILED.value),
                current_status,
            ),
        )
        current_cost = torch.where(was_running, cost, state.cost)
        current_grad_norm = torch.where(was_running, grad_norm, state.grad_norm)
        next_state = AdamState(
            m=next_m,
            v=next_v,
            step=torch.where(accepted, trial_step, state.step),
            cost=torch.where(accepted, trial_cost, current_cost),
            grad_norm=torch.where(accepted, trial_grad_norm, current_grad_norm),
            converged=next_status == AdamStatus.CONVERGED.value,
            status=next_status,
        )
        return next_values, next_state

    @staticmethod
    def _validate_warm_start(fresh: AdamState, state: AdamState, problem: Problem) -> None:
        if not isinstance(state, AdamState):
            raise TypeError("warm-start state must be an AdamState")
        names = tuple(spec.name for spec in problem.vars)
        if tuple(state.m) != names or tuple(state.v) != names:
            raise ValueError("warm-start moments must match Problem variable names and order")
        for label, actual, expected in (("m", state.m, fresh.m), ("v", state.v, fresh.v)):
            for name in names:
                if actual[name].shape != expected[name].shape:
                    raise ValueError(
                        f"warm-start {label}[{name!r}] shape {tuple(actual[name].shape)} "
                        f"does not match expected reduced shape {tuple(expected[name].shape)}"
                    )
                if actual[name].dtype != expected[name].dtype or actual[name].device != expected[name].device:
                    raise ValueError(
                        f"warm-start {label}[{name!r}] must share Values' dtype/device "
                        f"{expected[name].dtype}/{expected[name].device}"
                    )
        for name in ("step", "cost", "grad_norm", "converged", "status"):
            actual = getattr(state, name)
            expected = getattr(fresh, name)
            if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
                raise ValueError(f"warm-start {name} must have batch shape {tuple(expected.shape)}")
            if actual.dtype != expected.dtype or actual.device != expected.device:
                raise ValueError(f"warm-start {name} must have dtype/device {expected.dtype}/{expected.device}")
        if bool((state.step < 0).any()):  # bench-ok: eager warm-start boundary validation
            raise ValueError("warm-start step must be non-negative")

    def _warm_start(self, values: Values, state: AdamState, problem: Problem) -> AdamState:
        fresh = self.init_state(values, problem)
        self._validate_warm_start(fresh, state, problem)
        return fresh._replace(
            m=detach_values(state.m),
            v=detach_values(state.v),
            step=state.step.detach(),
        )

    def _refresh(self, values: Values, state: AdamState, problem: Problem) -> AdamState:
        batch_shape = _batch_shape(values, problem)
        cost, _gradient, grad_norm, finite = self._evaluate(values, problem, batch_shape)
        evaluated = _terminal_status(cost, grad_norm, finite, self.tol)
        status = torch.where(state.status == AdamStatus.RUNNING.value, evaluated, state.status)
        return state._replace(
            cost=cost,
            grad_norm=grad_norm,
            converged=status == AdamStatus.CONVERGED.value,
            status=status,
        )

    def run(
        self,
        values: Values,
        problem: Problem,
        state: AdamState | None = None,
    ) -> tuple[Values, AdamState]:
        """Run the detached eager loop, optionally retaining compatible moments."""
        current_values = detach_values(values)
        current_state = (
            self.init_state(current_values, problem)
            if state is None
            else self._warm_start(current_values, state, problem)
        )
        for _ in range(self.max_iter):
            if bool((current_state.status != AdamStatus.RUNNING.value).all()):  # bench-ok: eager loop boundary sync
                break
            current_values, current_state = self.update(current_values, current_state, problem)
            current_values = detach_values(current_values)
            current_state = _detach_state(current_state)
        current_state = self._refresh(current_values, current_state, problem)
        running = current_state.status == AdamStatus.RUNNING.value
        status = torch.where(
            running,
            torch.full_like(current_state.status, AdamStatus.MAXITER.value),
            current_state.status,
        )
        current_state = current_state._replace(
            converged=status == AdamStatus.CONVERGED.value,
            status=status,
        )
        return detach_values(current_values), _detach_state(current_state)


__all__ = ["Adam", "AdamState", "AdamStatus"]
