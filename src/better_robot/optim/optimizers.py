"""Torch-style optimizer drivers for object-referenced problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Real
from typing import Any

import torch

from .problem import Problem


class OptimizerStatus(IntEnum):
    """Per-element optimizer status stored in an ``int8`` tensor."""

    RUNNING = 0
    CONVERGED = 1
    STALLED_AT_BOUNDS = 2
    MAXITER = 3
    FAILED = 4


@dataclass(frozen=True)
class OptimizerInfo:
    """Minimal per-element result shared by every optimizer."""

    status: torch.Tensor
    iterations: torch.Tensor
    cost: torch.Tensor

    @property
    def converged(self) -> torch.Tensor:
        return (self.status == OptimizerStatus.CONVERGED) | (self.status == OptimizerStatus.STALLED_AT_BOUNDS)


class Optimizer(ABC):
    """Base class for optimizers that own a :class:`Problem`."""

    def __init__(
        self,
        problem: Problem,
        *,
        max_iterations: int = 50,
        tolerance: float = 1e-8,
    ) -> None:
        if not isinstance(problem, Problem):
            raise TypeError(f"problem must be a Problem, got {type(problem).__name__}")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
            raise ValueError(f"max_iterations must be a non-negative int, got {max_iterations!r}")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, Real)
            or not math.isfinite(float(tolerance))
            or float(tolerance) < 0.0
        ):
            raise ValueError(f"tolerance must be a finite non-negative number, got {tolerance!r}")
        self.problem = problem
        self.max_iterations = max_iterations
        self.tolerance = float(tolerance)

    @abstractmethod
    def step(self) -> OptimizerInfo:
        """Run one optimizer iteration and update referenced variables."""

    def optimize(self, *, verbose: bool = False) -> OptimizerInfo:
        """Run until terminal status or the configured iteration limit."""
        info: OptimizerInfo | None = None
        for _ in range(self.max_iterations):
            info = self.step()
            if verbose:
                converged = int(info.converged.sum().detach().cpu())  # bench-ok: eager verbose reporting
                total = info.converged.numel()
                print(
                    f"iteration={int(info.iterations.max().detach().cpu())} "  # bench-ok: eager verbose reporting
                    f"cost={float(info.cost.sum().detach().cpu()):.6g} "  # bench-ok: eager verbose reporting
                    f"converged={converged}/{total}"
                )
            if bool((info.status != OptimizerStatus.RUNNING).all()):  # bench-ok: eager driver termination
                break
        if info is None:
            info = self._initial_info()
        return info

    @abstractmethod
    def reset(self) -> None:
        """Clear optimizer state while retaining current variable values."""

    @abstractmethod
    def resume(self) -> None:
        """Return terminal elements to running while retaining compatible state."""

    @abstractmethod
    def _initial_info(self) -> OptimizerInfo:
        """Return current information before a step has run."""


_TorchOptimizerFactory = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]


class TorchOptimizer(Optimizer):
    """Persistent ``torch.optim`` state over rebased tangent buffers.

    Same-layout input updates preserve the inner optimizer and optional
    scheduler. ``resume()`` restarts terminal elements without clearing their
    cumulative iterations, moments, buffers, or scheduler state. A scheduler
    factory must return a no-argument-step ``LRScheduler`` and is advanced once
    after each public step that performs an optimizer update.

    ``torch.optim.LBFGS`` uses one closure over the summed batch objective;
    its line search and history therefore couple batch elements.

    ``step()`` reports the objective from the same forward used for gradients
    (the iterate entering the step); ``optimize()`` refreshes the reported
    cost at the final variables.
    """

    def __init__(
        self,
        problem: Problem,
        optimizer_cls: type[torch.optim.Optimizer] | _TorchOptimizerFactory,
        *,
        max_iterations: int = 100,
        tolerance: float = 0.0,
        scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None = None,
        **optimizer_kwargs: Any,
    ) -> None:
        super().__init__(problem, max_iterations=max_iterations, tolerance=tolerance)
        if not callable(optimizer_cls):
            raise TypeError(f"optimizer_cls must be callable, got {type(optimizer_cls).__name__}")
        if scheduler is not None and not callable(scheduler):
            raise TypeError(f"scheduler must be callable or None, got {type(scheduler).__name__}")
        self.optimizer_cls = optimizer_cls
        self.scheduler_factory = scheduler
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self._buffers: dict[str, torch.nn.Parameter] = {}
        self._zero_gradients: dict[str, torch.Tensor] = {}
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._status: torch.Tensor | None = None
        self._iterations: torch.Tensor | None = None
        self._cost: torch.Tensor | None = None
        self._evaluated_cost: torch.Tensor | None = None
        self._layout_key: tuple[object, ...] | None = None
        self.reset()

    def _layout(self) -> tuple[tuple[int, ...], torch.Tensor]:
        self.problem._freeze()
        if not self.problem.residuals or self.problem.tangent_dim_total <= 0:
            raise ValueError("TorchOptimizer requires a residual and a free tangent coordinate")
        return self.problem._batch_and_exemplar()

    def reset(self) -> None:
        batch_shape, exemplar = self._layout()
        self._buffers = {
            variable.name: torch.nn.Parameter(exemplar.new_zeros(*batch_shape, variable.free_dim))
            for variable in self.problem.vars
        }
        self._zero_gradients = {name: torch.zeros_like(buffer) for name, buffer in self._buffers.items()}
        optimizer = self.optimizer_cls(self._buffers.values(), **self.optimizer_kwargs)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"optimizer_cls must return a torch.optim.Optimizer, got {type(optimizer).__name__}")
        self._optimizer = optimizer
        scheduler = self.scheduler_factory(optimizer) if self.scheduler_factory is not None else None
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            raise TypeError(
                f"scheduler must return a torch.optim.lr_scheduler.LRScheduler, got {type(scheduler).__name__}"
            )
        self._scheduler = scheduler
        self._status = exemplar.new_full(batch_shape, OptimizerStatus.RUNNING, dtype=torch.int8)
        self._iterations = exemplar.new_zeros(batch_shape, dtype=torch.int64)
        with torch.no_grad():
            self._cost = self.problem.objective().detach()
        self._layout_key = self._current_layout_key(batch_shape, exemplar)

    def _current_layout_key(self, batch_shape: tuple[int, ...], exemplar: torch.Tensor) -> tuple[object, ...]:
        variables = tuple((variable.name, variable.free_dim) for variable in self.problem.vars)
        return variables, batch_shape, exemplar.dtype, exemplar.device

    def _ensure_current_layout(self) -> None:
        batch_shape, exemplar = self._layout()
        if self._current_layout_key(batch_shape, exemplar) != self._layout_key:
            self.reset()

    def resume(self) -> None:
        """Resume terminal elements without clearing moments or scheduler state."""
        self._ensure_current_layout()
        assert self._status is not None
        self._status = torch.full_like(self._status, OptimizerStatus.RUNNING)

    def _candidate(self, base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.problem.retract(base, self._buffers)

    def _gradient_norm(self) -> torch.Tensor:
        norms: list[torch.Tensor] = []
        for name, buffer in self._buffers.items():
            if not buffer.numel():
                continue
            if buffer.grad is None:
                buffer.grad = self._zero_gradients[name]
            norms.append(buffer.grad.detach().abs().amax(dim=-1))
        if not norms:
            assert self._status is not None
            return torch.zeros_like(self._status, dtype=self._cost.dtype if self._cost is not None else torch.float32)
        return torch.stack(norms, dim=-1).amax(dim=-1)

    def _closure(
        self,
        base: dict[str, torch.Tensor],
        active: torch.Tensor,
    ) -> Callable[[], torch.Tensor]:
        assert self._optimizer is not None

        def evaluate() -> torch.Tensor:
            self._optimizer.zero_grad(set_to_none=True)
            cost = self.problem.objective(self._candidate(base))
            self._evaluated_cost = cost.detach()
            objective = (cost * active).sum()
            objective.backward()
            return objective

        return evaluate

    def _rebase(self, base: dict[str, torch.Tensor], active: torch.Tensor) -> None:
        tangent = {name: buffer.detach() * active.unsqueeze(-1) for name, buffer in self._buffers.items()}
        self.problem._set_trainable(self.problem.retract(base, tangent), detach=True)
        with torch.no_grad():
            for buffer in self._buffers.values():
                buffer.zero_()

    def _initial_info(self) -> OptimizerInfo:
        assert self._status is not None and self._iterations is not None and self._cost is not None
        return OptimizerInfo(self._status.detach(), self._iterations.detach(), self._cost.detach())

    def step(self) -> OptimizerInfo:
        self._ensure_current_layout()
        assert self._optimizer is not None
        assert self._status is not None and self._iterations is not None
        running = self._status == OptimizerStatus.RUNNING
        if not bool(running.any()):  # bench-ok: eager torch.optim boundary
            return self._initial_info()

        base = {variable.name: variable.tensor.detach() for variable in self.problem.vars}
        closure = self._closure(base, running)
        closure()
        self._cost = self._evaluated_cost
        converged = self._gradient_norm() <= self.tolerance
        self._status = torch.where(
            running & converged,
            torch.full_like(self._status, OptimizerStatus.CONVERGED),
            self._status,
        )
        active = self._status == OptimizerStatus.RUNNING
        if bool(active.any()):  # bench-ok: eager torch.optim boundary
            if isinstance(self._optimizer, torch.optim.LBFGS):
                self._optimizer.step(self._closure(base, active))
            else:
                self._optimizer.step()
            self._rebase(base, active)
            self._iterations = self._iterations + active.to(self._iterations.dtype)
            if self._scheduler is not None:
                self._scheduler.step()

        self._optimizer.zero_grad(set_to_none=True)
        return self._initial_info()

    def optimize(self, *, verbose: bool = False, differentiate: str | None = None) -> OptimizerInfo:
        if differentiate is not None:
            raise ValueError("TorchOptimizer does not support differentiation")
        self._ensure_current_layout()
        info = super().optimize(verbose=verbose)
        if bool((info.status == OptimizerStatus.RUNNING).any()):
            assert self._status is not None
            self._status = torch.where(
                self._status == OptimizerStatus.RUNNING,
                torch.full_like(self._status, OptimizerStatus.MAXITER),
                self._status,
            )
        with torch.no_grad():
            self._cost = self.problem.objective().detach()
        return self._initial_info()


__all__ = [
    "Optimizer",
    "OptimizerInfo",
    "OptimizerStatus",
    "TorchOptimizer",
]
