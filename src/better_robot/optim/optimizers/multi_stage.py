"""``MultiStageOptimizer`` — chain N optimisers with per-stage cost overrides.

Pattern adopted from cuRobo's high-DoF humanoid IK: an LM stage gets into
the basin of attraction quickly, then a quasi-Newton stage finishes the
residual to tight tolerance. Generalises ``LMThenLBFGS`` to N stages and
exposes per-stage ``weight_overrides`` and ``disabled_items``.

Snapshots the cost-stack state on entry to every stage and **restores it
in a ``try / finally``** so a stage that raises does not leak active /
weight changes onto subsequent calls (or other code holding the same
``CostStack``).

See ``docs/concepts/solver_stack.md §5`` and
``docs/concepts/tasks.md``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..problem import LeastSquaresProblem
from ..state import SolverState


@dataclass
class OptimizerStage:
    """One stage in a :class:`MultiStageOptimizer` schedule."""

    optimizer: Any
    max_iter: int = 50
    disabled_items: tuple[str, ...] = ()
    weight_overrides: dict[str, float] = field(default_factory=dict)
    linear_solver: Any | None = None
    kernel: Any | None = None
    strategy: Any | None = None


class MultiStageOptimizer:
    """Run a sequence of optimisers; each may override the cost stack."""

    def __init__(self, *, stages: Sequence[OptimizerStage]) -> None:
        if not stages:
            raise ValueError("MultiStageOptimizer requires at least one stage")
        self.stages = tuple(stages)

    def minimize(
        self,
        problem: LeastSquaresProblem,
        *,
        max_iter: int | None = None,
        linear_solver=None,
        kernel=None,
        strategy=None,
        scheduler=None,
    ) -> SolverState:
        """Run every stage in order.

        ``max_iter`` if given splits evenly across stages; otherwise each
        stage uses its own ``max_iter``. Top-level ``linear_solver`` /
        ``kernel`` / ``strategy`` apply to stages that did not set their
        own.
        """
        per_stage_iter: list[int]
        if max_iter is not None:
            per_stage = max(1, max_iter // len(self.stages))
            per_stage_iter = [per_stage] * len(self.stages)
            per_stage_iter[-1] = max_iter - per_stage * (len(self.stages) - 1)
        else:
            per_stage_iter = [s.max_iter for s in self.stages]

        cumulative_history: list = []
        cumulative_iters = 0
        last_state: SolverState | None = None
        x = problem.x0

        for stage, n_iter in zip(self.stages, per_stage_iter):
            with _cost_stack_snapshot(
                problem.cost_stack,
                disabled=stage.disabled_items,
                weight_overrides=stage.weight_overrides,
            ):
                problem_stage = dataclasses.replace(problem, x0=x.detach())
                # First-order optimisers (Adam / L-BFGS) ignore normal-equation
                # kwargs and warn when they receive them; suppress the forward.
                ignores = getattr(stage.optimizer, "_ignores_normal_eqn_kwargs", False)
                stage_linear = stage.linear_solver if stage.linear_solver is not None else (None if ignores else linear_solver)
                stage_kernel = stage.kernel if stage.kernel is not None else (None if ignores else kernel)
                stage_strat = stage.strategy if stage.strategy is not None else (None if ignores else strategy)
                state = stage.optimizer.minimize(
                    problem_stage,
                    max_iter=n_iter,
                    linear_solver=stage_linear,
                    kernel=stage_kernel,
                    strategy=stage_strat,
                    scheduler=scheduler,
                )
            x = state.x
            cumulative_history.extend(state.history)
            cumulative_iters += state.iters
            last_state = state
            if state.status == "converged":
                break

        assert last_state is not None
        last_state.history = cumulative_history
        last_state.iters = cumulative_iters
        return last_state


class _cost_stack_snapshot:
    """Context manager: snapshot cost-stack ``active``/``weight``; restore on exit."""

    def __init__(
        self,
        cost_stack,
        *,
        disabled: Iterable[str] = (),
        weight_overrides: dict[str, float] | None = None,
    ) -> None:
        self.cost_stack = cost_stack
        self.disabled = tuple(disabled)
        self.weight_overrides = dict(weight_overrides or {})
        self._saved_active: dict[str, bool] = {}
        self._saved_weight: dict[str, float] = {}

    def __enter__(self) -> "_cost_stack_snapshot":
        for name in self.disabled:
            if name in self.cost_stack.items:
                self._saved_active[name] = self.cost_stack.items[name].active
                self.cost_stack.set_active(name, False)
        for name, w in self.weight_overrides.items():
            if name in self.cost_stack.items:
                self._saved_weight[name] = self.cost_stack.items[name].weight
                self.cost_stack.set_weight(name, w)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Restore in reverse order; survives even if the inner call raised.
        for name, w in self._saved_weight.items():
            self.cost_stack.set_weight(name, w)
        for name, active in self._saved_active.items():
            self.cost_stack.set_active(name, active)
