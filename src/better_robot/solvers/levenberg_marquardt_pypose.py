"""PyPose-based Levenberg-Marquardt solver (kept for benchmarking)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pypose.optim as ppo
import pypose.optim.strategy as ppo_strategy

from .problem import Problem
from .base import Solver
from .registry import SOLVERS


class _ProblemModule(nn.Module):
    """Wraps a Problem so PyPose's LM optimizer can call it."""

    def __init__(self, problem: Problem) -> None:
        super().__init__()
        self.x = nn.Parameter(problem.variables.clone().float())
        self._problem = problem

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        return self._problem.total_residual(self.x)


@SOLVERS.register("lm_pypose")
class PyposeLevenbergMarquardt(Solver):
    """LM solver using pypose.optim.LevenbergMarquardt.

    Always uses PyPose autograd for Jacobian. Ignores problem.jacobian_fn.
    Kept for benchmarking against the custom LM solver.
    """

    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        damping: float = 1e-4,
        **kwargs,
    ) -> torch.Tensor:
        module = _ProblemModule(problem)
        strategy = ppo_strategy.Adaptive(damping=damping)
        optimizer = ppo.LevenbergMarquardt(module, strategy=strategy, vectorize=True)

        lo = problem.lower_bounds
        hi = problem.upper_bounds
        dummy = torch.zeros(1, device=module.x.device)

        for _ in range(max_iter):
            optimizer.step(input=dummy)
            if lo is not None and hi is not None:
                with torch.no_grad():
                    module.x.data.clamp_(
                        lo.to(dtype=module.x.dtype, device=module.x.device),
                        hi.to(dtype=module.x.dtype, device=module.x.device),
                    )

        return module.x.detach()
