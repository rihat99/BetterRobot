"""L-BFGS solver wrapping torch.optim.LBFGS."""

from __future__ import annotations

import torch

from ._base import Problem, Solver


class LBFGSSolver(Solver):
    """Quasi-Newton solver using torch.optim.LBFGS.

    Best for smooth objectives, e.g. trajectory smoothing.
    """

    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        lr: float = 1.0,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run L-BFGS optimization.

        Args:
            problem: Problem instance.
            max_iter: Maximum iterations.
            lr: Step size.

        Returns:
            Optimized variable tensor.
        """
        raise NotImplementedError
