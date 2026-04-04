"""Gauss-Newton solver via pypose.optim.GaussNewton."""

from __future__ import annotations

import torch

from ._base import Problem, Solver


class GaussNewton(Solver):
    """GN solver wrapping pypose.optim.GaussNewton.

    Faster than LM on well-conditioned problems (no damping).
    """

    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run Gauss-Newton optimization.

        Args:
            problem: Problem instance.
            max_iter: Maximum iterations.

        Returns:
            Optimized variable tensor.
        """
        raise NotImplementedError
