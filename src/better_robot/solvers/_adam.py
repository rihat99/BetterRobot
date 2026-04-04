"""Adam solver wrapping torch.optim.Adam."""

from __future__ import annotations

import torch

from ._base import Problem, Solver


class AdamSolver(Solver):
    """Gradient descent solver using torch.optim.Adam.

    Best for noisy objectives or learning-integrated pipelines.
    """

    def solve(
        self,
        problem: Problem,
        max_iter: int = 500,
        lr: float = 1e-3,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run Adam optimization.

        Args:
            problem: Problem instance.
            max_iter: Number of gradient steps.
            lr: Learning rate.

        Returns:
            Optimized variable tensor.
        """
        raise NotImplementedError
