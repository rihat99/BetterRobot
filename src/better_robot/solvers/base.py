"""Solver abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .problem import Problem


class Solver(ABC):
    """Abstract base class for all optimization solvers."""

    @abstractmethod
    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run optimization and return the solution.

        Args:
            problem: Problem instance with variables and costs.
            max_iter: Maximum number of iterations.

        Returns:
            Optimized variable tensor, same shape as problem.variables.
        """
        raise NotImplementedError
