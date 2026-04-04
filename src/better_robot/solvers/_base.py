"""Base abstractions: CostTerm, Problem, Solver ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch


@dataclass
class CostTerm:
    """A single differentiable cost/constraint term."""

    residual_fn: Callable[[torch.Tensor], torch.Tensor]
    """Function mapping joint config to residual vector."""

    weight: float = 1.0
    """Scalar weight applied to the residual."""

    kind: Literal["soft", "constraint_leq_zero"] = "soft"
    """'soft': minimized via least squares.
    'constraint_leq_zero': enforced via augmented Lagrangian (residual <= 0)."""


@dataclass
class Problem:
    """Optimization problem: variables + list of cost terms."""

    variables: torch.Tensor
    """Initial values for the optimization variable (joint config or trajectory)."""

    costs: list[CostTerm] = field(default_factory=list)
    """List of cost/constraint terms."""

    lower_bounds: torch.Tensor | None = None
    """Optional lower bounds for variables (same shape as variables)."""

    upper_bounds: torch.Tensor | None = None
    """Optional upper bounds for variables (same shape as variables)."""

    jacobian_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    """Optional analytical Jacobian function.

    If provided, our LM calls jacobian_fn(x) -> (m, n) Tensor instead of
    computing J via torch.func.jacrev. The PyPose LM ignores this field.
    """

    def total_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all weighted soft residuals into a single vector.

        Args:
            x: Current variable values.

        Returns:
            1D residual vector.
        """
        parts = []
        for ct in self.costs:
            if ct.kind == "soft":
                r = ct.residual_fn(x)
                parts.append(r * ct.weight)
        if not parts:
            return torch.zeros(0, dtype=x.dtype, device=x.device)
        return torch.cat(parts, dim=0)

    def constraint_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all constraint_leq_zero residuals into a single vector.

        Used by solvers to evaluate constraint violations separately from
        the soft cost (e.g. for augmented Lagrangian enforcement).

        Args:
            x: Current variable values.

        Returns:
            1D residual vector of all constraint terms (should be <= 0 when satisfied).
            Returns empty tensor if no constraint terms exist.
        """
        parts = []
        for ct in self.costs:
            if ct.kind == "constraint_leq_zero":
                r = ct.residual_fn(x)
                parts.append(r * ct.weight)
        if not parts:
            return torch.zeros(0, dtype=x.dtype, device=x.device)
        return torch.cat(parts, dim=0)


class Solver(ABC):
    """Abstract base class for all solvers."""

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
            **kwargs: Solver-specific hyperparameters.

        Returns:
            Optimized variable tensor, same shape as problem.variables.
        """
        raise NotImplementedError
