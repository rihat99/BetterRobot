"""Problem: optimization problem specification (variables + costs + bounds)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from ..costs.cost_term import CostTerm


@dataclass
class Problem:
    """Optimization problem: variables + list of cost terms + optional bounds."""

    variables: torch.Tensor
    """Initial values for the optimization variable."""

    costs: list[CostTerm] = field(default_factory=list)
    """List of cost/constraint terms."""

    lower_bounds: torch.Tensor | None = None
    """Optional lower bounds (same shape as variables)."""

    upper_bounds: torch.Tensor | None = None
    """Optional upper bounds (same shape as variables)."""

    jacobian_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    """Optional analytical Jacobian. If provided, LM calls this instead of jacrev."""

    def total_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all weighted soft residuals into a single vector."""
        parts = []
        for ct in self.costs:
            if ct.kind == "soft":
                r = ct.residual_fn(x)
                parts.append(r * ct.weight)
        if not parts:
            return torch.zeros(0, dtype=x.dtype, device=x.device)
        return torch.cat(parts, dim=0)

    def constraint_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all constraint_leq_zero residuals into a single vector."""
        parts = []
        for ct in self.costs:
            if ct.kind == "constraint_leq_zero":
                r = ct.residual_fn(x)
                parts.append(r * ct.weight)
        if not parts:
            return torch.zeros(0, dtype=x.dtype, device=x.device)
        return torch.cat(parts, dim=0)
