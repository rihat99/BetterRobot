"""Legacy flat-residual composition owned by :mod:`better_robot.optim`.

``CostStack`` remains the composer for direct-use
:class:`LeastSquaresProblem` compatibility callers and the input container
that ``solve_trajopt`` adapts into a named-block ``Problem``.
The old :mod:`better_robot.costs.stack` module is an identity-preserving
compatibility re-export; new code should import from :mod:`better_robot.optim`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..kinematics.jacobian import residual_jacobian
from ..kinematics.jacobian_strategy import JacobianStrategy
from ..residuals.base import Residual, ResidualState

CostKind = Literal["soft", "constraint_leq_zero"]


@dataclass
class CostItem:
    """One named slot in a :class:`CostStack`."""

    name: str
    residual: Residual
    weight: float = 1.0
    active: bool = True
    kind: CostKind = "soft"
    # populated by CostStack at finalise time
    slice: slice | None = None


class CostStack:
    """Named, weighted, individually activatable stack of residuals."""

    items: dict[str, CostItem]

    def __init__(self) -> None:
        self.items = {}

    # ────────────────────────── mutation ──────────────────────────

    def add(
        self,
        name: str,
        residual: Residual,
        *,
        weight: float = 1.0,
        kind: CostKind = "soft",
    ) -> None:
        """Add a residual to the stack under ``name``.

        See docs/concepts/residuals_and_costs.md §3.
        """
        if name in self.items:
            raise ValueError(f"cost item {name!r} already in stack")
        self.items[name] = CostItem(name=name, residual=residual, weight=weight, kind=kind)

    def remove(self, name: str) -> None:
        """Remove an item by name."""
        del self.items[name]

    def set_active(self, name: str, active: bool) -> None:
        """Toggle a cost item on/off without removing it."""
        self.items[name].active = active

    def set_weight(self, name: str, weight: float) -> None:
        """Update a cost item's scalar weight."""
        self.items[name].weight = weight

    # ────────────────────────── introspection ──────────────────────────

    def total_dim(self) -> int:
        """Return the concatenated residual dimension over all active items.

        See docs/concepts/residuals_and_costs.md §3.
        """
        return sum(item.residual.dim for item in self.items.values() if item.active)

    def slice_map(self) -> dict[str, slice]:
        """Return the ``{name: slice}`` map into the flat residual buffer.

        See docs/concepts/residuals_and_costs.md §3.
        """
        slices: dict[str, slice] = {}
        offset = 0
        for name, item in self.items.items():
            if item.active:
                d = item.residual.dim
                slices[name] = slice(offset, offset + d)
                offset += d
        return slices

    # ────────────────────────── evaluation ──────────────────────────

    def residual(self, state: ResidualState) -> torch.Tensor:
        """Flatten every active residual into a ``(B..., total_dim)`` buffer.

        See docs/concepts/residuals_and_costs.md §3.
        """
        parts: list[torch.Tensor] = []
        for item in self.items.values():
            if not item.active:
                continue
            r = item.residual(state)  # (B..., dim)
            parts.append(r * item.weight)
        if not parts:
            return torch.zeros(0, dtype=state.variables.dtype, device=state.variables.device)
        return torch.cat(parts, dim=-1)  # (B..., total_dim)

    def jacobian(
        self,
        state: ResidualState,
        *,
        strategy: JacobianStrategy = JacobianStrategy.AUTO,
    ) -> torch.Tensor:
        """Per-item Jacobian dispatch, stacked into ``(B..., total_dim, nv)``.

        See docs/concepts/residuals_and_costs.md §3.
        """
        parts: list[torch.Tensor] = []
        for item in self.items.values():
            if not item.active:
                continue
            jacobian = residual_jacobian(item.residual, state, strategy=strategy)
            parts.append(jacobian * item.weight)
        if not parts:
            nv = state.model.nv
            return torch.zeros(
                *state.variables.shape[:-1],
                0,
                nv,
                dtype=state.variables.dtype,
                device=state.variables.device,
            )
        return torch.cat(parts, dim=-2)

    def gradient(self, state: ResidualState) -> torch.Tensor:
        """Return the gradient of ``0.5 * ||stack.residual(state)||^2``.

        Residuals that implement ``apply_jac_transpose(state, r)`` contribute
        via that sparse path. Anything else falls back to a dense ``J.T @ r``.
        """
        gradient: torch.Tensor | None = None
        for item in self.items.values():
            if not item.active:
                continue
            residual = item.residual(state)
            if hasattr(item.residual, "apply_jac_transpose"):
                contribution = item.residual.apply_jac_transpose(state, residual)
            else:
                jacobian = residual_jacobian(item.residual, state)
                contribution = jacobian.mT @ residual
            contribution = (item.weight**2) * contribution
            gradient = contribution if gradient is None else gradient + contribution
        if gradient is None:
            return torch.zeros_like(state.variables).reshape(-1)
        return gradient


__all__ = ["CostKind", "CostItem", "CostStack"]
