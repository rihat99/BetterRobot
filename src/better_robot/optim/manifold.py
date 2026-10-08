"""Manifolds: how a variable is stepped along a tangent vector.

A variable tensor has shape ``(B, *event)``. Its tangent is always flat, ``(B, n)``.
Second-order solvers step with ``Manifold.retract``; ``torch.optim`` steps the
ambient tensor and calls ``Manifold.project`` afterwards.
"""

from __future__ import annotations

from typing import Protocol

import torch


class Manifold(Protocol):
    """What the optimizer needs to know about a non-Euclidean variable."""

    def tangent_dim(self, x: torch.Tensor) -> int:
        """Tangent size ``n`` of one batch element of ``x: (B, *event)``."""
        ...

    def retract(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Step ``x: (B, *event)`` along ``delta: (B, n)``; ``retract(x, 0) == x``."""
        ...

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Map an ambient tensor back onto the manifold."""
        ...

    def tangent_bounds(
        self, x: torch.Tensor, *, lower: torch.Tensor, upper: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allowed step range ``(B, n), (B, n)`` at ``x`` for box bounds on the ambient tensor."""
        ...


class Euclidean:
    """Plain tensor: the tangent is the flattened event."""

    def tangent_dim(self, x: torch.Tensor) -> int:
        return x.shape[1:].numel()

    def retract(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return x + delta.reshape(x.shape)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def tangent_bounds(
        self, x: torch.Tensor, *, lower: torch.Tensor, upper: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (lower - x).expand_as(x).flatten(1), (upper - x).expand_as(x).flatten(1)


__all__ = ["Euclidean", "Manifold"]
