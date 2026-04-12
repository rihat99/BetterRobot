"""``Residual`` protocol and ``ResidualState`` struct.

Every residual is a **callable object** — not a plain function — so it can
optionally own an analytic ``.jacobian()``. Fallback to autodiff happens
one layer up in ``kinematics.jacobian.residual_jacobian``.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §2`` and
``docs/05_KINEMATICS.md §3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..data_model.data import Data
from ..data_model.model import Model


@dataclass
class ResidualState:
    """Thin struct passed to every residual.

    Attributes
    ----------
    model : Model
        Immutable Model.
    data : Data
        Per-query Data whose ``oMi``/``oMf`` are populated (for residuals
        that reach into kinematics).
    variables : torch.Tensor
        Flat optimisation variable tensor ``(B..., nx)``.
    """

    model: Model
    data: Data
    variables: torch.Tensor


class Residual(Protocol):
    """Protocol every residual class implements."""

    name: str
    dim: int

    def __call__(self, state: ResidualState) -> torch.Tensor:
        """Return the residual vector of shape ``(B..., dim)``."""
        ...

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """Return an analytic Jacobian of shape ``(B..., dim, nx)``, or
        ``None`` to fall back to autodiff."""
        ...
