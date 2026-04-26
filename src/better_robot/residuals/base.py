"""``Residual`` protocol and ``ResidualState`` struct.

Every residual is a **callable object** — not a plain function — so it can
optionally own an analytic ``.jacobian()``. Fallback to autodiff happens
one layer up in ``kinematics.jacobian.residual_jacobian``.

Trajectory-scale problems can also implement
``apply_jac_transpose(state, vec) -> Tensor`` to compute ``J^T @ vec`` in
the matrix-free path used by ``LeastSquaresProblem.gradient``. The
default implementation builds the dense Jacobian and multiplies — sparse
residuals (banded smoothness, sparse collisions) override this.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §2`` and
``docs/design/05_KINEMATICS.md §3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
        Per-query ``Data`` whose ``joint_pose_world`` (and
        ``frame_pose_world`` for frame-based residuals) are populated.
    variables : torch.Tensor
        Flat optimisation variable tensor ``(B..., nx)``.
    """

    model: Model
    data: Data
    variables: torch.Tensor


@runtime_checkable
class Residual(Protocol):
    """Protocol every residual class implements.

    Marked ``@runtime_checkable`` so the extension-seam docs can advertise
    ``isinstance(obj, Residual)`` as a valid contract check (docs/conventions/15_EXTENSION.md §1).
    """

    name: str
    dim: int

    def __call__(self, state: ResidualState) -> torch.Tensor:
        """Return the residual vector of shape ``(B..., dim)``."""
        ...

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """Return an analytic Jacobian of shape ``(B..., dim, nx)``, or
        ``None`` to fall back to autodiff."""
        ...


def default_apply_jac_transpose(
    residual: Residual, state: ResidualState, vec: torch.Tensor
) -> torch.Tensor:
    """Default ``apply_jac_transpose`` — materialise ``J`` and multiply.

    Concrete residuals override the bound method when they have block
    structure (banded smoothness, sparse collisions) that can be exploited
    without forming the dense Jacobian.

    Parameters
    ----------
    residual
        Any residual implementing the :class:`Residual` protocol.
    state
        The current :class:`ResidualState`.
    vec
        Tensor with shape ``(..., dim)`` — the same shape the residual
        returns.

    Returns
    -------
    torch.Tensor
        ``J^T @ vec`` with shape ``(..., nv)``.
    """
    from ..kinematics.jacobian import residual_jacobian

    J = residual_jacobian(residual, state)  # (..., dim, nv)
    return J.mT @ vec
