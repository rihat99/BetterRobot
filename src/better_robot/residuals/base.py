"""``Residual`` protocol and ``ResidualState`` struct.

Every residual is a **callable object** — not a plain function — so it can
optionally own an analytic ``.jacobian()``. The current fallback is central
finite differences in ``kinematics.jacobian.residual_jacobian``; real
``torch.func`` fallback is scheduled for M2.

Legacy trajectory residuals can also implement
``apply_jac_transpose(state, vec) -> Tensor`` to compute ``J^T @ vec`` through
``LeastSquaresProblem.gradient``. No production solver or task calls that
gradient path; it is retained only for the matrix-free contract tests. The
default implementation builds the dense Jacobian and multiplies.

See ``docs/concepts/residuals_and_costs.md §2`` and
``docs/concepts/kinematics.md §3``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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


def _as_residual_state(
    value: ResidualState | Mapping[str, Any],
    *,
    model: Model | None,
) -> ResidualState:
    """Normalize legacy state and named-block context inputs.

    The built-in kinematic residuals keep their ``ResidualState`` call shape
    for the legacy trajectory stack while also implementing the structural
    named-block protocol. New block problems provide ``q`` and the lazy
    ``RobotStateProvider`` output ``data`` through a read-only mapping.
    """
    if isinstance(value, ResidualState):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("kinematic residual input must be ResidualState or a named-block context")
    if model is None:
        raise TypeError("this residual needs model=... when used with a named-block Problem")
    try:
        q = value["q"]
        data = value["data"]
    except KeyError as exc:
        raise KeyError("named-block kinematic residuals require reads=('q', 'data')") from exc
    if not isinstance(q, torch.Tensor):
        raise TypeError("named-block context entry 'q' must be a torch.Tensor")
    return ResidualState(model=model, data=data, variables=q)


def _residual_model_q(
    value: ResidualState | Mapping[str, Any],
    *,
    model: Model,
) -> tuple[Model, torch.Tensor]:
    """Read only model/configuration for residuals that do not require FK."""
    if isinstance(value, ResidualState):
        return value.model, value.variables
    if not isinstance(value, Mapping):
        raise TypeError("kinematic residual input must be ResidualState or a named-block context")
    q = value["q"]
    if not isinstance(q, torch.Tensor):
        raise TypeError("named-block context entry 'q' must be a torch.Tensor")
    return model, q


@runtime_checkable
class Residual(Protocol):
    """Protocol every residual class implements.

    Marked ``@runtime_checkable`` so the extension-seam docs can advertise
    ``isinstance(obj, Residual)`` as a valid contract check (docs/conventions/extension.md §1).
    """

    name: str
    dim: int

    def __call__(self, state: ResidualState) -> torch.Tensor:
        """Return the residual vector of shape ``(B..., dim)``."""
        ...

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """Return an analytic Jacobian of shape ``(B..., dim, nx)``, or
        ``None`` to fall back to central finite differences."""
        ...


def default_apply_jac_transpose(residual: Residual, state: ResidualState, vec: torch.Tensor) -> torch.Tensor:
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
    # Keep this lazy: kinematics.jacobian's legacy finite-difference helper
    # imports ResidualState at call time, so a top-level edge would make the
    # compatibility protocol's import cycle eager.
    from ..kinematics.jacobian import residual_jacobian  # noqa: PLC0415

    J = residual_jacobian(residual, state)  # (..., dim, nv)
    return J.mT @ vec
