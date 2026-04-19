"""Inverse dynamics (Featherstone's RNEA) and related helpers.

**Skeleton only** — all functions raise ``NotImplementedError`` in v1.
See ``docs/06_DYNAMICS.md §2`` for milestone D2.
"""

from __future__ import annotations

import torch

from ..data_model.data import Data
from ..data_model.model import Model


def rnea(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    *,
    fext: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inverse dynamics: ``τ = M(q) a + b(q, v) + g(q) - J^T fext``.

    Populates ``data.tau``, ``data.joint_pose_world``, ``data.v``,
    ``data.a`` along the way. Two-pass Featherstone algorithm (forward
    velocities/accelerations, backward forces/torques).

    TODO(milestone D2). See docs/06_DYNAMICS.md §2.
    """
    raise NotImplementedError("TODO(milestone D2) — see docs/06_DYNAMICS.md §2")


def bias_forces(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Non-linear effects ``C(q, v) v + g(q)``. Populates ``data.bias_forces``.

    (This function was previously called ``nle`` — see docs/13_NAMING.md.)

    TODO(milestone D2). See docs/06_DYNAMICS.md §2.
    """
    raise NotImplementedError("TODO(milestone D2) — see docs/06_DYNAMICS.md §2")


# Deprecated alias — remove in v1.1.
nle = bias_forces


def compute_generalized_gravity(
    model: Model,
    data: Data,
    q: torch.Tensor,
) -> torch.Tensor:
    """``g(q)``, the generalized gravity torque vector. Populates ``data.gravity_torque``.

    TODO(milestone D2). See docs/06_DYNAMICS.md §2.
    """
    raise NotImplementedError("TODO(milestone D2) — see docs/06_DYNAMICS.md §2")


def compute_coriolis_matrix(
    model: Model,
    data: Data,
    q: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """``C(q, v)``, the Coriolis matrix. Populates ``data.coriolis_matrix``.

    TODO(milestone D2). See docs/06_DYNAMICS.md §2.
    """
    raise NotImplementedError("TODO(milestone D2) — see docs/06_DYNAMICS.md §2")
