"""Centre of mass and centroidal momentum of a `KinematicState`."""

from __future__ import annotations

import math

import torch

from ..kinematics import KinematicState
from ..kinematics.forward_kinematics import check_state
from ..kinematics.configuration import reduced_columns
from ..lie._checks import check_tensor
from ..model.model import Model
from ._world import centroidal_map_full, world_kinematics


def center_of_mass(model: Model, state: KinematicState) -> torch.Tensor:
    """Whole-body centre of mass in the world frame, ``(B..., 3)``."""
    check_state(model, state)
    return _center_of_mass(model, state)


def _center_of_mass(model: Model, state: KinematicState) -> torch.Tensor:
    """`center_of_mass` of a checked state, ``(B..., 3)``."""
    inertias = model.body_inertias  # (..., njoints, 10): mass, com, rotational inertia
    masses, com_local = inertias[..., 0], inertias[..., 1:4]
    com_world = (state.joint_rotations @ com_local.unsqueeze(-1)).squeeze(-1) + state.joint_translations
    return (masses.unsqueeze(-1) * com_world).sum(-2) / masses.sum(-1).unsqueeze(-1)


def centroidal_map(model: Model, state: KinematicState) -> torch.Tensor:
    """Centroidal momentum matrix ``A_g(q)``, ``(B..., 6, nv)``: ``h_g = A_g v`` in world axes at the centre of mass.

    Rows are ``[linear momentum, angular momentum]``; the motor armature does not enter, as in Pinocchio.
    """
    check_state(model, state)
    return _centroidal_map(model, state)


def _centroidal_map(model: Model, state: KinematicState) -> torch.Tensor:
    """`centroidal_map` of a checked state, ``(B..., 6, nv)``."""
    kin = world_kinematics(model, state)
    com = _center_of_mass(model, state).reshape(math.prod(kin.batch), 3)
    return reduced_columns(model, centroidal_map_full(kin, com)).view(*kin.batch, 6, model.nv)


def centroidal_momentum(model: Model, state: KinematicState, *, v: torch.Tensor) -> torch.Tensor:
    """Centroidal momentum ``h_g = A_g(q) v``, ``(B..., 6)``, for ``v: (B..., nv)``; rows as `centroidal_map`.

    Raises as `inverse_dynamics` for a ``v`` that is not a tensor of the model's dtype and device ending in ``nv``.
    """
    check_state(model, state)
    check_tensor("v", v, shape=(model.nv,), dtype=model.dtype, device=model.device)
    return (_centroidal_map(model, state) @ v.unsqueeze(-1)).squeeze(-1)


__all__ = ["center_of_mass", "centroidal_map", "centroidal_momentum"]
