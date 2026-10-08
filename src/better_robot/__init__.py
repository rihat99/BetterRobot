"""``better_robot`` — PyTorch-native robotics library.

The root public surface is intentionally small.
``tests/contract/test_public_api.py`` guards the required core without freezing
an exact symbol count. Other documented public symbols live under qualified
submodules; unlisted implementation details may be reshaped without a
deprecation. See ``docs/concepts/architecture.md §Public API contract``.

Layered DAG (arrows point from dependent to dependency)::

    terms → kinematics, dynamics, collision → data_model → spatial → lie
      │
      ▼
    optim                     (imports only torch; knows nothing about robots)

    io → data_model           (io reads nothing from optim or terms)
    viewer → terms            (topmost; no-one imports from viewer)
"""

from __future__ import annotations

from . import exceptions as exceptions, io as io, spatial as spatial
from ._version import __version__ as __version__
from .data_model import Body, Frame, Joint, Model, ModelStructure, ModelValues, Trajectory
from .io import ModelBuilder, load
from .dynamics.state import (
    center_of_mass,
    centroidal_map,
    centroidal_momentum,
    forward_dynamics,
    inverse_dynamics,
    mass_matrix,
)
from .kinematics.state import (
    KinematicState,
    forward_kinematics_state,
    frame_jacobian,
    frame_jacobian_time_variation,
    joint_jacobian,
    joint_jacobian_time_variation,
)
from .lie.types import SE3

__all__ = [
    # data_model
    "Model",
    "ModelStructure",
    "ModelValues",
    "Frame",
    "Joint",
    "Body",
    "Trajectory",
    # io
    "load",
    "ModelBuilder",
    # lie
    "SE3",
    # kinematics
    "forward_kinematics_state",
    "KinematicState",
    "frame_jacobian",
    "frame_jacobian_time_variation",
    "joint_jacobian",
    "joint_jacobian_time_variation",
    # dynamics
    "center_of_mass",
    "inverse_dynamics",
    "mass_matrix",
    "forward_dynamics",
    "centroidal_map",
    "centroidal_momentum",
]
