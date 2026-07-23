"""``better_robot`` — PyTorch-native robotics library.

The root public surface is intentionally small.
``tests/contract/test_public_api.py`` guards the required core without freezing
an exact symbol count. Other documented public symbols live under qualified
submodules; unlisted implementation details may be reshaped without a
deprecation. See ``docs/concepts/architecture.md §Public API contract``.

Layered DAG (arrows point from dependent to dependency)::

    tasks → optim → residuals → kinematics ↴
                                  │         dynamics ↴
                                  ▼                   ▼
                                 data_model ── spatial ── lie

    io → data_model           (io reads nothing from optim or tasks)
    viewer → tasks            (topmost; no-one imports from viewer)
"""

from __future__ import annotations

from . import exceptions as exceptions, io as io, spatial as spatial
from ._version import __version__ as __version__
from .data_model import Body, Data, Frame, Joint, Model, ModelStructure, ModelValues
from .dynamics import (
    aba,
    center_of_mass,
    compute_centroidal_map,
    crba,
    rnea,
)
from .io import ModelBuilder, load
from .kinematics import (
    compute_joint_jacobians,
    compute_joint_jacobians_time_variation,
    forward_kinematics,
    get_frame_jacobian,
    get_frame_jacobian_time_variation,
    get_joint_jacobian,
    get_joint_jacobian_time_variation,
    update_frame_placements,
)
from .lie.types import SE3
from .tasks import Trajectory, solve_contact_forces, solve_ik, solve_trajopt

__all__ = [
    # data_model (7)
    "Model",
    "ModelStructure",
    "ModelValues",
    "Data",
    "Frame",
    "Joint",
    "Body",
    # io (2)
    "load",
    "ModelBuilder",
    # lie (1)
    "SE3",
    # kinematics (8)
    "forward_kinematics",
    "update_frame_placements",
    "compute_joint_jacobians",
    "compute_joint_jacobians_time_variation",
    "get_joint_jacobian",
    "get_joint_jacobian_time_variation",
    "get_frame_jacobian",
    "get_frame_jacobian_time_variation",
    # dynamics (5)
    "rnea",
    "aba",
    "crba",
    "center_of_mass",
    "compute_centroidal_map",
    # tasks (4)
    "solve_ik",
    "solve_trajopt",
    "solve_contact_forces",
    "Trajectory",
]
