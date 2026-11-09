"""``better_robot`` — PyTorch-native robotics library.

The root public surface is intentionally small.
``tests/contract/test_public_api.py`` guards the required core without freezing
an exact symbol count. Other documented public symbols live under qualified
submodules; unlisted implementation details may be reshaped without a
deprecation. See ``docs/concepts/architecture.md``, section "Public API is what ``__all__`` lists".

Each package imports only the packages of its row in the edge table (``ALLOWED`` in
``tests/contract/test_layer_dependencies.py``, ``docs/contracts/07-dependencies-layering.md``); ``optim`` imports
only torch and knows nothing about robots.
"""

from __future__ import annotations

from . import exceptions as exceptions, io as io, spatial as spatial, testing as testing
from ._version import __version__ as __version__
from .model import Model
from .io import ModelBuilder, load
from .dynamics import (
    center_of_mass,
    centroidal_map,
    centroidal_momentum,
    forward_dynamics,
    inverse_dynamics,
    mass_matrix,
)
from .kinematics import (
    KinematicState,
    forward_kinematics,
    frame_jacobian,
    frame_jacobian_time_variation,
    joint_jacobian,
    joint_jacobian_time_variation,
)

__all__ = [
    # model
    "Model",
    # io
    "load",
    "ModelBuilder",
    # kinematics
    "forward_kinematics",
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
    # repository checks shared with BetterHuman
    "testing",
]
