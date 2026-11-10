"""``better_robot``: batched robot kinematics, dynamics and nonlinear least squares in PyTorch.

The root lists its twelve subpackages and the fourteen names a first script needs (decision record 0019); every
other public name is reached as ``better_robot.<package>.<name>``. Importing the root imports no optional package:
``Viewer`` loads viser when it starts a server. ``__version__`` is the installed distribution's version.
"""

from importlib import metadata as _metadata

from . import (
    collision,
    dynamics,
    exceptions,
    io,
    kinematics,
    lie,
    model,
    optim,
    spatial,
    terms,
    testing,
    viewer,
)
from .dynamics import center_of_mass, forward_dynamics, inverse_dynamics, mass_matrix
from .io import ModelBuilder, load
from .kinematics import KinematicState, forward_kinematics, frame_jacobian, joint_jacobian
from .model import JointFreeFlyer, Model
from .terms import solve_ik
from .viewer import Viewer

__version__ = _metadata.version("better-robot")

__all__ = [
    # subpackages
    "exceptions",
    "lie",
    "spatial",
    "model",
    "kinematics",
    "dynamics",
    "collision",
    "io",
    "optim",
    "terms",
    "viewer",
    "testing",
    # first-level names
    "load",
    "Model",
    "ModelBuilder",
    "JointFreeFlyer",
    "forward_kinematics",
    "KinematicState",
    "frame_jacobian",
    "joint_jacobian",
    "inverse_dynamics",
    "mass_matrix",
    "forward_dynamics",
    "center_of_mass",
    "solve_ik",
    "Viewer",
]
