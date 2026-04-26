"""``better_robot`` — PyTorch-native robotics library.

The public surface is intentionally small: **26 symbols** — the frozen
``EXPECTED`` set in ``tests/contract/test_public_api.py``. Everything else is
internal and may be reshaped without a deprecation. See
``docs/design/01_ARCHITECTURE.md §Public API contract``.

Layered DAG (arrows point from dependent to dependency)::

    tasks → optim → residuals → kinematics ↴
                                  │         dynamics ↴
                                  ▼                   ▼
                                 data_model ── spatial ── lie ── backends

    io → data_model           (io reads nothing from optim or tasks)
    viewer → tasks            (topmost; no-one imports from viewer)
"""

from __future__ import annotations

from . import exceptions, io
from ._version import __version__
from .costs import CostStack
from .data_model import Body, Data, Frame, Joint, Model
from .dynamics import (
    aba,
    center_of_mass,
    compute_centroidal_map,
    crba,
    rnea,
)
from .io import ModelBuilder, load
from .kinematics import (
    JacobianStrategy,
    compute_joint_jacobians,
    forward_kinematics,
    get_frame_jacobian,
    get_joint_jacobian,
    update_frame_placements,
)
from .lie.types import SE3
from .optim import LeastSquaresProblem
from .residuals import register_residual
from .tasks import Trajectory, retarget, solve_ik, solve_trajopt

__all__ = [
    # data_model (5)
    "Model",
    "Data",
    "Frame",
    "Joint",
    "Body",
    # io (2)
    "load",
    "ModelBuilder",
    # lie (1)
    "SE3",
    # kinematics (6)
    "forward_kinematics",
    "update_frame_placements",
    "compute_joint_jacobians",
    "get_joint_jacobian",
    "get_frame_jacobian",
    "JacobianStrategy",
    # dynamics (5)
    "rnea",
    "aba",
    "crba",
    "center_of_mass",
    "compute_centroidal_map",
    # residuals (1)
    "register_residual",
    # costs (1)
    "CostStack",
    # optim (1)
    "LeastSquaresProblem",
    # tasks (4)
    "solve_ik",
    "solve_trajopt",
    "retarget",
    "Trajectory",
]
