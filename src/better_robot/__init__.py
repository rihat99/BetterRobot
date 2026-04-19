"""``better_robot`` — PyTorch-native robotics library, phase-1 skeleton.

The public surface is intentionally small: **25 symbols** (the ceiling in
``docs/01_ARCHITECTURE.md``). Everything else is internal and may be
reshaped without a deprecation. See
``docs/01_ARCHITECTURE.md §Public API contract``.

Layered DAG (arrows point from dependent to dependency)::

    tasks → optim → residuals → kinematics ↴
                                  │         dynamics ↴
                                  ▼                   ▼
                                 data_model ── spatial ── lie ── backends

    io → data_model           (io reads nothing from optim or tasks)
    viewer → tasks            (topmost; no-one imports from viewer)
"""

from __future__ import annotations

from . import exceptions
from .costs import CostStack
from .data_model import Body, Data, Frame, Joint, Model
from .dynamics import (
    aba,
    center_of_mass,
    compute_centroidal_map,
    crba,
    rnea,
)
from .io import load
from .kinematics import (
    JacobianStrategy,
    compute_joint_jacobians,
    forward_kinematics,
    get_frame_jacobian,
    get_joint_jacobian,
    update_frame_placements,
)
from .optim import LeastSquaresProblem, solve
from .residuals import register_residual
from .tasks import Trajectory, retarget, solve_ik, solve_trajopt

# Exactly 25 symbols — the non-negotiable ceiling.
__all__ = [
    # data_model (5)
    "Model",
    "Data",
    "Frame",
    "Joint",
    "Body",
    # io (1)
    "load",
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
    # optim (2)
    "LeastSquaresProblem",
    "solve",
    # tasks (4)
    "solve_ik",
    "solve_trajopt",
    "retarget",
    "Trajectory",
]

assert len(__all__) == 25, f"better_robot.__all__ must have 25 symbols, not {len(__all__)}"
