"""Robot cost terms for ``better_robot.optim``: plain functions with optional analytic Jacobians.

Each factory returns a ``TermFn`` to register on a ``better_robot.optim.Problem``. ``kinematic_state`` is the cached
forward-kinematics pass that ``frame_pose`` and ``frame_position`` read; ``joint_limits``, ``rest``,
``finite_difference`` and ``velocity_limit`` act on the configuration ``q``; ``sphere_clearance`` and
``self_clearance`` keep collision spheres apart. ``RobotManifold`` steps ``q`` with ``Model.integrate``.
``ik_problem`` and ``solve_ik`` are the inverse-kinematics recipe built from these terms, and ``random_start``
samples restarts.
"""

from __future__ import annotations

from .collision import self_clearance, sphere_clearance
from .ik import ik_problem, random_start, solve_ik
from .robot import (
    RobotManifold,
    finite_difference,
    frame_pose,
    frame_position,
    joint_limits,
    kinematic_state,
    rest,
    velocity_limit,
)

__all__ = [
    "RobotManifold",
    "finite_difference",
    "frame_pose",
    "frame_position",
    "ik_problem",
    "joint_limits",
    "kinematic_state",
    "random_start",
    "rest",
    "self_clearance",
    "solve_ik",
    "sphere_clearance",
    "velocity_limit",
]
