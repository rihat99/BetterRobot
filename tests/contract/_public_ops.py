"""Registry of public entry points for the dtype/device and no-in-place contract tests.

Each case builds its inputs for one dtype/device and returns ``(fn, args, kwargs)``. The model is a small
Panda-free two-link arm so the cases run in milliseconds on CPU.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import torch

from better_robot import dynamics, kinematics
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.lie import se3, so3
from better_robot.spatial import Force, Inertia, Motion

Case = tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]


def two_link_arm(dtype: torch.dtype, device: torch.device):
    """Two revolute joints with offset masses so ``M(q)`` is positive definite."""
    builder = ModelBuilder("contract_arm")
    inertia = 0.01 * torch.eye(3)
    builder.add_body("base", mass=0.5, inertia=inertia)
    builder.add_body("link1", mass=1.0, com=torch.tensor([0.1, 0.0, 0.0]), inertia=inertia)
    builder.add_body("link2", mass=0.7, com=torch.tensor([0.1, 0.0, 0.0]), inertia=inertia)
    origin = torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    builder.add_revolute_z("j1", parent="base", child="link1", origin=origin, lower=-math.pi, upper=math.pi)
    builder.add_revolute_y("j2", parent="link1", child="link2", origin=origin, lower=-math.pi, upper=math.pi)
    return build_model(builder.finalize()).to(device=device, dtype=dtype)


def tensors_of(obj: Any) -> list[torch.Tensor]:
    """Every tensor held directly by ``obj`` (a tensor, a dataclass, or a plain object)."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (Motion, Force, Inertia)):
        return [obj.data]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        values = [getattr(obj, field.name) for field in dataclasses.fields(obj)]
    elif hasattr(obj, "__dict__"):
        values = list(vars(obj).values())
    else:
        return []
    return [value for value in values if isinstance(value, torch.Tensor)]


def _kinematic_cases(dtype: torch.dtype, device: torch.device) -> dict[str, Case]:
    model = two_link_arm(dtype, device)
    q = torch.tensor([0.3, -0.4], dtype=dtype, device=device)
    v = torch.tensor([0.5, 0.2], dtype=dtype, device=device)
    a = torch.tensor([-0.1, 0.4], dtype=dtype, device=device)
    state = kinematics.forward_kinematics_state(model, q)
    frame = model.frame_names[-1]
    return {
        "forward_kinematics_state": (kinematics.forward_kinematics_state, (model, q), {}),
        "frame_jacobian": (kinematics.frame_jacobian, (model, state, frame), {}),
        "joint_jacobian": (kinematics.joint_jacobian, (model, state, 2), {}),
        "frame_jacobian_time_variation": (kinematics.frame_jacobian_time_variation, (model, state), {"v": v, "frame": frame}),
        "joint_jacobian_time_variation": (kinematics.joint_jacobian_time_variation, (model, state), {"v": v, "joint": 2}),
        "inverse_dynamics": (dynamics.inverse_dynamics, (model, state, v, a), {}),
        "mass_matrix": (dynamics.mass_matrix, (model, state), {}),
        "forward_dynamics": (dynamics.forward_dynamics, (model, state, v, a), {}),
        "centroidal_map": (dynamics.centroidal_map, (model, state), {}),
        "centroidal_momentum": (dynamics.centroidal_momentum, (model, state), {"v": v}),
        "center_of_mass": (dynamics.center_of_mass, (model, state), {}),
    }


def _lie_spatial_cases(dtype: torch.dtype, device: torch.device) -> dict[str, Case]:
    def tensor(*values: float) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    twist = tensor(0.1, -0.2, 0.3, 0.2, 0.1, -0.3)
    pose_a, pose_b = se3.exp(twist), se3.exp(-0.5 * twist)
    rot_a, rot_b = so3.exp(twist[3:]), so3.exp(-twist[3:])
    motion, other = Motion(twist), Motion(tensor(0.3, 0.1, 0.0, -0.1, 0.2, 0.4))
    force = Force(tensor(1.0, 0.0, -1.0, 0.1, 0.2, 0.3))
    inertia = Inertia(tensor(1.0, 0.1, 0.0, 0.0, 0.02, 0.0, 0.0, 0.03, 0.0, 0.04))
    return {
        "se3.exp": (se3.exp, (twist,), {}),
        "se3.log": (se3.log, (pose_a,), {}),
        "se3.compose": (se3.compose, (pose_a, pose_b), {}),
        "se3.inverse": (se3.inverse, (pose_a,), {}),
        "so3.exp": (so3.exp, (twist[3:],), {}),
        "so3.log": (so3.log, (rot_a,), {}),
        "so3.compose": (so3.compose, (rot_a, rot_b), {}),
        "so3.inverse": (so3.inverse, (rot_a,), {}),
        "Motion.cross_motion": (Motion.cross_motion, (motion, other), {}),
        "Motion.cross_force": (Motion.cross_force, (motion, force), {}),
        "Motion.se3_action": (Motion.se3_action, (motion, pose_a), {}),
        "Force.se3_action": (Force.se3_action, (force, pose_a), {}),
        "Inertia.apply": (Inertia.apply, (inertia, motion), {}),
        "Inertia.add": (Inertia.add, (inertia, inertia), {}),
    }


def build_case(name: str, dtype: torch.dtype, device: torch.device) -> Case:
    """The ``(fn, args, kwargs)`` triple of entry point ``name`` in ``dtype`` on ``device``."""
    return {**_kinematic_cases(dtype, device), **_lie_spatial_cases(dtype, device)}[name]


CASE_NAMES: tuple[str, ...] = tuple(
    {**_kinematic_cases(torch.float64, torch.device("cpu")), **_lie_spatial_cases(torch.float64, torch.device("cpu"))}
)
