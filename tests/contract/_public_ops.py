"""Registry of every public kinematics, dynamics, Lie and spatial function for the contract tests.

`PUBLIC_FUNCTIONS` is read from the packages' ``__all__``; each function has a case here or a reasoned entry in
`NOT_DRIVEN`, and ``test_dtype_device.py`` checks that the two cover it exactly, so the registry cannot go stale.
A case builds its inputs for one dtype, device and ``variant`` and returns ``(fn, args, kwargs)``: variants are
distinct valid inputs, so a batch stacked from them has distinct elements. The model is a small two-link arm, so
the cases run in milliseconds on CPU.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch

from better_robot import dynamics, kinematics, lie, spatial
from better_robot.io import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.lie import se3, so3, tangents

Case = tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]


def two_link_arm(dtype: torch.dtype, device: torch.device):
    """Two revolute joints with offset masses so ``M(q)`` is positive definite."""
    builder = ModelBuilder("contract_arm")
    inertia = 0.01 * torch.eye(3)
    builder.add_body("base", mass=0.5, inertia=inertia)
    builder.add_body("link1", mass=1.0, com=torch.tensor([0.1, 0.0, 0.0]), inertia=inertia)
    builder.add_body("link2", mass=0.7, com=torch.tensor([0.1, 0.0, 0.0]), inertia=inertia)
    origin = torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    builder.add_revolute(
        "j1", axis=(0.0, 0.0, 1.0), parent="base", child="link1", origin=origin, lower=-math.pi, upper=math.pi
    )
    builder.add_revolute(
        "j2", axis=(0.0, 1.0, 0.0), parent="link1", child="link2", origin=origin, lower=-math.pi, upper=math.pi
    )
    return build_model(builder.finalize()).to(device=device, dtype=dtype)


def tensors_of(obj: Any) -> list[torch.Tensor]:
    """Every tensor held directly by ``obj`` (a tensor, a tuple of them, a dataclass, or a plain object)."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, tuple):
        return [value for value in obj if isinstance(value, torch.Tensor)]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        values = [getattr(obj, field.name) for field in dataclasses.fields(obj)]
    elif hasattr(obj, "__dict__"):
        values = list(vars(obj).values())
    else:
        return []
    return [value for value in values if isinstance(value, torch.Tensor)]


def _kinematic_cases(dtype: torch.dtype, device: torch.device, scale: float) -> dict[str, Case]:
    model = two_link_arm(dtype, device)

    def tensor(*values: float) -> torch.Tensor:
        return scale * torch.tensor(values, dtype=dtype, device=device)

    q, v, a = tensor(0.3, -0.4), tensor(0.5, 0.2), tensor(-0.1, 0.4)
    state = kinematics.forward_kinematics(model, q)
    frame = model.frame_names[-1]
    return {
        "forward_kinematics": (kinematics.forward_kinematics, (model, q), {}),
        "frame_jacobian": (kinematics.frame_jacobian, (model, state), {"frame": frame}),
        "joint_jacobian": (kinematics.joint_jacobian, (model, state), {"joint": 2}),
        "point_jacobian": (
            kinematics.point_jacobian,
            (model, state),
            {"frames": [frame], "points": tensor(0.1, 0, 0.2)[None]},
        ),
        "frame_jacobian_time_variation": (
            kinematics.frame_jacobian_time_variation,
            (model, state),
            {"v": v, "frame": frame},
        ),
        "joint_jacobian_time_variation": (
            kinematics.joint_jacobian_time_variation,
            (model, state),
            {"v": v, "joint": 2},
        ),
        "inverse_dynamics": (dynamics.inverse_dynamics, (model, state), {"v": v, "a": a}),
        "mass_matrix": (dynamics.mass_matrix, (model, state), {}),
        "forward_dynamics": (dynamics.forward_dynamics, (model, state), {"v": v, "tau": a}),
        "centroidal_map": (dynamics.centroidal_map, (model, state), {}),
        "centroidal_momentum": (dynamics.centroidal_momentum, (model, state), {"v": v}),
        "center_of_mass": (dynamics.center_of_mass, (model, state), {}),
        "integrate": (kinematics.integrate, (model, q, v), {}),
        "difference": (kinematics.difference, (model, q, a), {}),
        "full_configuration": (kinematics.full_configuration, (model, q), {}),
    }


def _lie_spatial_cases(dtype: torch.dtype, device: torch.device, scale: float) -> dict[str, Case]:
    def tensor(*values: float) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    twist = scale * tensor(0.1, -0.2, 0.3, 0.2, 0.1, -0.3)
    pose_a, pose_b = se3.exp(twist), se3.exp(-0.5 * twist)
    rot_a, rot_b = so3.exp(twist[3:]), so3.exp(-twist[3:])
    other = scale * tensor(0.3, 0.1, 0.0, -0.1, 0.2, 0.4)
    wrench = scale * tensor(1.0, 0.0, -1.0, 0.1, 0.2, 0.3)
    inertia = scale * tensor(1.0, 0.1, 0.0, 0.0, 0.02, 0.0, 0.0, 0.03, 0.0, 0.04)
    axis = tensor(1.0, 2.0, 2.0) / 3.0
    angle, fraction, mass = (torch.tensor(scale * value, dtype=dtype, device=device) for value in (0.7, 0.3, 2.0))
    point, sizes = scale * tensor(0.1, -0.2, 0.3), scale * tensor(0.3, 0.2, 0.1)
    source = scale * tensor(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0).reshape(5, 3)
    cube = scale * torch.cartesian_prod(*3 * [tensor(-0.5, 0.5)])  # vertex 4 x + 2 y + z
    cube_faces = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4]]
    cube_faces += [[2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]
    faces = torch.tensor(cube_faces, device=device)
    return {
        "lie.umeyama": (lie.umeyama, (source, se3.act(pose_a, source)), {}),
        "se3.act": (se3.act, (pose_a, point), {}),
        "se3.adjoint": (se3.adjoint, (pose_a,), {}),
        "se3.adjoint_inverse": (se3.adjoint_inverse, (pose_a,), {}),
        "se3.compose": (se3.compose, (pose_a, pose_b), {}),
        "se3.exp": (se3.exp, (twist,), {}),
        "se3.from_axis_angle": (se3.from_axis_angle, (axis, angle), {}),
        "se3.from_matrix": (se3.from_matrix, (se3.to_matrix(pose_a),), {}),
        "se3.from_translation": (se3.from_translation, (point,), {}),
        "se3.inverse": (se3.inverse, (pose_a,), {}),
        "se3.log": (se3.log, (pose_a,), {}),
        "se3.normalize": (se3.normalize, (pose_a,), {}),
        "se3.sclerp": (se3.sclerp, (pose_a, pose_b, fraction), {}),
        "se3.to_matrix": (se3.to_matrix, (pose_a,), {}),
        "so3.act": (so3.act, (rot_a, point), {}),
        "so3.compose": (so3.compose, (rot_a, rot_b), {}),
        "so3.exp": (so3.exp, (twist[3:],), {}),
        "so3.from_axis_angle": (so3.from_axis_angle, (axis, angle), {}),
        "so3.from_euler": (so3.from_euler, (point,), {}),
        "so3.from_matrix": (so3.from_matrix, (so3.to_matrix(rot_a),), {}),
        "so3.inverse": (so3.inverse, (rot_a,), {}),
        "so3.log": (so3.log, (rot_a,), {}),
        "so3.normalize": (so3.normalize, (rot_a,), {}),
        "so3.slerp": (so3.slerp, (rot_a, rot_b, fraction), {}),
        "so3.to_euler": (so3.to_euler, (rot_a,), {}),
        "so3.to_matrix": (so3.to_matrix, (rot_a,), {}),
        "tangents.hat_se3": (tangents.hat_se3, (twist,), {}),
        "tangents.hat_so3": (tangents.hat_so3, (twist[3:],), {}),
        "tangents.left_jacobian_inverse_se3": (tangents.left_jacobian_inverse_se3, (twist,), {}),
        "tangents.left_jacobian_inverse_so3": (tangents.left_jacobian_inverse_so3, (twist[3:],), {}),
        "tangents.left_jacobian_se3": (tangents.left_jacobian_se3, (twist,), {}),
        "tangents.left_jacobian_so3": (tangents.left_jacobian_so3, (twist[3:],), {}),
        "tangents.right_jacobian_inverse_se3": (tangents.right_jacobian_inverse_se3, (twist,), {}),
        "tangents.right_jacobian_inverse_so3": (tangents.right_jacobian_inverse_so3, (twist[3:],), {}),
        "tangents.right_jacobian_se3": (tangents.right_jacobian_se3, (twist,), {}),
        "tangents.right_jacobian_so3": (tangents.right_jacobian_so3, (twist[3:],), {}),
        "tangents.vee_se3": (tangents.vee_se3, (tangents.hat_se3(twist),), {}),
        "tangents.vee_so3": (tangents.vee_so3, (tangents.hat_so3(twist[3:]),), {}),
        "spatial.cross_motion": (spatial.cross_motion, (twist, other), {}),
        "spatial.cross_force": (spatial.cross_force, (twist, wrench), {}),
        "spatial.transform_motion": (spatial.transform_motion, (pose_a, twist), {}),
        "spatial.transform_force": (spatial.transform_force, (pose_a, wrench), {}),
        "spatial.transform_inertia": (spatial.transform_inertia, (pose_a, inertia), {}),
        "spatial.inertia_matrix": (spatial.inertia_matrix, (inertia,), {}),
        "spatial.rotational_inertia": (spatial.rotational_inertia, (inertia,), {}),
        "spatial.inertia_from_mass_com": (
            spatial.inertia_from_mass_com,
            (mass, point),
            {"rotational_inertia": torch.diag(sizes)},
        ),
        "spatial.inertia_of_sphere": (spatial.inertia_of_sphere, (mass,), {"radius": angle}),
        "spatial.inertia_of_box": (spatial.inertia_of_box, (mass,), {"size": sizes}),
        "spatial.inertia_of_capsule": (spatial.inertia_of_capsule, (mass,), {"radius": fraction, "length": angle}),
        "spatial.inertia_of_ellipsoid": (spatial.inertia_of_ellipsoid, (mass,), {"semi_axes": sizes}),
        "spatial.inertia_of_mesh": (spatial.inertia_of_mesh, (cube, faces), {}),
    }


def _functions(prefix: str, module: ModuleType) -> dict[str, Callable[..., Any]]:
    return {
        f"{prefix}{name}": getattr(module, name) for name in module.__all__ if inspect.isfunction(getattr(module, name))
    }


#: Every public function of kinematics, dynamics, lie and spatial, by registry name.
PUBLIC_FUNCTIONS: dict[str, Callable[..., Any]] = {
    **_functions("", kinematics),
    **_functions("", dynamics),
    **_functions("lie.", lie),
    **_functions("se3.", se3),
    **_functions("so3.", so3),
    **_functions("tangents.", tangents),
    **_functions("spatial.", spatial),
}
_TOPOLOGY = "reads only the model's topology: no tensor input, integer ids out"
_NO_TENSOR = "takes no tensor: dtype, device and batch shape are its arguments"
_FACES = "an integer face-table routine: no floating result whose dtype or batch could change"
#: Public functions without a case, each with the reason no generic contract test can drive it.
NOT_DRIVEN: dict[str, str] = {
    "euclidean_columns": _TOPOLOGY,
    "subtree": _TOPOLOGY,
    "support": _TOPOLOGY,
    "se3.identity": _NO_TENSOR,
    "so3.identity": _NO_TENSOR,
    "spatial.orient_faces_by_component": _FACES,
    "spatial.orient_faces_consistently": _FACES,
    "spatial.validate_closed_manifold": _FACES,
}


def build_case(name: str, dtype: torch.dtype, device: torch.device, *, variant: int = 0) -> Case:
    """The ``(fn, args, kwargs)`` triple of entry point ``name`` in ``dtype`` on ``device``; ``variant`` scales
    every input by ``1 + variant / 10``, which keeps it valid."""
    scale = 1.0 + variant / 10
    return {**_kinematic_cases(dtype, device, scale), **_lie_spatial_cases(dtype, device, scale)}[name]


CASE_NAMES: tuple[str, ...] = tuple(
    {
        **_kinematic_cases(torch.float64, torch.device("cpu"), 1.0),
        **_lie_spatial_cases(torch.float64, torch.device("cpu"), 1.0),
    }
)
