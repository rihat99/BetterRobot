"""The coupling standard (record 0026, principle 1.3): every line of ``docs/contracts/coupling-standard.md`` checked
against the library.

Each test but the last two reads the library only and compares it with formulas written here (Rodrigues' rotation,
the scalar-last quaternion matrix, the Euler factors, finite differences), on one-joint ``ModelBuilder`` models and a
two-joint arm in float64. The last two read the page: no line of its ``text`` blocks leaves it, and every line it adds
is frozen in ``tests/contract/coupling_standard_lines.txt`` in the same commit (a run of whitespace counts as one
space).
"""

from __future__ import annotations

import math
import re
import socket
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot import kinematics, spatial
from better_robot.exceptions import BetterRobotError, ShapeError
from better_robot.io import ModelBuilder, build_model
from better_robot.lie import se3, so3
from better_robot.model import (
    JointFixed,
    JointFreeFlyer,
    JointHelical,
    JointModel,
    JointPlanar,
    JointPrismatic,
    JointRevolute,
    JointRevoluteUnbounded,
    JointSpherical,
    JointTranslation,
    Model,
)
from better_robot.optim import Problem
from better_robot.testing import public_objects
from better_robot.viewer import Viewer
from better_robot.viewer.layers import FramesLayer

PAGE = Path(__file__).resolve().parents[2] / "docs" / "contracts" / "coupling-standard.md"
FROZEN = Path(__file__).parent / "coupling_standard_lines.txt"
F64 = torch.float64
AXIS = (0.0, 0.6, 0.8)
ANGLE = 0.7
STEP = 0.4
Z = (0.0, 0.0, 1.0)
QUARTER_TURN_Z = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))  # [qx, qy, qz, qw]
QUATERNION = tuple(value / math.sqrt(1.02) for value in (0.2, -0.4, 0.1, 0.9))  # unit, [qx, qy, qz, qw]
ARM_Q = (0.4, -0.7)


def _tensor(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=F64)


def _rotation(axis: Sequence[float], angle: float) -> torch.Tensor:
    """Rodrigues: the rotation ``(3, 3)`` by ``angle`` about the unit ``axis``."""
    x, y, z = axis
    skew = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=F64)
    return torch.eye(3, dtype=F64) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def _quaternion_matrix(quaternion: Sequence[float]) -> torch.Tensor:
    """The rotation ``(3, 3)`` of a unit quaternion written ``[x, y, z, w]``."""
    x, y, z, w = quaternion
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=F64,
    )


def _pose(translation: Sequence[float], axis: Sequence[float], angle: float) -> torch.Tensor:
    """``[t, qx, qy, qz, qw] (7,)``: ``translation`` and the rotation by ``angle`` about the unit ``axis``."""
    half = 0.5 * angle
    return _tensor(*translation, *(math.sin(half) * value for value in axis), math.cos(half))


def _one_joint(kind: JointModel) -> Model:
    """A body on one joint of ``kind`` under the universe: joint 1, placed at the identity."""
    builder = ModelBuilder("one_joint")
    builder.add_body("link", mass=1.0)
    builder.add_joint("joint", kind=kind, parent="world", child="link")
    return build_model(builder.finalize(), dtype=F64)


def _arm(*, root_joint: JointModel | None = None) -> Model:
    """A base, a shoulder about z and an elbow about y, and a frame ``tip`` on the forearm; no placement is the
    identity, so the world, local and local-world-aligned Jacobians all differ."""
    builder = ModelBuilder("arm")
    builder.add_body("base")
    builder.add_body("upper", mass=1.5, com=_tensor(0.2, 0.0, 0.0), inertia=0.01 * torch.eye(3, dtype=F64))
    builder.add_body("fore", mass=0.8, com=_tensor(0.1, 0.02, 0.0), inertia=0.005 * torch.eye(3, dtype=F64))
    shoulder, elbow = JointRevolute(axis=Z), JointRevolute(axis=(0.0, 1.0, 0.0))
    builder.add_joint(
        "shoulder", kind=shoulder, parent="base", child="upper", origin=_pose((0, 0, 0.1), (1, 0, 0), 0.3)
    )
    builder.add_joint("elbow", kind=elbow, parent="upper", child="fore", origin=_pose((0.4, 0, 0), (0, 0, 1), -0.2))
    builder.add_frame("tip", parent="fore", placement=_pose((0.3, 0.05, 0.1), (0, 1, 0), 0.5))
    return build_model(builder.finalize(), root_joint=root_joint, dtype=F64)


def _layouts() -> list:
    """``(kind, q, nv, rotation, translation)``: the child placement that ``q`` gives, from the page's layout."""
    rotation, identity, zero = _rotation(AXIS, ANGLE), torch.eye(3, dtype=F64), torch.zeros(3, dtype=F64)
    axis = _tensor(*AXIS)
    cos, sin = math.cos(ANGLE), math.sin(ANGLE)
    return [
        pytest.param(JointFixed(), [], 0, identity, zero, id="fixed"),
        pytest.param(JointRevolute(axis=AXIS), [ANGLE], 1, rotation, zero, id="revolute"),
        pytest.param(JointPrismatic(axis=AXIS), [0.3], 1, identity, 0.3 * axis, id="prismatic"),
        pytest.param(JointHelical(axis=AXIS, pitch=0.2), [ANGLE], 1, rotation, 0.2 * ANGLE * axis, id="helical"),
        pytest.param(JointRevoluteUnbounded(axis=AXIS), [cos, sin], 1, rotation, zero, id="continuous"),
        pytest.param(JointSpherical(), list(QUATERNION), 3, _quaternion_matrix(QUATERNION), zero, id="spherical"),
        pytest.param(
            JointPlanar(), [0.3, -0.2, cos, sin], 3, _rotation(Z, ANGLE), _tensor(0.3, -0.2, 0.0), id="planar"
        ),
        pytest.param(JointTranslation(), [0.3, -0.2, 0.5], 3, identity, _tensor(0.3, -0.2, 0.5), id="translation"),
        pytest.param(
            JointFreeFlyer(),
            [0.3, -0.2, 0.5, *QUATERNION],
            6,
            _quaternion_matrix(QUATERNION),
            _tensor(0.3, -0.2, 0.5),
            id="free flyer",
        ),
    ]


@pytest.mark.parametrize(("kind", "q", "nv", "rotation", "translation"), _layouts())
def test_q_layout_per_joint_kind(
    kind: JointModel, q: list[float], nv: int, rotation: torch.Tensor, translation: torch.Tensor
) -> None:
    """A one-joint model has the kind's slice widths, and its ``q`` places the child as the page's layout says."""
    model = _one_joint(kind)
    assert (model.nqs[1], model.nvs[1]) == (len(q), nv)
    state = br.forward_kinematics(model, torch.tensor(q, dtype=F64))
    torch.testing.assert_close(state.joint_rotations[1], rotation)
    torch.testing.assert_close(state.joint_translations[1], translation)


def _steps() -> list:
    """``(kind, q0, v, rotation, translation)``: the child placement after the step ``v`` from ``q0``; the child frame
    of the spherical, planar and free-flyer cases starts a quarter turn about z, so a step in it differs from one in
    the parent frame."""
    after, axis, zero = _rotation(AXIS, 0.2 + STEP), _tensor(*AXIS), torch.zeros(3, dtype=F64)
    quarter, turned = _rotation(Z, math.pi / 2), _rotation(Z, math.pi / 2) @ _rotation((1, 0, 0), STEP)
    return [
        pytest.param(JointRevolute(axis=AXIS), [0.2], [STEP], after, zero, id="revolute"),
        pytest.param(JointPrismatic(axis=AXIS), [0.2], [STEP], torch.eye(3, dtype=F64), 0.6 * axis, id="prismatic"),
        pytest.param(JointHelical(axis=AXIS, pitch=0.2), [0.2], [STEP], after, 0.12 * axis, id="helical"),
        pytest.param(
            JointRevoluteUnbounded(axis=AXIS), [math.cos(0.2), math.sin(0.2)], [STEP], after, zero, id="continuous"
        ),
        pytest.param(JointSpherical(), list(QUARTER_TURN_Z), [STEP, 0, 0], turned, zero, id="spherical"),
        pytest.param(JointPlanar(), [1, 2, 0, 1], [1, 0, 0], quarter, _tensor(1, 3, 0), id="planar linear"),
        pytest.param(
            JointPlanar(),
            [1, 2, 0, 1],
            [0, 0, STEP],
            _rotation(Z, math.pi / 2 + STEP),
            _tensor(1, 2, 0),
            id="planar angular",
        ),
        pytest.param(
            JointTranslation(),
            [1, 2, 3],
            [0.1, 0.2, 0.3],
            torch.eye(3, dtype=F64),
            _tensor(1.1, 2.2, 3.3),
            id="translation",
        ),
        pytest.param(
            JointFreeFlyer(),
            [1, 2, 3, *QUARTER_TURN_Z],
            [1, 0, 0, 0, 0, 0],
            quarter,
            _tensor(1, 3, 3),
            id="free flyer linear",
        ),
        pytest.param(
            JointFreeFlyer(),
            [1, 2, 3, *QUARTER_TURN_Z],
            [0, 0, 0, STEP, 0, 0],
            turned,
            _tensor(1, 2, 3),
            id="free flyer angular",
        ),
    ]


@pytest.mark.parametrize(("kind", "q0", "v", "rotation", "translation"), _steps())
def test_v_is_the_tangent_of_q_in_the_child_frame(
    kind: JointModel, q0: list[float], v: list[float], rotation: torch.Tensor, translation: torch.Tensor
) -> None:
    """A step ``v`` moves the child as the page's ``v`` column says, and ``v`` is the difference of the two ``q``."""
    model = _one_joint(kind)
    q0_tensor, v_tensor = torch.tensor(q0, dtype=F64), torch.tensor(v, dtype=F64)
    q1 = kinematics.integrate(model, q0_tensor, v_tensor)
    state = br.forward_kinematics(model, q1)
    torch.testing.assert_close(state.joint_rotations[1], rotation)
    torch.testing.assert_close(state.joint_translations[1], translation)
    torch.testing.assert_close(kinematics.difference(model, q0_tensor, q1), v_tensor)


def test_pose_rotation_and_transform_layouts() -> None:
    """A pose is ``[t, qx, qy, qz, qw]``, a quaternion and its negative are one rotation, a transform is ``[R | p]``."""
    translation, rotation = _tensor(0.3, -0.2, 0.5), _quaternion_matrix(QUATERNION)
    pose = torch.cat((translation, _tensor(*QUATERNION)))
    matrix = se3.to_matrix(pose)
    torch.testing.assert_close(matrix[:3, :3], rotation)
    torch.testing.assert_close(matrix[:3, 3], translation)
    torch.testing.assert_close(so3.to_matrix(_tensor(*QUATERNION)), rotation)
    torch.testing.assert_close(so3.to_matrix(-_tensor(*QUATERNION)), rotation)
    state = br.forward_kinematics(_one_joint(JointFreeFlyer()), pose)
    torch.testing.assert_close(state.joint_transforms[1], torch.cat((rotation, translation[:, None]), dim=-1))
    torch.testing.assert_close(state.joint_poses[1, :3], translation)
    torch.testing.assert_close(_quaternion_matrix(state.joint_poses[1, 3:].tolist()), rotation)


def test_twist_is_linear_first() -> None:
    """The twist ``[v, 0]`` translates by ``v``; ``[0, w]`` turns about ``w`` and does not translate."""
    torch.testing.assert_close(se3.exp(_tensor(0.3, -0.2, 0.5, 0, 0, 0)), _tensor(0.3, -0.2, 0.5, 0, 0, 0, 1))
    turn = se3.to_matrix(se3.exp(_tensor(0, 0, 0, *(ANGLE * value for value in AXIS))))
    torch.testing.assert_close(turn[:3, :3], _rotation(AXIS, ANGLE))
    torch.testing.assert_close(turn[:3, 3], torch.zeros(3, dtype=F64))


def test_wrench_is_force_first() -> None:
    """Holding a floating body still takes ``[f, c x f]`` with ``f = -m g``, and that wrench applied to it holds it."""
    mass, com = 2.0, _tensor(0.3, 0.0, 0.0)
    builder = ModelBuilder("floating", gravity=(0.0, 0.0, -9.81))
    builder.add_body("body", mass=mass, com=com)
    builder.add_joint("base", kind=JointFreeFlyer(), parent="world", child="body")
    model = build_model(builder.finalize(), dtype=F64)
    state = br.forward_kinematics(model, model.q_neutral)
    still = torch.zeros(6, dtype=F64)
    force = _tensor(0.0, 0.0, mass * 9.81)
    wrench = torch.cat((force, torch.linalg.cross(com, force)))
    torch.testing.assert_close(br.inverse_dynamics(model, state, v=still, a=still), wrench)
    applied = torch.zeros(model.njoints, 6, dtype=F64)
    applied[1] = wrench
    torch.testing.assert_close(br.inverse_dynamics(model, state, v=still, a=still, external_forces=applied), still)


def test_inertia_layout() -> None:
    """A body's row is ``[m, c, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`` with the rotational inertia about the centre of mass."""
    mass, com = 2.0, _tensor(0.1, 0.2, 0.3)
    rotational = torch.tensor([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]], dtype=F64)
    builder = ModelBuilder("body")
    builder.add_body("body", mass=mass, com=com, inertia=rotational)
    builder.add_joint("joint", kind=JointFixed(), parent="world", child="body")
    inertia = build_model(builder.finalize(), dtype=F64).body_inertias[1]
    torch.testing.assert_close(inertia, _tensor(2.0, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3))
    torch.testing.assert_close(spatial.rotational_inertia(inertia), rotational)
    about_origin = spatial.inertia_matrix(inertia)  # (6, 6), linear first: the parallel-axis shift of the 3x3
    torch.testing.assert_close(about_origin[:3, :3], mass * torch.eye(3, dtype=F64))
    shifted = rotational + mass * (com.dot(com) * torch.eye(3, dtype=F64) - torch.outer(com, com))
    torch.testing.assert_close(about_origin[3:, 3:], shifted)


def test_euler_angles_compose_rz_ry_rx() -> None:
    """``[roll, pitch, yaw]`` is the rotation ``Rz(yaw) Ry(pitch) Rx(roll)``, both ways."""
    roll, pitch, yaw = 0.3, -0.5, 1.1
    rotation = _rotation(Z, yaw) @ _rotation((0, 1, 0), pitch) @ _rotation((1, 0, 0), roll)
    torch.testing.assert_close(so3.to_matrix(so3.from_euler(_tensor(roll, pitch, yaw))), rotation)
    torch.testing.assert_close(so3.to_euler(so3.from_matrix(rotation)), _tensor(roll, pitch, yaw))


@pytest.mark.parametrize("target", ["frame", "joint"])
def test_jacobian_default_is_local_world_aligned(target: str) -> None:
    """The default Jacobian ``(6, nv)`` maps ``v`` to the target origin's velocity and the angular velocity, both in
    world axes (central differences of the placement along ``v``)."""
    model, step = _arm(), 1e-6
    q, v = _tensor(*ARM_Q), _tensor(0.9, -0.3)

    def placement(configuration: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = br.forward_kinematics(model, configuration)
        if target == "frame":
            tip = model.frame_id("tip")
            return state.frame_rotations[tip], state.frame_translations[tip]
        elbow = model.joint_id("elbow")
        return state.joint_rotations[elbow], state.joint_translations[elbow]

    state = br.forward_kinematics(model, q)
    if target == "frame":
        jacobian = br.frame_jacobian(model, state, frame="tip")
    else:
        jacobian = br.joint_jacobian(model, state, joint="elbow")
    (rotation_plus, origin_plus), (rotation_minus, origin_minus) = (
        placement(kinematics.integrate(model, q, sign * step * v)) for sign in (1.0, -1.0)
    )
    spin = (rotation_plus - rotation_minus) / (2 * step) @ placement(q)[0].mT  # [w]x in world axes
    expected = torch.cat(((origin_plus - origin_minus) / (2 * step), torch.stack((spin[2, 1], spin[0, 2], spin[1, 0]))))
    assert jacobian.shape == (6, model.nv)
    torch.testing.assert_close(jacobian @ v, expected, rtol=0.0, atol=1e-8)


def test_joint_zero_is_the_fixed_identity_universe() -> None:
    """Joint 0, body 0 and frame 0 are "universe": fixed, at the identity, without a parent or coordinates."""
    model = _arm()
    assert (model.joint_names[0], model.body_names[0], model.frame_names[0]) == ("universe",) * 3
    assert isinstance(model.joints[0], JointFixed)
    assert (model.parent_joint_ids[0], model.nqs[0], model.nvs[0]) == (-1, 0, 0)
    torch.testing.assert_close(model.joint_placements[0], _tensor(0, 0, 0, 0, 0, 0, 1))
    state = br.forward_kinematics(model, _tensor(*ARM_Q))
    torch.testing.assert_close(state.joint_transforms[0], torch.eye(3, 4, dtype=F64))


def test_joint_one_is_the_root() -> None:
    """Joint 1 holds the root body under the universe: fixed when nothing says otherwise, else the given kind."""
    fixed = _arm()
    assert (fixed.body_names[1], fixed.parent_joint_ids[1], type(fixed.joints[1])) == ("base", 0, JointFixed)
    floating = _arm(root_joint=JointFreeFlyer())
    assert (floating.body_names[1], type(floating.joints[1])) == ("base", JointFreeFlyer)
    assert (floating.idx_qs[1], floating.idx_vs[1]) == (0, 0)
    written = _one_joint(JointFreeFlyer())  # the description's own joint from "world"
    assert (written.joint_names[1], written.body_names[1], type(written.joints[1])) == ("joint", "link", JointFreeFlyer)


def _joint_order(description: Sequence[tuple[str, str, str]]) -> tuple[str, ...]:
    """The model's joint names after the universe and the root, for joints ``(name, parent, child)`` in this order."""
    builder = ModelBuilder("tree")
    for body in ("base", "upper", "fore", "sensor"):
        builder.add_body(body)
    for name, parent, child in description:
        builder.add_joint(name, kind=JointRevolute(axis=Z), parent=parent, child=child)
    return build_model(builder.finalize(), dtype=F64).joint_names[2:]


def test_joints_come_parents_first_then_in_description_order() -> None:
    """A description in a parents-first order keeps it (a depth-first walk would not); a child listed before its
    parent follows it, ahead of the joints listed after it."""
    shoulder, elbow, mount = ("shoulder", "base", "upper"), ("elbow", "upper", "fore"), ("mount", "base", "sensor")
    assert _joint_order([shoulder, mount, elbow]) == ("shoulder", "mount", "elbow")
    assert _joint_order([elbow, shoulder, mount]) == ("shoulder", "elbow", "mount")


def test_mimic_joints_appear_only_in_the_full_layout() -> None:
    """A mimic joint has no public slice; the full layout holds it as ``multiplier * q_source + offset``."""
    builder = ModelBuilder("mimic")
    for body in ("base", "left", "right"):
        builder.add_body(body)
    builder.add_joint("driver", kind=JointRevolute(axis=Z), parent="base", child="left")
    builder.add_joint(
        "follower",
        kind=JointRevolute(axis=Z),
        parent="left",
        child="right",
        mimic_source="driver",
        mimic_multiplier=2.0,
        mimic_offset=0.1,
    )
    model = build_model(builder.finalize(), dtype=F64)
    follower = model.joint_id("follower")
    assert (model.nq, model.nv, model.nqs[follower], model.nvs[follower]) == (1, 1, 0, 0)
    assert (model.nq_full, model.nv_full, model.nqs_full[follower], model.nvs_full[follower]) == (2, 2, 1, 1)
    torch.testing.assert_close(kinematics.full_configuration(model, _tensor(0.3)), _tensor(0.3, 0.7))
    state = br.forward_kinematics(model, _tensor(0.3))
    torch.testing.assert_close(state.joint_rotations[follower], _rotation(Z, 1.0))


@pytest.mark.parametrize("kind", ["joint", "frame", "body"])
def test_ids_take_an_int_or_a_name_and_check_both(kind: str) -> None:
    """Every id and every name resolves to the same id; an int outside ``[0, n)`` or an unknown name raises."""
    model = _arm()
    resolve, names = getattr(model, f"{kind}_id"), getattr(model, f"{kind}_names")
    assert (
        [resolve(name) for name in names] == [resolve(index) for index in range(len(names))] == list(range(len(names)))
    )
    for bad in (-1, len(names), "missing"):
        with pytest.raises(BetterRobotError):
            resolve(bad)


def test_a_public_boundary_takes_the_id_or_the_name() -> None:
    """A Jacobian of the frame given by id equals the one given by name; an id out of range raises."""
    model = _arm()
    state = br.forward_kinematics(model, _tensor(*ARM_Q))
    by_name = br.frame_jacobian(model, state, frame="tip")
    torch.testing.assert_close(br.frame_jacobian(model, state, frame=model.frame_id("tip")), by_name)
    with pytest.raises(BetterRobotError):
        br.frame_jacobian(model, state, frame=model.nframes)


def test_names_stay_as_the_description_gives_them() -> None:
    """Joint, body and frame names keep every character; each body has a frame of its name, placed on the body."""
    builder = ModelBuilder("names")
    builder.add_body("Pelvis")
    builder.add_body("L_Hip.link-1")
    builder.add_joint("L_Hip joint", kind=JointSpherical(), parent="Pelvis", child="L_Hip.link-1")
    builder.add_frame("heel contact", parent="L_Hip.link-1", placement=_tensor(0, 0, -0.1, 0, 0, 0, 1))
    model = build_model(builder.finalize(), dtype=F64)
    assert model.joint_names[2:] == ("L_Hip joint",)
    assert model.body_names == ("universe", "Pelvis", "L_Hip.link-1")
    assert model.frame_names == ("universe", "Pelvis", "L_Hip.link-1", "heel contact")
    state = br.forward_kinematics(model, _tensor(*QUATERNION))
    for body_id, body in enumerate(model.body_names):
        torch.testing.assert_close(state.frame_transforms[model.frame_id(body)], state.joint_transforms[body_id])


def test_identity_batch_broadcasts_against_input_batch() -> None:
    """Model values with ``I = (2,)`` against inputs with ``B = (3, 1)`` give ``(3, 2, ...)`` results, each the
    result of one identity at one input; a ``B`` that does not broadcast with ``I`` raises ``ShapeError``."""
    model = _arm()
    longer = model.joint_placements.clone()
    longer[model.joint_id("elbow"), 0] += 0.25  # the second identity's forearm hangs further out
    placements = torch.stack((model.joint_placements, longer))  # (2, njoints, 7)
    batched = model.with_values(joint_placements=placements)
    generator = torch.Generator().manual_seed(0)
    q, v, a = (torch.randn(3, 1, 2, generator=generator, dtype=F64) for _ in range(3))
    state = br.forward_kinematics(batched, q)
    jacobian = br.frame_jacobian(batched, state, frame="tip")
    torque = br.inverse_dynamics(batched, state, v=v, a=a)
    assert state.joint_transforms.shape == (3, 2, model.njoints, 3, 4)
    assert (jacobian.shape, torque.shape) == ((3, 2, 6, model.nv), (3, 2, model.nv))
    for identity in range(2):
        single = model.with_values(joint_placements=placements[identity])
        for sample in range(3):
            one = br.forward_kinematics(single, q[sample, 0])
            torch.testing.assert_close(state.joint_transforms[sample, identity], one.joint_transforms)
            torch.testing.assert_close(jacobian[sample, identity], br.frame_jacobian(single, one, frame="tip"))
            expected = br.inverse_dynamics(single, one, v=v[sample, 0], a=a[sample, 0])
            torch.testing.assert_close(torque[sample, identity], expected)
    with pytest.raises(ShapeError):
        br.forward_kinematics(batched, torch.zeros(3, 5, model.nq, dtype=F64))


@pytest.mark.parametrize(
    "field",
    [
        "lower_position_limit",
        "upper_position_limit",
        "velocity_limit",
        "effort_limit",
        "armature",
        "damping",
        "gravity",
        "q_neutral",
    ],
)
def test_other_model_values_carry_no_batch(field: str) -> None:
    """Only placements and inertias take an identity batch: any other value with a leading axis raises."""
    model = _arm()
    value = getattr(model, field)
    with pytest.raises(ShapeError):
        model.with_values(**{field: value.expand(2, *value.shape)})


def test_optim_variables_share_one_leading_batch_axis() -> None:
    """``optim`` takes ``(B, *event)``: one ``B`` across variables of any event shape; another ``B`` or none raises."""
    Problem({"x": torch.zeros(4, 2), "y": torch.zeros(4, 3, 3)})
    with pytest.raises(ShapeError):
        Problem({"x": torch.zeros(4, 2), "y": torch.zeros(5, 2)})
    with pytest.raises(ShapeError):
        Problem({"x": torch.zeros(())})


URDF = """<robot name="hinge">
  <link name="base"/>
  <link name="arm">
    <inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>"""
MJCF = """<mujoco><worldbody>
  <body name="arm"><joint name="hinge" type="hinge" axis="0 0 1"/><geom size="0.1"/></body>
</worldbody></mujoco>"""


@pytest.mark.parametrize(("suffix", "text"), [(".urdf", URDF), (".xml", MJCF)], ids=["URDF", "MJCF"])
def test_urdf_and_mjcf_are_read(tmp_path: Path, suffix: str, text: str) -> None:
    """The same hinge written in URDF and in MJCF loads as one revolute slice that turns about its axis."""
    path = tmp_path / f"hinge{suffix}"
    path.write_text(text)
    model = br.load(path, dtype=F64)
    hinge = model.joint_id("hinge")
    assert (model.nq, model.nqs[hinge], model.nvs[hinge]) == (1, 1, 1)
    state = br.forward_kinematics(model, _tensor(ANGLE))
    torch.testing.assert_close(state.joint_rotations[hinge], _rotation(Z, ANGLE))


def test_the_viewer_writes_a_viser_recording(tmp_path: Path) -> None:
    """``Viewer`` records its timeline into a ``.viser`` file through viser's own serializer, on a loopback server."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    viewer = Viewer(port=port)
    try:
        poses = np.zeros((2, 1, 7), dtype=np.float32)
        poses[..., 6] = 1.0
        viewer.add([FramesLayer("frames", poses)])
        path = viewer.save(tmp_path / "clip.viser")
    finally:
        viewer.server.stop()
    assert path == tmp_path / "clip.viser" and path.stat().st_size > 0


def test_betterrobot_writes_no_model_file() -> None:
    """The one public writer of a file is the viewer's recording: no name saves, writes, dumps or exports a model."""
    writer = re.compile(r"(^|_)(save|write|dump|export|serialize)(_|$)")
    writers = sorted(path for path in public_objects("better_robot") if writer.search(path.rsplit(".", 1)[-1]))
    assert writers == ["better_robot.Viewer.save", "better_robot.viewer.Viewer.save"]


def _page_lines() -> set[str]:
    """Every non-blank line of the page's ``text`` blocks, with each run of whitespace as one space."""
    blocks = re.findall(r"^```text\n(.*?)^```$", PAGE.read_text(), re.MULTILINE | re.DOTALL)
    return {" ".join(line.split()) for block in blocks for line in block.splitlines() if line.strip()}


def _frozen_lines() -> list[str]:
    return FROZEN.read_text().splitlines()


def test_page_only_grows() -> None:
    """No frozen line leaves the page: removing or changing a line of the standard needs a decision record."""
    page = _page_lines()
    gone = [line for line in _frozen_lines() if line not in page]
    assert not gone, (
        "these lines of the coupling standard left docs/contracts/coupling-standard.md; put them back, or change the "
        "standard by a decision record (record 0026) whose commit also edits coupling_standard_lines.txt:\n  "
        + "\n  ".join(gone)
    )


def test_lines_added_to_the_page_are_frozen() -> None:
    """A line added to the page is added to ``coupling_standard_lines.txt`` in the same commit."""
    added = sorted(_page_lines() - set(_frozen_lines()))
    assert not added, "add these lines to tests/contract/coupling_standard_lines.txt:\n  " + "\n  ".join(added)
