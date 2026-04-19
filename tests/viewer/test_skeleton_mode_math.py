"""Tests that SkeletonMode sphere/cylinder transforms match
``data.joint_pose_world``.

Uses MockBackend — no viser, no pyrender.

See ``docs/12_VIEWER.md §13``.
"""

from __future__ import annotations

import math

import pytest
import torch

import better_robot as br
from better_robot.viewer.render_modes.skeleton import SkeletonMode, _align_z_to_vec
from better_robot.viewer.render_modes.base import RenderContext
from better_robot.viewer.renderers.testing import MockBackend
from better_robot.viewer.themes import DEFAULT_THEME
from robot_descriptions import panda_description


@pytest.fixture(scope="module")
def panda():
    return br.load(panda_description.URDF_PATH)


def _attach_skeleton(model) -> tuple[SkeletonMode, MockBackend]:
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test", theme=DEFAULT_THEME)
    q0 = model.q_neutral.clone().clamp(model.lower_pos_limit, model.upper_pos_limit)
    data = br.forward_kinematics(model, q0)
    mode = SkeletonMode()
    mode.attach(ctx, model, data)
    return mode, backend


def test_spheres_created_for_articulated_joints(panda):
    mode, backend = _attach_skeleton(panda)
    sphere_calls = backend.calls_for("add_sphere")
    # There must be at least one sphere (Panda has 7 revolute joints)
    assert len(sphere_calls) >= 7


def test_cylinders_created_for_links(panda):
    mode, backend = _attach_skeleton(panda)
    cyl_calls = backend.calls_for("add_cylinder")
    # Panda has 7+ non-root joints → 7+ cylinders
    assert len(cyl_calls) >= 7


def test_sphere_transforms_match_joint_pose_world(panda):
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test", theme=DEFAULT_THEME)
    q0 = panda.q_neutral.clone().clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    data = br.forward_kinematics(panda, q0)
    mode = SkeletonMode()
    mode.attach(ctx, panda, data)

    # For each sphere joint, the set_transform position must match
    # data.joint_pose_world
    for j in mode._sphere_joints:
        name = f"/test/sphere_{j}"
        pose = backend.last_transform(name)
        assert pose is not None, f"No transform set for sphere_{j}"
        expected_pos = data.joint_pose_world[j, :3]
        assert torch.allclose(pose[:3], expected_pos, atol=1e-5), \
            f"Sphere {j}: pos {pose[:3]} != expected {expected_pos}"


def test_cylinder_midpoint_correct(panda):
    backend = MockBackend()
    ctx = RenderContext(backend=backend, namespace="/test", theme=DEFAULT_THEME)
    q0 = panda.q_neutral.clone().clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    data = br.forward_kinematics(panda, q0)
    mode = SkeletonMode()
    mode.attach(ctx, panda, data)

    for j in mode._cylinder_joints:
        p_idx = panda.parents[j]
        name = f"/test/link_{j}"
        pose = backend.last_transform(name)
        assert pose is not None
        p_j = data.joint_pose_world[j, :3]
        p_p = data.joint_pose_world[p_idx, :3]
        expected_mid = (p_j + p_p) / 2
        assert torch.allclose(pose[:3], expected_mid, atol=1e-5), \
            f"Cylinder {j}: midpoint {pose[:3]} != expected {expected_mid}"


def test_update_moves_spheres(panda):
    mode, backend = _attach_skeleton(panda)
    # Update with a different configuration
    q1 = panda.q_neutral.clone().clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    q1[0] += 0.5  # shift first joint
    q1 = q1.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    data1 = br.forward_kinematics(panda, q1)
    backend.reset()
    mode.update(data1)
    # set_transform should be called for every sphere and cylinder
    assert len(backend.calls_for("set_transform")) >= len(mode._sphere_joints)


def test_detach_removes_all_nodes(panda):
    mode, backend = _attach_skeleton(panda)
    n_nodes = len(backend.nodes)
    assert n_nodes > 0
    mode.detach()
    assert len(backend.nodes) == 0


def test_align_z_to_vec_identity():
    # Aligning z to z → identity quaternion [0,0,0,1]
    d = torch.tensor([0.0, 0.0, 1.0])
    q = _align_z_to_vec(d)
    assert torch.allclose(q, torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)


def test_align_z_to_vec_flip():
    # Aligning z to -z → 180° rotation
    d = torch.tensor([0.0, 0.0, -1.0])
    q = _align_z_to_vec(d)
    # Should be a unit quaternion representing 180° rotation
    assert abs(float(q.norm()) - 1.0) < 1e-5


def test_align_z_to_vec_x():
    # Aligning z to x → 90° rotation around y
    d = torch.tensor([1.0, 0.0, 0.0])
    q = _align_z_to_vec(d)
    # Apply rotation to z-axis and check it points along x
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    # Rotate [0,0,1] by q
    # Using v' = q v q*
    v = torch.tensor([0.0, 0.0, 1.0])
    # Manual quaternion rotation
    t = 2.0 * torch.cross(q[:3], v)
    rotated = v + qw * t + torch.cross(q[:3], t)
    assert torch.allclose(rotated, d / d.norm(), atol=1e-5)
