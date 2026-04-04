"""Tests for Robot and forward kinematics."""

import torch
from better_robot import Robot
from robot_descriptions.loaders.yourdfpy import load_robot_description


def _load_panda():
    urdf = load_robot_description("panda_description")
    return Robot.from_urdf(urdf)


def test_robot_from_urdf():
    """Robot.from_urdf should succeed on panda_description."""
    robot = _load_panda()
    # 7 revolute + 2 prismatic finger joints = 9 actuated
    assert robot.joints.num_actuated_joints == 9
    assert robot.links.num_links > 0


def test_forward_kinematics_shape():
    """FK should return (num_links, 7) for a single config."""
    robot = _load_panda()
    n = robot.joints.num_actuated_joints
    poses = robot.forward_kinematics(torch.zeros(n))
    assert poses.shape == (robot.links.num_links, 7)


def test_forward_kinematics_unit_quaternions():
    """FK output quaternions should be unit norm."""
    robot = _load_panda()
    n = robot.joints.num_actuated_joints
    poses = robot.forward_kinematics(torch.zeros(n))
    quat_norms = torch.norm(poses[..., 3:7], dim=-1)
    assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5)


def test_forward_kinematics_batch():
    """FK should support batch dimensions."""
    robot = _load_panda()
    n = robot.joints.num_actuated_joints
    cfg = torch.zeros(3, 4, n)
    poses = robot.forward_kinematics(cfg)
    assert poses.shape == (3, 4, robot.links.num_links, 7)


def test_get_link_index():
    """get_link_index should return an integer for a known link."""
    robot = _load_panda()
    idx = robot.get_link_index("panda_hand")
    assert isinstance(idx, int)
    assert 0 <= idx < robot.links.num_links


def test_get_link_index_invalid():
    """get_link_index should raise ValueError for unknown link."""
    robot = _load_panda()
    import pytest
    with pytest.raises(ValueError, match="not found"):
        robot.get_link_index("nonexistent_link")
