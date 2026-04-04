"""Tests for Robot and forward kinematics."""

import pytest
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
    with pytest.raises(ValueError, match="not found"):
        robot.get_link_index("nonexistent_link")


def test_forward_kinematics_with_base_pose_shape():
    """FK with base_pose returns same shape as without."""
    robot = _load_panda()
    n = robot.joints.num_actuated_joints
    cfg = torch.zeros(n)
    base_pose = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # translate x=1
    poses_base = robot.forward_kinematics(cfg, base_pose=base_pose)
    poses_orig = robot.forward_kinematics(cfg)
    assert poses_base.shape == poses_orig.shape


def test_forward_kinematics_base_pose_translates_all_links():
    """A pure-translation base_pose shifts every link position by the same offset."""
    robot = _load_panda()
    n = robot.joints.num_actuated_joints
    cfg = robot._default_cfg
    offset = torch.tensor([1.0, 2.0, 3.0])
    base_pose = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])  # identity rotation
    poses_orig = robot.forward_kinematics(cfg)
    poses_base = robot.forward_kinematics(cfg, base_pose=base_pose)
    # Every link should be shifted by (1, 2, 3) in position (first 3 dims)
    # Orientation should be unchanged (base rotation is identity)
    assert torch.allclose(poses_base[..., :3], poses_orig[..., :3] + offset, atol=1e-4)
    assert torch.allclose(poses_base[..., 3:], poses_orig[..., 3:], atol=1e-4)


def test_forward_kinematics_none_base_unchanged():
    """base_pose=None is identical to omitting base_pose."""
    robot = _load_panda()
    n = robot.joints.num_actuated_joints
    cfg = robot._default_cfg
    assert torch.allclose(
        robot.forward_kinematics(cfg, base_pose=None),
        robot.forward_kinematics(cfg),
        atol=1e-6,
    )


@pytest.fixture(scope="session")
def panda():
    return _load_panda()


def test_get_chain_panda_hand(panda):
    """Chain from root to panda_hand has exactly 7 actuated joints."""
    hand_idx = panda.get_link_index("panda_hand")
    chain = panda.get_chain(hand_idx)
    assert len(chain) == 7
    # All returned indices are actuated joints
    assert all(panda._fk_cfg_indices[j] >= 0 for j in chain)


def test_get_chain_root_link(panda):
    """Chain to root link is empty."""
    chain = panda.get_chain(panda._root_link_idx)
    assert chain == []


def test_get_chain_ordering(panda):
    """Chain is in root→EE topological order (parent joint before child joint)."""
    hand_idx = panda.get_link_index("panda_hand")
    chain = panda.get_chain(hand_idx)
    # cfg indices should be 0,1,2,... (BFS order for Panda arm)
    cfg_indices = [panda._fk_cfg_indices[j] for j in chain]
    assert cfg_indices == sorted(cfg_indices)


import torch
from better_robot.core._lie_ops import adjoint_se3, se3_identity, se3_inverse


def test_adjoint_se3_identity():
    """Ad(identity) = I_6."""
    T = se3_identity()
    Ad = adjoint_se3(T)
    assert Ad.shape == (6, 6)
    assert torch.allclose(Ad, torch.eye(6), atol=1e-6)


def test_adjoint_se3_pure_translation():
    """Ad([1,0,0, 0,0,0,1]) has skew(p)@R in top-right block."""
    T = torch.tensor([1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0])  # x=1, identity rotation
    Ad = adjoint_se3(T)
    # Top-left: R = I
    assert torch.allclose(Ad[:3, :3], torch.eye(3), atol=1e-6)
    # Bottom-right: R = I
    assert torch.allclose(Ad[3:, 3:], torch.eye(3), atol=1e-6)
    # Bottom-left: zeros
    assert torch.allclose(Ad[3:, :3], torch.zeros(3, 3), atol=1e-6)
    # Top-right: skew([1,0,0]) @ I = [[0,0,0],[0,0,-1],[0,1,0]]
    expected_top_right = torch.tensor([
        [ 0.,  0.,  0.],
        [ 0.,  0., -1.],
        [ 0.,  1.,  0.],
    ])
    assert torch.allclose(Ad[:3, 3:], expected_top_right, atol=1e-6)
