"""Tests for cost residuals."""

import pytest
import torch
from robot_descriptions.loaders.yourdfpy import load_robot_description
from better_robot import Robot
from better_robot.costs import pose_residual, limit_residual, rest_residual


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return Robot.from_urdf(urdf)


def test_pose_residual_shape(panda):
    cfg = panda._default_cfg
    hand_idx = panda.get_link_index("panda_hand")
    # Get FK at default config to use as target (should give ~zero residual)
    fk = panda.forward_kinematics(cfg)
    target = fk[hand_idx].detach()

    res = pose_residual(cfg, panda, hand_idx, target)
    assert res.shape == (6,)


def test_pose_residual_zero_at_target(panda):
    cfg = panda._default_cfg
    hand_idx = panda.get_link_index("panda_hand")
    fk = panda.forward_kinematics(cfg)
    target = fk[hand_idx].detach()

    res = pose_residual(cfg, panda, hand_idx, target)
    assert res.abs().max() < 1e-4, f"Expected ~zero residual, got {res}"


def test_pose_residual_gradient(panda):
    cfg = panda._default_cfg.clone().requires_grad_(True)
    hand_idx = panda.get_link_index("panda_hand")
    target = torch.tensor([0.3, 0.0, 0.5, 0., 0., 0., 1.])

    res = pose_residual(cfg, panda, hand_idx, target)
    loss = (res ** 2).sum()
    loss.backward()
    assert cfg.grad is not None
    assert not torch.any(torch.isnan(cfg.grad))


def test_limit_residual_shape(panda):
    cfg = panda._default_cfg
    res = limit_residual(cfg, panda)
    assert res.shape == (2 * panda.joints.num_actuated_joints,)


def test_limit_residual_no_violation_at_default(panda):
    cfg = panda._default_cfg
    res = limit_residual(cfg, panda)
    # Default config is midpoint of limits, should have no violations
    assert (res <= 0).all(), f"Unexpected violations: {res}"


def test_rest_residual_zero_at_rest(panda):
    cfg = panda._default_cfg
    res = rest_residual(cfg, cfg)
    assert res.abs().max() < 1e-6


def test_pose_residual_with_base_pose_zero_at_fk(panda):
    """pose_residual with base_pose=fk-target is zero when cfg matches."""
    cfg = panda._default_cfg
    base_pose = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    fk = panda.forward_kinematics(cfg, base_pose=base_pose)
    hand_idx = panda.get_link_index("panda_hand")
    target = fk[hand_idx].detach()
    res = pose_residual(cfg, panda, hand_idx, target, base_pose=base_pose)
    assert res.shape == (6,)
    assert res.abs().max().item() < 1e-4
