"""Tests for cost residuals."""

import pytest
import torch
from robot_descriptions.loaders.yourdfpy import load_robot_description
from better_robot import load_urdf
from better_robot.costs import pose_residual, limit_residual, rest_residual


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return load_urdf(urdf)


def test_pose_residual_shape(panda):
    cfg = panda._q_default
    hand_idx = panda.link_index("panda_hand")
    # Get FK at default config to use as target (should give ~zero residual)
    fk = panda.forward_kinematics(cfg)
    target = fk[hand_idx].detach()

    res = pose_residual(cfg, panda, hand_idx, target)
    assert res.shape == (6,)


def test_pose_residual_zero_at_target(panda):
    cfg = panda._q_default
    hand_idx = panda.link_index("panda_hand")
    fk = panda.forward_kinematics(cfg)
    target = fk[hand_idx].detach()

    res = pose_residual(cfg, panda, hand_idx, target)
    assert res.abs().max() < 1e-4, f"Expected ~zero residual, got {res}"


def test_pose_residual_gradient(panda):
    cfg = panda._q_default.clone().requires_grad_(True)
    hand_idx = panda.link_index("panda_hand")
    target = torch.tensor([0.3, 0.0, 0.5, 0., 0., 0., 1.])

    res = pose_residual(cfg, panda, hand_idx, target)
    loss = (res ** 2).sum()
    loss.backward()
    assert cfg.grad is not None
    assert not torch.any(torch.isnan(cfg.grad))


def test_limit_residual_shape(panda):
    cfg = panda._q_default
    res = limit_residual(cfg, panda)
    assert res.shape == (2 * panda.joints.num_actuated_joints,)


def test_limit_residual_no_violation_at_default(panda):
    cfg = panda._q_default
    res = limit_residual(cfg, panda)
    # Default config is midpoint of limits, should have no violations
    assert (res <= 0).all(), f"Unexpected violations: {res}"


def test_rest_residual_zero_at_rest(panda):
    cfg = panda._q_default
    res = rest_residual(cfg, cfg)
    assert res.abs().max() < 1e-6


def test_pose_residual_with_base_pose_zero_at_fk(panda):
    """pose_residual with base_pose=fk-target is zero when cfg matches."""
    cfg = panda._q_default
    base_pose = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    fk = panda.forward_kinematics(cfg, base_pose=base_pose)
    hand_idx = panda.link_index("panda_hand")
    target = fk[hand_idx].detach()
    # With matching base_pose: residual should be near zero
    res = pose_residual(cfg, panda, hand_idx, target, base_pose=base_pose)
    assert res.shape == (6,)
    assert res.abs().max().item() < 1e-4
    # Without base_pose: residual must be large (proves base_pose is actually forwarded)
    res_no_base = pose_residual(cfg, panda, hand_idx, target)
    assert res_no_base.abs().max().item() > 0.1


def test_manipulability_residual_shape(panda):
    from better_robot.costs.manipulability import manipulability_residual
    cfg = panda._q_default.clone()
    link_idx = panda.link_index("panda_hand")
    r = manipulability_residual(cfg, panda, link_idx)
    assert r.shape == (1,)
    assert r.item() > 0.0  # always positive (1/manip > 0)


def test_manipulability_cost_factory(panda):
    from better_robot.costs.manipulability import manipulability_cost
    from better_robot.costs.cost_term import CostTerm
    link_idx = panda.link_index("panda_hand")
    ct = manipulability_cost(panda, link_idx, weight=2.0)
    assert isinstance(ct, CostTerm)
    cfg = panda._q_default.clone()
    r = ct.residual_fn(cfg)
    assert r.shape == (1,)
