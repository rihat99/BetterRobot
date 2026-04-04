"""Tests for the unified solve_ik API."""

import torch
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description
from better_robot import Robot, solve_ik, IKConfig


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return Robot.from_urdf(urdf)


# --- Fixed-base tests ---

def test_solve_ik_returns_correct_shape(panda):
    target = torch.tensor([0.3, 0.0, 0.5, 0., 0., 0., 1.])
    result = solve_ik(panda, targets={"panda_hand": target}, max_iter=30)
    assert result.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_respects_joint_limits(panda):
    target = torch.tensor([0.3, 0.0, 0.5, 0., 0., 0., 1.])
    result = solve_ik(panda, targets={"panda_hand": target}, max_iter=50)
    lo = panda.joints.lower_limits
    hi = panda.joints.upper_limits
    assert (result >= lo - 0.1).all(), f"Below lower limits: {result}"
    assert (result <= hi + 0.1).all(), f"Above upper limits: {result}"


def test_solve_ik_converges_to_reachable_pose(panda):
    """Target is FK of default config — guaranteed reachable, should converge in few iters."""
    cfg_default = panda._default_cfg
    fk = panda.forward_kinematics(cfg_default)
    hand_idx = panda.get_link_index("panda_hand")
    target = fk[hand_idx].detach()

    result = solve_ik(
        panda,
        targets={"panda_hand": target},
        initial_cfg=cfg_default.clone(),
        max_iter=5,
    )
    fk_result = panda.forward_kinematics(result)
    pos_error = (fk_result[hand_idx, :3] - target[:3]).norm().item()
    assert pos_error < 0.05, f"Position error too large: {pos_error}"


def test_solve_ik_with_custom_config(panda):
    target = torch.tensor([0.4, 0.0, 0.4, 0., 0., 0., 1.])
    result = solve_ik(
        panda,
        targets={"panda_hand": target},
        cfg=IKConfig(pose_weight=2.0, limit_weight=0.5, rest_weight=0.001),
        max_iter=30,
    )
    assert result.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_multi_target_shape(panda):
    """Multi-target is just solve_ik with multiple keys in targets dict."""
    cfg = panda._default_cfg
    fk = panda.forward_kinematics(cfg)
    targets = {
        "panda_link6": fk[panda.get_link_index("panda_link6")].detach(),
        "panda_hand": fk[panda.get_link_index("panda_hand")].detach(),
    }
    result = solve_ik(panda, targets=targets, max_iter=5)
    assert result.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_multi_target_converges(panda):
    cfg = panda._default_cfg
    fk = panda.forward_kinematics(cfg)
    hand_idx = panda.get_link_index("panda_hand")
    link6_idx = panda.get_link_index("panda_link6")
    targets = {
        "panda_link6": fk[link6_idx].detach(),
        "panda_hand": fk[hand_idx].detach(),
    }
    result = solve_ik(panda, targets=targets, initial_cfg=cfg.clone(), max_iter=5)
    fk_result = panda.forward_kinematics(result)
    assert (fk_result[hand_idx, :3] - fk[hand_idx, :3]).norm().item() < 0.05
    assert (fk_result[link6_idx, :3] - fk[link6_idx, :3]).norm().item() < 0.05


# --- Floating-base tests ---

def test_solve_ik_floating_base_return_shapes(panda):
    """Floating-base solve_ik returns (base_pose(7), cfg(n)) tuple."""
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    hand_idx = panda.get_link_index("panda_hand")
    fk = panda.forward_kinematics(panda._default_cfg)
    targets = {"panda_hand": fk[hand_idx].detach()}

    base_pose, cfg = solve_ik(
        panda, targets=targets, initial_base_pose=identity_base, max_iter=3
    )
    assert base_pose.shape == (7,)
    assert cfg.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_floating_base_converges(panda):
    """Target is FK of default config at identity base — should converge quickly."""
    cfg0 = panda._default_cfg
    hand_idx = panda.get_link_index("panda_hand")
    fk0 = panda.forward_kinematics(cfg0)
    target = fk0[hand_idx].detach()
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

    base_pose, cfg = solve_ik(
        panda,
        targets={"panda_hand": target},
        initial_base_pose=identity_base,
        initial_cfg=cfg0.clone(),
        max_iter=5,
    )
    fk_result = panda.forward_kinematics(cfg, base_pose=base_pose)
    pos_err = (fk_result[hand_idx, :3] - target[:3]).norm().item()
    assert pos_err < 0.05, f"Position error: {pos_err}"


def test_solve_ik_floating_base_base_moves(panda):
    """Target 1 m in X from default FK — only reachable by translating the base."""
    cfg0 = panda._default_cfg
    fk0 = panda.forward_kinematics(cfg0)
    hand_idx = panda.get_link_index("panda_hand")
    target = fk0[hand_idx].detach().clone()
    target[0] += 1.0
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

    base_pose, _ = solve_ik(
        panda, targets={"panda_hand": target},
        initial_base_pose=identity_base,
        max_iter=30,
    )
    assert base_pose[0].abs().item() > 0.1, f"Base did not translate: {base_pose}"
