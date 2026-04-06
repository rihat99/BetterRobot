"""Tests for the unified solve_ik API."""

import torch
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description
from better_robot import load_urdf, solve_ik, IKConfig, RobotData


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return load_urdf(urdf)


# --- Fixed-base tests ---

def test_solve_ik_returns_correct_shape(panda):
    target = torch.tensor([0.3, 0.0, 0.5, 0., 0., 0., 1.])
    data = solve_ik(panda, targets={"panda_hand": target}, max_iter=30)
    result = data.q
    assert result.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_respects_joint_limits(panda):
    target = torch.tensor([0.3, 0.0, 0.5, 0., 0., 0., 1.])
    data = solve_ik(panda, targets={"panda_hand": target}, max_iter=50)
    result = data.q
    lo = panda.joints.lower_limits
    hi = panda.joints.upper_limits
    assert (result >= lo - 0.1).all(), f"Below lower limits: {result}"
    assert (result <= hi + 0.1).all(), f"Above upper limits: {result}"


def test_solve_ik_converges_to_reachable_pose(panda):
    """Target is FK of default config — guaranteed reachable, should converge in few iters."""
    cfg_default = panda._q_default
    fk = panda.forward_kinematics(cfg_default)
    hand_idx = panda.link_index("panda_hand")
    target = fk[hand_idx].detach()

    data = solve_ik(
        panda,
        targets={"panda_hand": target},
        initial_q=cfg_default.clone(),
        max_iter=5,
    )
    result = data.q
    fk_result = panda.forward_kinematics(result)
    pos_error = (fk_result[hand_idx, :3] - target[:3]).norm().item()
    assert pos_error < 0.05, f"Position error too large: {pos_error}"


def test_solve_ik_with_custom_config(panda):
    target = torch.tensor([0.4, 0.0, 0.4, 0., 0., 0., 1.])
    data = solve_ik(
        panda,
        targets={"panda_hand": target},
        config=IKConfig(pose_weight=2.0, limit_weight=0.5, rest_weight=0.001),
        max_iter=30,
    )
    result = data.q
    assert result.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_multi_target_shape(panda):
    """Multi-target is just solve_ik with multiple keys in targets dict."""
    q = panda._q_default
    fk = panda.forward_kinematics(q)
    targets = {
        "panda_link6": fk[panda.link_index("panda_link6")].detach(),
        "panda_hand": fk[panda.link_index("panda_hand")].detach(),
    }
    data = solve_ik(panda, targets=targets, max_iter=5)
    result = data.q
    assert result.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_multi_target_converges(panda):
    q = panda._q_default
    fk = panda.forward_kinematics(q)
    hand_idx = panda.link_index("panda_hand")
    link6_idx = panda.link_index("panda_link6")
    targets = {
        "panda_link6": fk[link6_idx].detach(),
        "panda_hand": fk[hand_idx].detach(),
    }
    data = solve_ik(panda, targets=targets, initial_q=q.clone(), max_iter=5)
    result = data.q
    fk_result = panda.forward_kinematics(result)
    assert (fk_result[hand_idx, :3] - fk[hand_idx, :3]).norm().item() < 0.05
    assert (fk_result[link6_idx, :3] - fk[link6_idx, :3]).norm().item() < 0.05


# --- Floating-base tests ---

def test_solve_ik_floating_base_return_shapes(panda):
    """Floating-base solve_ik returns RobotData with .q and .base_pose."""
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    hand_idx = panda.link_index("panda_hand")
    fk = panda.forward_kinematics(panda._q_default)
    targets = {"panda_hand": fk[hand_idx].detach()}

    data = solve_ik(
        panda, targets=targets, initial_base_pose=identity_base, max_iter=3
    )
    assert isinstance(data, RobotData)
    base_pose = data.base_pose
    cfg = data.q
    assert base_pose.shape == (7,)
    assert cfg.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_floating_base_converges(panda):
    """Target is FK of default config at identity base — should converge quickly."""
    cfg0 = panda._q_default
    hand_idx = panda.link_index("panda_hand")
    fk0 = panda.forward_kinematics(cfg0)
    target = fk0[hand_idx].detach()
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

    data = solve_ik(
        panda,
        targets={"panda_hand": target},
        initial_base_pose=identity_base,
        initial_q=cfg0.clone(),
        max_iter=5,
    )
    cfg = data.q
    base_pose = data.base_pose
    fk_result = panda.forward_kinematics(cfg, base_pose=base_pose)
    pos_err = (fk_result[hand_idx, :3] - target[:3]).norm().item()
    assert pos_err < 0.05, f"Position error: {pos_err}"


def test_solve_ik_floating_base_base_moves(panda):
    """Target 1 m in X from default FK — only reachable by translating the base."""
    cfg0 = panda._q_default
    fk0 = panda.forward_kinematics(cfg0)
    hand_idx = panda.link_index("panda_hand")
    target = fk0[hand_idx].detach().clone()
    target[0] += 1.0
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

    data = solve_ik(
        panda, targets={"panda_hand": target},
        initial_base_pose=identity_base,
        max_iter=30,
    )
    base_pose = data.base_pose
    assert base_pose[0].abs().item() > 0.1, f"Base did not translate: {base_pose}"


# --- Analytic Jacobian IK tests ---

def test_solve_ik_analytic_converges(panda):
    """IKConfig(jacobian='analytic') gives correct solution for a reachable target."""
    cfg0 = panda._q_default
    hand_idx = panda.link_index("panda_hand")
    target = panda.forward_kinematics(cfg0)[hand_idx].detach()

    data = solve_ik(
        panda,
        targets={"panda_hand": target},
        config=IKConfig(jacobian="analytic"),
        initial_q=cfg0.clone(),
        max_iter=5,
    )
    cfg_analytic = data.q
    fk = panda.forward_kinematics(cfg_analytic)
    pos_err = (fk[hand_idx, :3] - target[:3]).norm().item()
    assert pos_err < 0.05, f"Analytic IK pos_err={pos_err:.4f}"


def test_solve_ik_analytic_matches_autodiff(panda):
    """Analytic and autodiff IK reach solutions within 1 cm of each other."""
    cfg0 = panda._q_default
    hand_idx = panda.link_index("panda_hand")
    target = panda.forward_kinematics(cfg0)[hand_idx].detach().clone()
    target[0] += 0.05

    data_autodiff = solve_ik(
        panda, targets={"panda_hand": target},
        config=IKConfig(jacobian="autodiff"), initial_q=cfg0.clone(), max_iter=20,
    )
    data_analytic = solve_ik(
        panda, targets={"panda_hand": target},
        config=IKConfig(jacobian="analytic"), initial_q=cfg0.clone(), max_iter=20,
    )

    cfg_autodiff = data_autodiff.q
    cfg_analytic = data_analytic.q
    fk_ad = panda.forward_kinematics(cfg_autodiff)
    fk_an = panda.forward_kinematics(cfg_analytic)
    err_diff = (fk_ad[hand_idx, :3] - fk_an[hand_idx, :3]).norm().item()
    assert err_diff < 0.01, f"Solutions differ by {err_diff:.4f} m"


# --- Floating-base analytic tests ---

def test_solve_ik_floating_analytic_converges(panda):
    """Floating-base IK with jacobian='analytic' converges to a reachable target."""
    cfg0 = panda._q_default
    hand_idx = panda.link_index("panda_hand")
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    target = panda.forward_kinematics(cfg0)[hand_idx].detach()

    data = solve_ik(
        panda,
        targets={"panda_hand": target},
        config=IKConfig(jacobian="analytic"),
        initial_base_pose=identity_base,
        initial_q=cfg0.clone(),
        max_iter=5,
    )
    cfg = data.q
    base_pose = data.base_pose
    fk = panda.forward_kinematics(cfg, base_pose=base_pose)
    pos_err = (fk[hand_idx, :3] - target[:3]).norm().item()
    assert pos_err < 0.05, f"Floating analytic pos_err={pos_err:.4f}"


def test_solve_ik_floating_analytic_matches_autodiff(panda):
    """Floating analytic and PyPose LM give solutions within 2 cm."""
    cfg0 = panda._q_default
    hand_idx = panda.link_index("panda_hand")
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    target = panda.forward_kinematics(cfg0)[hand_idx].detach().clone()
    target[0] += 0.05

    data_pp = solve_ik(
        panda, targets={"panda_hand": target},
        config=IKConfig(jacobian="autodiff"),
        initial_base_pose=identity_base.clone(), initial_q=cfg0.clone(), max_iter=20,
    )
    data_an = solve_ik(
        panda, targets={"panda_hand": target},
        config=IKConfig(jacobian="analytic"),
        initial_base_pose=identity_base.clone(), initial_q=cfg0.clone(), max_iter=20,
    )

    base_pp, cfg_pp = data_pp.base_pose, data_pp.q
    base_an, cfg_an = data_an.base_pose, data_an.q
    fk_pp = panda.forward_kinematics(cfg_pp, base_pose=base_pp)
    fk_an = panda.forward_kinematics(cfg_an, base_pose=base_an)
    diff = (fk_pp[hand_idx, :3] - fk_an[hand_idx, :3]).norm().item()
    assert diff < 0.02, f"Floating solutions differ by {diff:.4f} m"
