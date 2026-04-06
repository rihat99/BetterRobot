"""Tests for analytical Jacobian functions."""

import torch
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description
import better_robot as br
from better_robot.algorithms.kinematics import compute_jacobian, limit_jacobian, rest_jacobian
from better_robot.costs import pose_residual


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return br.load_urdf(urdf)


def test_pose_jacobian_shape(panda):
    q = panda._q_default
    link_idx = panda.link_index("panda_hand")
    target = panda.forward_kinematics(q)[link_idx].detach()
    J = compute_jacobian(panda, q, link_idx, target, 1.0, 0.1)
    assert J.shape == (6, panda.joints.num_actuated_joints)


def test_pose_jacobian_finite_diff(panda):
    """Analytical Jacobian matches central finite differences."""
    eps = 1e-3  # float32: eps < 1e-4 causes catastrophic cancellation in se3_log
    q = panda._q_default.clone()
    link_idx = panda.link_index("panda_hand")
    target = panda.forward_kinematics(q)[link_idx].detach()

    J_analytic = compute_jacobian(panda, q, link_idx, target, 1.0, 0.1)

    n = len(q)
    J_fd = torch.zeros(6, n)
    for i in range(n):
        cp = q.clone(); cp[i] += eps
        cm = q.clone(); cm[i] -= eps
        rp = pose_residual(cp, panda, link_idx, target, 1.0, 0.1)
        rm = pose_residual(cm, panda, link_idx, target, 1.0, 0.1)
        J_fd[:, i] = (rp - rm) / (2 * eps)

    assert torch.allclose(J_analytic, J_fd, atol=1e-3), \
        f"Max diff: {(J_analytic - J_fd).abs().max():.5f}"


def test_limit_jacobian_shape(panda):
    q = panda._q_default
    J = limit_jacobian(q, panda)
    n = panda.joints.num_actuated_joints
    assert J.shape == (2 * n, n)


def test_limit_jacobian_inside_limits_is_zero(panda):
    """Within limits, limit_jacobian rows are all zero."""
    q = panda._q_default  # midpoint — guaranteed inside limits
    J = limit_jacobian(q, panda)
    assert torch.all(J == 0.0)


def test_limit_jacobian_violated_lower(panda):
    """Below lower limit: lower violation row has -1 on diagonal."""
    q = panda.joints.lower_limits.clone() - 0.1  # below lower limit
    J = limit_jacobian(q, panda)
    n = panda.joints.num_actuated_joints
    # Lower violation rows (top n): diagonal should be -1
    assert torch.allclose(J[:n], torch.diag(torch.full((n,), -1.0)), atol=1e-6)


def test_limit_jacobian_violated_upper(panda):
    """Above upper limit: upper violation row has +1 on diagonal."""
    q = panda.joints.upper_limits.clone() + 0.1  # above upper limit
    J = limit_jacobian(q, panda)
    n = panda.joints.num_actuated_joints
    # Upper violation rows (bottom n): diagonal should be +1
    assert torch.allclose(J[n:], torch.diag(torch.full((n,), 1.0)), atol=1e-6)


def test_rest_jacobian_is_identity(panda):
    q = panda._q_default
    rest = panda._q_default.clone()
    J = rest_jacobian(q, rest)
    n = panda.joints.num_actuated_joints
    assert J.shape == (n, n)
    assert torch.allclose(J, torch.eye(n), atol=1e-6)
