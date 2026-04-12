"""Tests for lie/so3.py — SO3 group operations."""
import math
import torch
from better_robot.lie import so3


def test_identity():
    q = so3.identity()
    assert q.shape == (4,)
    assert torch.allclose(q, torch.tensor([0., 0., 0., 1.]))


def test_compose_inverse():
    """q * q^{-1} == identity."""
    q = so3.exp(torch.randn(3) * 0.5)
    q_inv = so3.inverse(q)
    result = so3.compose(q, q_inv)
    expected = so3.identity()
    # Quaternion can be +/- identity
    assert torch.allclose(result.abs(), expected.abs(), atol=1e-5)


def test_exp_log_roundtrip():
    omega = torch.randn(3) * 0.3
    q = so3.exp(omega)
    omega_back = so3.log(q)
    assert torch.allclose(omega, omega_back, atol=1e-5)


def test_act_rotation():
    """90-degree rotation about Z."""
    axis = torch.tensor([0., 0., 1.])
    angle = torch.tensor(math.pi / 2)
    q = so3.from_axis_angle(axis, angle)
    p = torch.tensor([1., 0., 0.])
    result = so3.act(q, p)
    expected = torch.tensor([0., 1., 0.])
    assert torch.allclose(result, expected, atol=1e-5)


def test_act_identity():
    q = so3.identity()
    p = torch.randn(3)
    assert torch.allclose(so3.act(q, p), p, atol=1e-6)


def test_adjoint_is_rotation_matrix():
    """Adjoint of SO3 is the rotation matrix."""
    q = so3.exp(torch.randn(3) * 0.5)
    Ad = so3.adjoint(q)
    R = so3.to_matrix(q)
    assert torch.allclose(Ad, R, atol=1e-6)


def test_from_matrix_roundtrip():
    """from_matrix(to_matrix(q)) ≈ q."""
    q = so3.normalize(torch.randn(4))
    R = so3.to_matrix(q)
    q_back = so3.from_matrix(R)
    # Quaternion ±q represent same rotation
    assert (torch.allclose(q_back, q, atol=1e-5) or
            torch.allclose(q_back, -q, atol=1e-5))


def test_normalize():
    q = torch.tensor([0., 0., 0., 2.])
    q_norm = so3.normalize(q)
    assert torch.allclose(q_norm.norm(), torch.tensor(1.0), atol=1e-6)
