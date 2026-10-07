"""Tests for lie/se3.py — SE3 group operations."""
import math
import torch
from better_robot.lie import se3


def test_identity_batch():
    T = se3.identity(batch_shape=(3, 2))
    assert T.shape == (3, 2, 7)
    assert torch.allclose(T[..., 6], torch.ones(3, 2))


def test_compose_inverse():
    """T * T^{-1} == identity."""
    T = se3.exp(torch.randn(6))
    T_inv = se3.inverse(T)
    result = se3.compose(T, T_inv)
    identity = se3.identity()
    assert torch.allclose(result[:3], identity[:3], atol=1e-5)
    # Quaternion: either same or negated
    assert torch.allclose(result[3:].abs(), identity[3:].abs(), atol=1e-5)


def test_exp_log_roundtrip():
    """log(exp(xi)) == xi for small xi."""
    xi = torch.randn(6) * 0.1
    T = se3.exp(xi)
    xi_back = se3.log(T)
    assert torch.allclose(xi, xi_back, atol=1e-5)


def test_act_translation():
    """Pure translation shifts the point."""
    T = torch.tensor([1., 0., 0., 0., 0., 0., 1.])  # translate by (1,0,0)
    p = torch.tensor([0., 0., 0.])
    result = se3.act(T, p)
    assert torch.allclose(result, torch.tensor([1., 0., 0.]), atol=1e-6)


def test_adjoint_inv_is_inverse():
    """Ad(T) @ Ad(T^{-1}) == I."""
    T = se3.exp(torch.randn(6))
    Ad = se3.adjoint(T)
    Ad_inv = se3.adjoint_inv(T)
    prod = Ad @ Ad_inv
    assert torch.allclose(prod, torch.eye(6), atol=1e-5)


def test_from_axis_angle_pure_rotation():
    """Pure rotation about Z by pi/2 maps (1,0,0) to (0,1,0)."""
    axis = torch.tensor([0., 0., 1.])
    angle = torch.tensor(math.pi / 2)
    T = se3.from_axis_angle(axis, angle)
    p = torch.tensor([1., 0., 0.])
    result = se3.act(T, p)
    expected = torch.tensor([0., 1., 0.])
    assert torch.allclose(result, expected, atol=1e-5)


def test_from_translation():
    """Pure translation."""
    axis = torch.tensor([1., 0., 0.])
    disp = torch.tensor(2.5)
    T = se3.from_translation(axis, disp)
    p = torch.tensor([0., 0., 0.])
    result = se3.act(T, p)
    expected = torch.tensor([2.5, 0., 0.])
    assert torch.allclose(result, expected, atol=1e-6)


def test_normalize():
    """Normalize bad quaternion → unit quaternion."""
    T = torch.tensor([0., 0., 0., 0., 0., 0., 2.])  # qw=2, not normalized
    T_norm = se3.normalize(T)
    q = T_norm[3:]
    assert torch.allclose(q.norm(), torch.tensor(1.0), atol=1e-6)
