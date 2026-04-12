"""Tests for lie/tangents.py — hat/vee operators and Jacobians.

Key property checked: Jr(xi) satisfies the finite-difference definition
    log(exp(-xi) * exp(xi + eps*ej)) / eps ≈ Jr(xi)[:, j]
"""
import math
import torch
import pytest
from better_robot.lie import se3 as _se3, so3 as _so3
from better_robot.lie import tangents


# ──────────────────────────── hat / vee ──────────────────────────────────

def test_hat_so3_antisymmetric():
    w = torch.randn(3)
    W = tangents.hat_so3(w)
    assert W.shape == (3, 3)
    assert torch.allclose(W, -W.T, atol=1e-7)


def test_hat_vee_so3_roundtrip():
    w = torch.randn(3)
    assert torch.allclose(tangents.vee_so3(tangents.hat_so3(w)), w, atol=1e-7)


def test_hat_vee_so3_batch():
    w = torch.randn(5, 3)
    assert torch.allclose(tangents.vee_so3(tangents.hat_so3(w)), w, atol=1e-7)


def test_hat_se3_structure():
    """hat_se3 top-left 3x3 should be skew-sym, top-right column is v."""
    xi = torch.randn(6)
    X = tangents.hat_se3(xi)
    assert X.shape == (4, 4)
    W = X[:3, :3]
    assert torch.allclose(W, -W.T, atol=1e-7)
    assert torch.allclose(X[:3, 3], xi[:3], atol=1e-7)  # translation column


def test_hat_vee_se3_roundtrip():
    xi = torch.randn(6)
    assert torch.allclose(tangents.vee_se3(tangents.hat_se3(xi)), xi, atol=1e-7)


def test_hat_vee_se3_batch():
    xi = torch.randn(4, 6)
    assert torch.allclose(tangents.vee_se3(tangents.hat_se3(xi)), xi, atol=1e-7)


# ──────────────────────── SO3 Jacobians ──────────────────────────────────


def _fd_jr_so3(omega, eps=1e-5):
    """Finite-difference right Jacobian of SO3 exp."""
    n = 3
    # Promote to float64 to avoid float32 round-off error (≈1e-7/eps ≈ 0.01 at eps=1e-5)
    omega_d = omega.double()
    Jr_d = torch.zeros(n, n, dtype=torch.float64)
    q0 = _so3.exp(omega_d)
    q0_inv = _so3.inverse(q0)
    for j in range(n):
        delta = torch.zeros(n, dtype=torch.float64)
        delta[j] = eps
        q1 = _so3.exp(omega_d + delta)
        diff_q = _so3.compose(q0_inv, q1)
        Jr_d[:, j] = _so3.log(_so3.normalize(diff_q)) / eps
    return Jr_d.to(dtype=omega.dtype)


@pytest.mark.parametrize("scale", [0.3, 1.0, 2.5])
def test_right_jacobian_so3_fd(scale):
    torch.manual_seed(0)
    omega = torch.randn(3) * scale
    Jr_analytic = tangents.right_jacobian_so3(omega)
    Jr_fd = _fd_jr_so3(omega)
    assert torch.allclose(Jr_analytic, Jr_fd, atol=1e-4), \
        f"Jr_so3 mismatch at scale={scale}:\n{Jr_analytic}\nvs FD:\n{Jr_fd}"


def test_right_jacobian_so3_small_angle():
    """Near zero: Jr ≈ I - 1/2 * hat(omega)."""
    omega = torch.randn(3) * 1e-3
    Jr = tangents.right_jacobian_so3(omega)
    Jr_approx = torch.eye(3) - 0.5 * tangents.hat_so3(omega)
    assert torch.allclose(Jr, Jr_approx, atol=1e-5)


def test_jr_jrinv_so3_product():
    """Jr(omega) @ Jr_inv(omega) == I."""
    omega = torch.randn(3) * 0.8
    Jr = tangents.right_jacobian_so3(omega)
    Jr_inv = tangents.right_jacobian_inv_so3(omega)
    assert torch.allclose(Jr @ Jr_inv, torch.eye(3), atol=1e-5)


def test_left_right_so3_relation():
    """Jl(phi) == Jr(-phi)."""
    omega = torch.randn(3) * 0.5
    Jl = tangents.left_jacobian_so3(omega)
    Jr_neg = tangents.right_jacobian_so3(-omega)
    assert torch.allclose(Jl, Jr_neg, atol=1e-7)


# ──────────────────────── SE3 Jacobians ──────────────────────────────────


def _fd_jr_se3(xi, eps=1e-5):
    """Finite-difference right Jacobian of SE3 exp."""
    n = 6
    # Promote to float64 to avoid float32 round-off error
    xi_d = xi.double()
    Jr_d = torch.zeros(n, n, dtype=torch.float64)
    T0 = _se3.exp(xi_d)
    T0_inv = _se3.inverse(T0)
    for j in range(n):
        delta = torch.zeros(n, dtype=torch.float64)
        delta[j] = eps
        T1 = _se3.exp(xi_d + delta)
        diff = _se3.compose(T0_inv, T1)
        Jr_d[:, j] = _se3.log(diff) / eps
    return Jr_d.to(dtype=xi.dtype)


@pytest.mark.parametrize("scale", [0.1, 0.5, 1.5])
def test_right_jacobian_se3_fd(scale):
    torch.manual_seed(1)
    xi = torch.randn(6) * scale
    Jr_analytic = tangents.right_jacobian_se3(xi)
    Jr_fd = _fd_jr_se3(xi)
    assert torch.allclose(Jr_analytic, Jr_fd, atol=2e-4), \
        f"Jr_se3 mismatch at scale={scale}:\nmax err={(Jr_analytic - Jr_fd).abs().max():.6f}"


def test_jr_jrinv_se3_product():
    """Jr(xi) @ Jr_inv(xi) == I."""
    xi = torch.randn(6) * 0.4
    Jr = tangents.right_jacobian_se3(xi)
    Jr_inv = tangents.right_jacobian_inv_se3(xi)
    assert torch.allclose(Jr @ Jr_inv, torch.eye(6), atol=1e-5)
