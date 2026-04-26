"""Edge-case θ values for the pure-PyTorch backend.

* ``θ = 0``         — must use Taylor expansion, finite gradients.
* ``θ = π/2``       — generic mid-range value.
* ``θ ≈ π − 1e-6``  — close to the antipodal singularity in ``Log``.

Every value is checked on both SE3 and SO3.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.lie import _torch_native_backend as tn


@pytest.mark.parametrize("theta", [0.0, math.pi / 2, math.pi - 1e-6])
def test_so3_log_exp_round_trip(theta) -> None:
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    omega_in = theta * axis
    q = tn.so3_exp(omega_in)
    omega_out = tn.so3_log(q)
    torch.testing.assert_close(omega_out, omega_in, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("theta", [0.0, math.pi / 2, math.pi - 1e-6])
def test_se3_log_exp_round_trip(theta) -> None:
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    v_lin = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    omega = theta * axis
    xi_in = torch.cat([v_lin, omega])
    T = tn.se3_exp(xi_in)
    xi_out = tn.se3_log(T)
    torch.testing.assert_close(xi_out, xi_in, atol=1e-9, rtol=1e-9)


def test_so3_exp_at_zero_is_identity() -> None:
    omega = torch.zeros(3, dtype=torch.float64)
    q = tn.so3_exp(omega)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(q, expected, atol=1e-15, rtol=1e-15)


def test_se3_exp_at_zero_is_identity() -> None:
    xi = torch.zeros(6, dtype=torch.float64)
    T = tn.se3_exp(xi)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(T, expected, atol=1e-15, rtol=1e-15)


def test_se3_log_finite_at_theta_zero() -> None:
    """``∇log`` near θ=0 stays finite."""
    xi = (torch.randn(6, dtype=torch.float64) * 1e-7).requires_grad_(True)
    T = tn.se3_exp(xi)
    xi_back = tn.se3_log(T)
    loss = (xi_back ** 2).sum()
    loss.backward()
    assert xi.grad is not None
    assert torch.isfinite(xi.grad).all()
