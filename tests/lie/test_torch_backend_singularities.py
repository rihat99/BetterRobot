"""Edge-case θ values for the pure-PyTorch backend.

* ``θ = 0``         — must use Taylor expansion, finite gradients.
* ``θ = π/2``       — generic mid-range value.
* ``θ ≈ π − 1e-6``  — close to the antipodal singularity in ``Log``.

Every value is checked on both SE3 and SO3.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from better_robot.lie import se3, so3
from better_robot.lie.so3 import _taylor_theta2


@pytest.mark.parametrize("theta", [0.0, math.pi / 2, math.pi - 1e-6])
def test_so3_log_exp_round_trip(theta) -> None:
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    omega_in = theta * axis
    q = so3.exp(omega_in)
    omega_out = so3.log(q)
    torch.testing.assert_close(omega_out, omega_in, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("theta", [0.0, math.pi / 2, math.pi - 1e-6])
def test_se3_log_exp_round_trip(theta) -> None:
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    v_lin = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    omega = theta * axis
    xi_in = torch.cat([v_lin, omega])
    T = se3.exp(xi_in)
    xi_out = se3.log(T)
    torch.testing.assert_close(xi_out, xi_in, atol=1e-9, rtol=1e-9)


def test_so3_exp_at_zero_is_identity() -> None:
    omega = torch.zeros(3, dtype=torch.float64)
    q = so3.exp(omega)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(q, expected, atol=1e-15, rtol=1e-15)


def test_se3_exp_at_zero_is_identity() -> None:
    xi = torch.zeros(6, dtype=torch.float64)
    T = se3.exp(xi)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(T, expected, atol=1e-15, rtol=1e-15)


def test_se3_log_finite_at_theta_zero() -> None:
    """``∇log`` near θ=0 stays finite."""
    xi = (torch.randn(6, dtype=torch.float64) * 1e-7).requires_grad_(True)
    T = se3.exp(xi)
    xi_back = se3.log(T)
    loss = (xi_back**2).sum()
    loss.backward()
    assert xi.grad is not None
    assert torch.isfinite(xi.grad).all()


def _singular_input(name: str, theta: float) -> torch.Tensor:
    """Build a direct fp64 input at rotational distance ``theta``."""
    omega = torch.tensor([theta, 0.0, 0.0], dtype=torch.float64)
    if name == "so3_exp":
        return omega
    if name == "so3_log":
        return so3.exp(omega).detach()

    xi = torch.cat([torch.zeros(3, dtype=torch.float64), omega])
    if name == "se3_exp":
        return xi
    if name == "se3_log":
        return se3.exp(xi).detach()
    raise AssertionError(f"unknown Lie operation: {name}")


@pytest.mark.parametrize(
    ("name", "op"),
    [
        ("so3_exp", so3.exp),
        ("so3_log", so3.log),
        ("se3_exp", se3.exp),
        ("se3_log", se3.log),
    ],
)
@pytest.mark.parametrize("theta", [0.0, 1e-9], ids=["identity", "theta_1e-9"])
def test_lie_singularities_first_and_second_order_gradients(name, op, theta) -> None:
    """Safe-where keeps fp64 first- and second-order derivatives finite."""
    value = _singular_input(name, theta).requires_grad_(True)

    assert torch.autograd.gradcheck(op, (value,), atol=1e-6, rtol=1e-5)
    assert torch.autograd.gradgradcheck(op, (value,), atol=1e-6, rtol=1e-5)

    (gradient,) = torch.autograd.grad(op(value).sum(), value)
    assert torch.isfinite(gradient).all()


def test_so3_exp_sum_gradient_at_identity_is_finite_and_correct() -> None:
    """The identity derivative is 1/2 per axis, not zero or NaN."""
    omega = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    so3.exp(omega).sum().backward()

    assert omega.grad is not None
    torch.testing.assert_close(omega.grad, torch.full_like(omega, 0.5))


def _fp32_cutoff_input(name: str, theta: float) -> torch.Tensor:
    """Build fp32 inputs whose SE3 translation exposes the V coefficients."""
    omega = torch.tensor([theta, 0.0, 0.0], dtype=torch.float32)
    if name == "so3_exp":
        return omega
    if name == "so3_log":
        return so3.exp(omega).detach()

    xi = torch.tensor([0.1, -0.2, 0.3, theta, 0.0, 0.0], dtype=torch.float32)
    if name == "se3_exp":
        return xi
    if name == "se3_log":
        return se3.exp(xi).detach()
    raise AssertionError(f"unknown Lie operation: {name}")


def test_fp32_taylor_cutoff_is_larger_than_fp64() -> None:
    assert _taylor_theta2(torch.float32) == 1e-5
    assert _taylor_theta2(torch.float64) == 1e-8


@pytest.mark.parametrize(
    ("name", "op"),
    [
        ("so3_exp", so3.exp),
        ("so3_log", so3.log),
        ("se3_exp", se3.exp),
        ("se3_log", se3.log),
    ],
)
@pytest.mark.parametrize("cutoff_ratio", [0.5, 2.0], ids=["taylor_branch", "full_branch"])
def test_fp32_gradcheck_on_both_sides_of_taylor_cutoff(name, op, cutoff_ratio) -> None:
    """Exercise both branches without finite-difference samples crossing them.

    fp32 finite differences need a larger step than fp64.  At these two
    angles, ``eps=5e-4`` remains on the selected side of the cutoff; the
    tolerances account for fp32 output quantization divided by that step.
    """
    cutoff = _taylor_theta2(torch.float32)
    theta = math.sqrt(cutoff) * cutoff_ratio
    value = _fp32_cutoff_input(name, theta).requires_grad_(True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input #0 requires gradient and is not a double precision")
        assert torch.autograd.gradcheck(
            op,
            (value,),
            eps=5e-4,
            atol=4e-3,
            rtol=2e-2,
        )
