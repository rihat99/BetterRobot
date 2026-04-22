"""Sanity tests for ``VelocityResidual`` / ``AccelerationResidual``.

Covers:
* Linearly-interpolated trajectory in config space → acceleration ≈ 0.
* Perturbed trajectory → acceleration ≠ 0.
* Analytic Jacobian matches central finite differences through
  ``model.integrate`` (the identity-right-Jacobian approximation is
  valid in the small-step regime these residuals operate in).
"""

from __future__ import annotations

import math

import pytest
import torch

import better_robot as br
from better_robot.residuals import AccelerationResidual, VelocityResidual
from better_robot.residuals.base import ResidualState


@pytest.fixture(scope="module")
def panda_model():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description
    return br.load(panda_description.URDF_PATH, dtype=torch.float64)


def _state(model, q_traj):
    data = br.forward_kinematics(model, q_traj, compute_frames=True)
    return ResidualState(model=model, data=data, variables=q_traj)


def test_acceleration_zero_on_linear_trajectory(panda_model):
    """A linear interpolation in config space has zero acceleration."""
    T = 10
    q0 = panda_model.q_neutral.double()
    q1 = q0.clone()
    q1[0] = 0.5
    alpha = torch.linspace(0, 1, T, dtype=torch.float64).unsqueeze(1)
    q_traj = q0 * (1 - alpha) + q1 * alpha

    res = AccelerationResidual(panda_model, dt=0.1)
    r = res(_state(panda_model, q_traj))
    # Machine-precision zero for a pure Euclidean linear interpolation.
    assert float(r.abs().max()) < 1e-12, f"expected ~0, got {float(r.abs().max()):.3e}"


def test_acceleration_nonzero_on_perturbed_trajectory(panda_model):
    """Random perturbation per frame → non-trivial acceleration."""
    torch.manual_seed(42)
    T = 10
    q = panda_model.q_neutral.double().unsqueeze(0).expand(T, -1).clone()
    q += torch.randn_like(q) * 0.1

    res = AccelerationResidual(panda_model, dt=0.05)
    r = res(_state(panda_model, q))
    assert float(r.norm()) > 1.0, "perturbed trajectory should have nonzero acceleration"


def test_velocity_zero_on_constant_trajectory(panda_model):
    """Constant configuration across time → zero velocity."""
    T = 6
    q_traj = panda_model.q_neutral.double().unsqueeze(0).expand(T, -1).clone()
    res = VelocityResidual(panda_model, dt=0.1)
    r = res(_state(panda_model, q_traj))
    assert float(r.abs().max()) < 1e-12


def test_acceleration_analytic_jacobian_matches_fd(panda_model):
    """Analytic Jacobian agrees with central FD through ``model.integrate``."""
    torch.manual_seed(0)
    T = 6
    dt = 0.1
    q_traj = panda_model.q_neutral.double().unsqueeze(0).expand(T, -1).clone()
    q_traj = q_traj + torch.randn_like(q_traj) * 0.02

    res = AccelerationResidual(panda_model, dt=dt)
    state = _state(panda_model, q_traj)
    J_an = res.jacobian(state)

    nv = panda_model.nv
    eps = 1e-6
    J_fd = torch.zeros_like(J_an)
    for s in range(T):
        for i in range(nv):
            dv = torch.zeros(T, nv, dtype=q_traj.dtype)
            dv[s, i] = eps
            q_p = panda_model.integrate(q_traj, dv)
            q_m = panda_model.integrate(q_traj, -dv)
            r_p = res(_state(panda_model, q_p))
            r_m = res(_state(panda_model, q_m))
            J_fd[:, s * nv + i] = (r_p - r_m) / (2 * eps)

    torch.testing.assert_close(J_an, J_fd, atol=1e-6, rtol=1e-4)


def test_velocity_analytic_jacobian_matches_fd(panda_model):
    torch.manual_seed(1)
    T = 6
    dt = 0.1
    q_traj = panda_model.q_neutral.double().unsqueeze(0).expand(T, -1).clone()
    q_traj = q_traj + torch.randn_like(q_traj) * 0.02

    res = VelocityResidual(panda_model, dt=dt)
    state = _state(panda_model, q_traj)
    J_an = res.jacobian(state)

    nv = panda_model.nv
    eps = 1e-6
    J_fd = torch.zeros_like(J_an)
    for s in range(T):
        for i in range(nv):
            dv = torch.zeros(T, nv, dtype=q_traj.dtype)
            dv[s, i] = eps
            q_p = panda_model.integrate(q_traj, dv)
            q_m = panda_model.integrate(q_traj, -dv)
            r_p = res(_state(panda_model, q_p))
            r_m = res(_state(panda_model, q_m))
            J_fd[:, s * nv + i] = (r_p - r_m) / (2 * eps)

    torch.testing.assert_close(J_an, J_fd, atol=1e-6, rtol=1e-4)


def test_autograd_through_difference(panda_model):
    """End-to-end autograd: loss via AccelerationResidual + tangent-space
    parameterisation backprops cleanly into the tangent variable."""
    T = 6
    dt = 0.1
    q_init = panda_model.q_neutral.double().unsqueeze(0).expand(T, -1).clone()
    delta_v = torch.zeros(T, panda_model.nv, dtype=torch.float64, requires_grad=True)

    q_traj = panda_model.integrate(q_init, delta_v)
    res = AccelerationResidual(panda_model, dt=dt)
    r = res(_state(panda_model, q_traj))
    loss = 0.5 * (r * r).sum()
    loss.backward()

    assert delta_v.grad is not None
    assert torch.isfinite(delta_v.grad).all()
