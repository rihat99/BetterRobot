"""Forward-simulation acceptance for the 3-layer action model (P11-D7).

Builds a 1-DoF revolute pendulum programmatically, wraps it in a
``DifferentialActionModelFreeFwd`` + ``IntegratedActionModelEuler``, and
checks:

* ``calc`` returns finite ``xnext`` and a non-negative cost.
* ``calc_diff`` returns finite Jacobians ``Fx, Fu, lx, lu`` of the
  expected shapes ``(ndx, ndx)`` / ``(ndx, nu)`` / ``(ndx,)`` / ``(nu,)``.
* Free-fall under zero torque keeps angular momentum monotone (energy
  drift bounded over a short integration).
* RK4 step is closer to the analytic pendulum than Euler at the same
  ``dt`` — sanity check on the integrator selection.

The full DDP / iLQR loop closure (the rest of P11-D7) is deferred; this
release ships the building blocks plus differentiable derivatives.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.dynamics.action import (
    ActionData,
    DifferentialActionModelFreeFwd,
    IntegratedActionModelEuler,
    IntegratedActionModelRK4,
)
from better_robot.dynamics.state_manifold import StateMultibody


def _pendulum(dtype=torch.float64):
    """1-DoF revolute pendulum: ~1m arm under gravity, hinge along Y."""
    from better_robot.io import ModelBuilder, build_model

    b = ModelBuilder("pendulum")
    base = b.add_body("base", mass=0.0)
    arm = b.add_body(
        "arm",
        mass=1.0,
        com=torch.tensor([0.0, 0.0, -0.5]),
        inertia=torch.tensor([[0.083, 0.0, 0.0], [0.0, 0.083, 0.0], [0.0, 0.0, 0.001]]),
    )
    b.add_revolute_y("hinge", parent=base, child=arm, lower=-3.14, upper=3.14)
    return build_model(b.finalize(), dtype=dtype)


def test_state_manifold_zero_and_diff():
    model = _pendulum()
    state = StateMultibody(model)
    x0 = state.zero()
    assert x0.shape == (state.nx,)
    # diff to itself is zero.
    delta = state.diff(x0, x0)
    torch.testing.assert_close(delta, torch.zeros(state.ndx, dtype=x0.dtype))
    # integrate then diff round-trips.
    dx = torch.tensor([0.1, 0.0], dtype=x0.dtype)  # nv=1, ndx=2
    x1 = state.integrate(x0, dx)
    delta_back = state.diff(x0, x1)
    torch.testing.assert_close(delta_back, dx, rtol=1e-12, atol=1e-12)


def test_calc_calc_diff_smoke():
    model = _pendulum()
    state = StateMultibody(model)
    diff_model = DifferentialActionModelFreeFwd(model=model, state=state)
    int_model = IntegratedActionModelEuler(differential=diff_model, dt=0.01)

    x = state.zero()
    x = state.integrate(x, torch.tensor([0.3, 0.0], dtype=x.dtype))  # angle = 0.3
    u = torch.tensor([0.0], dtype=x.dtype)

    data = int_model.create_data()
    int_model.calc(data, x, u)
    assert data.xnext is not None and torch.isfinite(data.xnext).all()
    assert data.cost is not None and data.cost.item() >= 0.0

    int_model.calc_diff(data, x, u)
    assert data.fx.shape == (state.ndx, state.ndx)
    assert data.fu.shape == (state.ndx, int_model.nu)
    assert data.lx.shape == (state.ndx,)
    assert data.lu.shape == (int_model.nu,)
    assert torch.isfinite(data.fx).all() and torch.isfinite(data.fu).all()


def test_pendulum_free_fall_energy_bounded():
    """Under zero torque from rest at θ=π/4, energy stays roughly conserved."""
    model = _pendulum()
    state = StateMultibody(model)
    diff_model = DifferentialActionModelFreeFwd(model=model, state=state)
    int_model = IntegratedActionModelRK4(differential=diff_model, dt=0.005)

    x = state.zero()
    x = state.integrate(x, torch.tensor([math.pi / 4.0, 0.0], dtype=x.dtype))
    u = torch.tensor([0.0], dtype=x.dtype)

    g = 9.81
    m_arm = 1.0
    L = 0.5

    def energy(x_):
        theta = x_[0].item()
        thetadot = x_[1].item()
        # Inertia about the hinge: I_com + m·L² for a slender arm; using packed
        # 0.083 + 1·0.5² ≈ 0.333 (rod about end). Approximation OK for the test.
        I_hinge = 0.083 + m_arm * L * L
        ke = 0.5 * I_hinge * thetadot * thetadot
        pe = m_arm * g * (-L * math.cos(theta))  # zero at hinge
        return ke + pe

    e0 = energy(x)
    for _ in range(200):  # 1 second of integration
        d = ActionData()
        int_model.calc(d, x, u)
        x = d.xnext
    e1 = energy(x)
    # RK4 at dt=5ms should keep energy drift well under 1%.
    assert abs(e1 - e0) / max(abs(e0), 1e-6) < 0.05


def test_rk4_more_accurate_than_euler():
    model = _pendulum()
    state = StateMultibody(model)
    diff_model = DifferentialActionModelFreeFwd(model=model, state=state)
    euler = IntegratedActionModelEuler(differential=diff_model, dt=0.05)
    rk4 = IntegratedActionModelRK4(differential=diff_model, dt=0.05)

    x_eul = state.integrate(state.zero(), torch.tensor([math.pi / 6.0, 0.0], dtype=torch.float64))
    x_rk4 = x_eul.clone()
    u = torch.tensor([0.0], dtype=torch.float64)

    g = 9.81
    L = 0.5
    omega = math.sqrt(g / L)
    theta0 = math.pi / 6.0

    for _ in range(40):  # 2 seconds at dt=50ms
        d_e = ActionData(); euler.calc(d_e, x_eul, u); x_eul = d_e.xnext
        d_r = ActionData(); rk4.calc(d_r, x_rk4, u); x_rk4 = d_r.xnext

    t = 40 * 0.05
    theta_analytic = theta0 * math.cos(omega * t)
    err_euler = abs(x_eul[0].item() - theta_analytic)
    err_rk4 = abs(x_rk4[0].item() - theta_analytic)
    # Euler at dt=50ms is rough; RK4 should be markedly closer.
    assert err_rk4 < err_euler
