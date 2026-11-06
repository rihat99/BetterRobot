"""RNEA parity with advanced joint types — spherical and free-flyer.

* **Free-flyer**: the tracked G1 humanoid (``tests/support/robots.py``) with ``root_joint=JointFreeFlyer()``.
  Compares against Pinocchio's ``buildModelFromUrdf(..., JointModelFreeFlyer())``.
* **Spherical**: a 2-body programmatic model built through
  :class:`ModelBuilder`, compared against a hand-assembled ``pin.Model``.

Pinocchio orders the joints depth-first while BetterRobot keeps the URDF's order, so the
G1 cases sample ``q``, ``v`` and ``a`` in Pinocchio's layout and gather them into BetterRobot's
through the ``perm_q``/``perm_v`` indices of the ``g1_both`` fixture (``Model.q_permutation``).
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest
import torch

from better_robot.dynamics import inverse_dynamics
from better_robot.kinematics import forward_kinematics
from better_robot.model.joints import JointFixed, JointRevolute, JointSpherical

from .conftest import _build_spherical_chain


# ───────────────────────── free-flyer (G1) ─────────────────────────


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rnea_free_flyer_g1_matches_pinocchio(g1_both, seed):
    br_m, pin_m, pin_d, perm_q, perm_v = g1_both

    # Random q in Pinocchio's layout, with a valid free-flyer quaternion
    rng = torch.Generator().manual_seed(seed)
    q = torch.zeros(br_m.nq, dtype=torch.float64)
    # Base translation
    q[0:3] = torch.rand(3, generator=rng, dtype=torch.float64) * 0.5 - 0.25
    # Base quaternion (normalized)
    quat = torch.randn(4, generator=rng, dtype=torch.float64)
    q[3:7] = quat / quat.norm()
    # Remaining revolute joints in a safe range
    if br_m.nq > 7:
        q[7:] = torch.rand(br_m.nq - 7, generator=rng, dtype=torch.float64) * 0.6 - 0.3

    v = torch.randn(br_m.nv, generator=rng, dtype=torch.float64) * 0.2
    a = torch.randn(br_m.nv, generator=rng, dtype=torch.float64) * 0.2

    state = forward_kinematics(br_m, q[perm_q])
    tau_br = inverse_dynamics(br_m, state, v=v[perm_v], a=a[perm_v]).detach().cpu().numpy()
    tau_pin = np.asarray(pin.rnea(pin_m, pin_d, q.numpy(), v.numpy(), a.numpy()))

    np.testing.assert_allclose(tau_br, tau_pin[perm_v], atol=1e-10, err_msg=f"G1 rnea by joint name, seed {seed}")


def test_rnea_free_flyer_base_wrench_matches_pinocchio(g1_both):
    """The base wrench (first 6 components) is always at v-index 0 in both."""
    br_m, pin_m, pin_d, _, _ = g1_both
    q = torch.zeros(br_m.nq, dtype=torch.float64)
    q[6] = 1.0  # identity quat
    zero = torch.zeros(br_m.nv, dtype=torch.float64)
    tau_br = inverse_dynamics(br_m, forward_kinematics(br_m, q), v=zero, a=zero).detach().cpu().numpy()
    tau_pin = np.asarray(pin.rnea(pin_m, pin_d, q.numpy(), zero.numpy(), zero.numpy()))
    np.testing.assert_allclose(tau_br[:6], tau_pin[:6], atol=1e-10, err_msg="G1 base wrench at rest")


# ───────────────────────── spherical joint ─────────────────────────


def _spherical_random_qva(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random (q=[quat, angle], v=[ω, ω_rz], a=[α, α_rz]) for the 2-body chain."""
    rng = torch.Generator().manual_seed(seed)
    quat = torch.randn(4, generator=rng, dtype=torch.float64)
    quat = quat / quat.norm()
    angle = torch.rand(1, generator=rng, dtype=torch.float64) * 2.0 - 1.0
    q = torch.cat([quat, angle])
    v = torch.randn(4, generator=rng, dtype=torch.float64) * 0.3
    a = torch.randn(4, generator=rng, dtype=torch.float64) * 0.3
    return q, v, a


@pytest.mark.parametrize("seed", [10, 20, 30, 40])
def test_rnea_spherical_matches_pinocchio(seed):
    br_m, pin_m, pin_d = _build_spherical_chain()
    q, v, a = _spherical_random_qva(seed)
    tau_br = inverse_dynamics(br_m, forward_kinematics(br_m, q), v=v, a=a).detach().cpu().numpy()
    tau_pin = np.asarray(pin.rnea(pin_m, pin_d, q.numpy(), v.numpy(), a.numpy()))
    # No URDF fp32 round-trip here — programmatic, fp64 end-to-end.
    np.testing.assert_allclose(tau_br, tau_pin, atol=1e-10)


def test_rnea_spherical_gravity_only():
    br_m, pin_m, pin_d = _build_spherical_chain()
    q = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)  # identity quat + θ=0
    z = torch.zeros(4, dtype=torch.float64)
    tau_br = inverse_dynamics(br_m, forward_kinematics(br_m, q), v=z, a=z).detach().cpu().numpy()
    tau_pin = np.asarray(pin.rnea(pin_m, pin_d, q.numpy(), z.numpy(), z.numpy()))
    np.testing.assert_allclose(tau_br, tau_pin, atol=1e-12)


def test_rnea_spherical_populates_tangent_fields_correctly():
    """Spherical has nv=3, revolute has nv=1 → total nv=4.

    ``ModelBuilder`` without an explicit ``world``-parent joint inserts a
    synthetic ``JointFixed`` root at model-index 1. Our spherical joint lands
    at index 2, and the revolute at index 3.
    """
    br_m, _, _ = _build_spherical_chain()
    assert br_m.nv == 4
    assert br_m.nq == 5  # 4 quat + 1 angle
    assert isinstance(br_m.joints[1], JointFixed)  # synthetic base
    assert isinstance(br_m.joints[2], JointSpherical)
    assert br_m.joints[3] == JointRevolute(axis=(0.0, 0.0, 1.0))
    q = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    v = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    a = torch.tensor([0.05, -0.1, 0.2, 0.3], dtype=torch.float64)
    tau = inverse_dynamics(br_m, forward_kinematics(br_m, q), v=v, a=a)
    assert tau.shape == (4,)
