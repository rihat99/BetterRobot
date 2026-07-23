"""RNEA parity with advanced joint types — spherical and free-flyer.

* **Free-flyer**: G1 humanoid via ``robot_descriptions`` with ``free_flyer=True``.
  Compares against Pinocchio's ``buildModelFromUrdf(..., JointModelFreeFlyer())``.
* **Spherical**: a 2-body programmatic model built through
  :class:`ModelBuilder`, compared against a hand-assembled ``pin.Model``.

Since BetterRobot switched to DFS topological ordering (matching Pinocchio), the
``q``/``v``/``tau`` indexings align 1-to-1 with Pinocchio for URDF-loaded models,
so comparisons are direct tensor operations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br

from .conftest import _build_spherical_chain

pin = pytest.importorskip("pinocchio")


# ───────────────────────── free-flyer (G1) ─────────────────────────


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rnea_free_flyer_g1_matches_pinocchio(g1_both, seed):
    br_m, pin_m, pin_d = g1_both

    # Random q with valid free-flyer quat
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

    tau_br = br.rnea(br_m, q, v, a).detach().cpu().numpy()
    tau_pin = np.asarray(pin.rnea(pin_m, pin_d, q.numpy(), v.numpy(), a.numpy()))

    # Looser than Panda's 1e-5 because G1 has deeper chains → more fp32-URDF
    # error compounds through the parallel-axis shifts.
    np.testing.assert_allclose(tau_br, tau_pin, atol=1e-3)


def test_rnea_free_flyer_base_wrench_matches_pinocchio(g1_both):
    """The base wrench (first 6 components) is always at v-index 0 in both."""
    br_m, pin_m, pin_d = g1_both
    q = torch.zeros(br_m.nq, dtype=torch.float64)
    q[6] = 1.0  # identity quat
    zero = torch.zeros(br_m.nv, dtype=torch.float64)
    tau_br = br.rnea(br_m, q, zero, zero).detach().cpu().numpy()
    tau_pin = np.asarray(pin.rnea(pin_m, pin_d, q.numpy(), zero.numpy(), zero.numpy()))
    np.testing.assert_allclose(tau_br[:6], tau_pin[:6], atol=1e-5)


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
    tau_br = br.rnea(br_m, q, v, a).detach().cpu().numpy()
    tau_pin = np.asarray(pin.rnea(pin_m, pin_d, q.numpy(), v.numpy(), a.numpy()))
    # No URDF fp32 round-trip here — programmatic, fp64 end-to-end.
    np.testing.assert_allclose(tau_br, tau_pin, atol=1e-10)


def test_rnea_spherical_gravity_only():
    br_m, pin_m, pin_d = _build_spherical_chain()
    q = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)  # identity quat + θ=0
    z = torch.zeros(4, dtype=torch.float64)
    tau_br = br.rnea(br_m, q, z, z).detach().cpu().numpy()
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
    assert br_m.joint_models[1].kind == "fixed"  # synthetic base
    assert br_m.joint_models[2].kind == "spherical"
    assert br_m.joint_models[3].kind == "revolute_rz"
    q = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    v = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    a = torch.tensor([0.05, -0.1, 0.2, 0.3], dtype=torch.float64)
    data = br_m.create_data()
    tau = br.rnea(br_m, q, v, a, data=data)
    assert tau.shape == (4,)
    assert data.joint_velocity_local.shape == (br_m.njoints, 6)
    assert data.joint_acceleration_local.shape == (br_m.njoints, 6)
    assert data.joint_forces.shape == (br_m.njoints, 6)
