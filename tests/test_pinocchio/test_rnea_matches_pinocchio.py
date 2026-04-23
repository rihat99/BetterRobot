"""RNEA parity — BetterRobot vs Pinocchio."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot.dynamics import bias_forces as br_bias_forces
from better_robot.dynamics import compute_coriolis_matrix as br_compute_coriolis_matrix

from .conftest import sample_panda_q

pin = pytest.importorskip("pinocchio")

# URDF parser round-trips inertias through np.float32 (io/parsers/urdf.py:152–163).
# Every RNEA accumulates several mass/com/inertia products per joint, so the
# effective floor is ~1e-5 even in fp64. Matches the FK parity tolerance policy.
_RTOL = 1e-5


# ─────────────────────────── helpers ────────────────────────────────

def _random_v(nv: int, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.rand(nv, generator=rng, dtype=torch.float64) * 2.0 - 1.0


def _random_matched_fext(br_model, pin_model, seed: int) -> tuple[torch.Tensor, list]:
    """Build matching ``fext`` vectors for BetterRobot and Pinocchio.

    BetterRobot keeps every URDF joint (including fixed ones) as a Model joint
    (njoints=14 for Panda); Pinocchio collapses fixed joints into the parent
    body (njoints=10). To compare apples to apples, we only place wrenches on
    joints whose name exists in both models. BR-side fixed joints get zero
    wrench, and Pinocchio's matching vector is sized to ``pin_model.njoints``.
    """
    rng = torch.Generator().manual_seed(seed)
    br_fext = torch.zeros(br_model.njoints, 6, dtype=torch.float64)
    pin_fext = pin.StdVec_Force()
    for _ in range(pin_model.njoints):
        pin_fext.append(pin.Force.Zero())

    for br_jid, name in enumerate(br_model.joint_names):
        if name == "universe" or not pin_model.existJointName(name):
            continue
        wrench = (torch.rand(6, generator=rng, dtype=torch.float64) * 2.0 - 1.0)
        br_fext[br_jid] = wrench
        pin_jid = pin_model.getJointId(name)
        w = wrench.numpy()
        pin_fext[pin_jid] = pin.Force(w[:3], w[3:])

    return br_fext, pin_fext


# ─────────────────────────── tests ──────────────────────────────────

@pytest.mark.parametrize("i", range(4))
def test_rnea_gravity_matches_pinocchio(panda_both, i):
    br_model, pin_model, pin_data, _ = panda_both
    q = sample_panda_q(n=4)[i]
    zeros_v = torch.zeros(br_model.nv, dtype=torch.float64)

    data = br_model.create_data()
    tau_br = br.rnea(br_model, data, q, zeros_v, zeros_v).detach().cpu().numpy()
    tau_pin = np.asarray(
        pin.computeGeneralizedGravity(pin_model, pin_data, q.detach().cpu().numpy())
    )
    np.testing.assert_allclose(tau_br, tau_pin, atol=1e-5, rtol=_RTOL)


@pytest.mark.parametrize("i", range(4))
def test_rnea_bias_matches_pinocchio(panda_both, i):
    """a = 0 → τ = non-linear effects (Coriolis + gravity)."""
    br_model, pin_model, pin_data, _ = panda_both
    q = sample_panda_q(n=4)[i]
    v = _random_v(br_model.nv, seed=100 + i)

    data = br_model.create_data()
    tau_br = br_bias_forces(br_model, data, q, v).detach().cpu().numpy()
    tau_pin = np.asarray(
        pin.nonLinearEffects(pin_model, pin_data, q.detach().cpu().numpy(), v.detach().cpu().numpy())
    )
    np.testing.assert_allclose(tau_br, tau_pin, atol=1e-5, rtol=_RTOL)


@pytest.mark.parametrize("i", range(4))
def test_rnea_full_matches_pinocchio(panda_both, i):
    """Random q, v, a — full τ = M·a + b + g."""
    br_model, pin_model, pin_data, _ = panda_both
    q = sample_panda_q(n=4)[i]
    v = _random_v(br_model.nv, seed=200 + i)
    a = _random_v(br_model.nv, seed=300 + i)

    data = br_model.create_data()
    tau_br = br.rnea(br_model, data, q, v, a).detach().cpu().numpy()
    tau_pin = np.asarray(
        pin.rnea(pin_model, pin_data,
                 q.detach().cpu().numpy(),
                 v.detach().cpu().numpy(),
                 a.detach().cpu().numpy())
    )
    np.testing.assert_allclose(tau_br, tau_pin, atol=1e-5, rtol=_RTOL)


@pytest.mark.parametrize("i", range(4))
def test_rnea_with_fext_matches_pinocchio(panda_both, i):
    """Random fext in local joint frames."""
    br_model, pin_model, pin_data, _ = panda_both
    q = sample_panda_q(n=4)[i]
    v = _random_v(br_model.nv, seed=400 + i)
    a = _random_v(br_model.nv, seed=500 + i)
    br_fext, fext_pin = _random_matched_fext(br_model, pin_model, seed=600 + i)

    data = br_model.create_data()
    tau_br = br.rnea(br_model, data, q, v, a, fext=br_fext).detach().cpu().numpy()

    tau_pin = np.asarray(
        pin.rnea(pin_model, pin_data,
                 q.detach().cpu().numpy(),
                 v.detach().cpu().numpy(),
                 a.detach().cpu().numpy(),
                 fext_pin)
    )
    np.testing.assert_allclose(tau_br, tau_pin, atol=1e-5, rtol=_RTOL)


def test_rnea_bias_forces_matches_rnea_zero_accel(panda_both):
    """Internal consistency: bias_forces(q, v) == rnea(q, v, 0)."""
    br_model, _, _, _ = panda_both
    q = sample_panda_q(n=1)[0]
    v = _random_v(br_model.nv, seed=42)
    zeros_v = torch.zeros_like(v)

    d1 = br_model.create_data()
    d2 = br_model.create_data()
    tau_bias = br_bias_forces(br_model, d1, q, v)
    tau_rnea = br.rnea(br_model, d2, q, v, zeros_v)

    assert torch.allclose(tau_bias, tau_rnea, atol=1e-12)
    assert d1.bias_forces is not None
    assert torch.allclose(d1.bias_forces, tau_bias, atol=0)


def test_rnea_populates_data_fields(panda_both):
    br_model, _, _, _ = panda_both
    q = sample_panda_q(n=1)[0]
    v = _random_v(br_model.nv, seed=1)
    a = _random_v(br_model.nv, seed=2)

    data = br_model.create_data()
    br.rnea(br_model, data, q, v, a)

    assert data.tau is not None and data.tau.shape == (br_model.nv,)
    assert data.joint_velocity_local is not None
    assert data.joint_velocity_local.shape == (br_model.njoints, 6)
    assert data.joint_acceleration_local is not None
    assert data.joint_acceleration_local.shape == (br_model.njoints, 6)
    assert data.joint_forces is not None
    assert data.joint_forces.shape == (br_model.njoints, 6)
    assert data.joint_pose_world is not None
    assert data.joint_pose_local is not None


def test_rnea_batched(panda_both):
    """Batched call agrees with single-sample calls."""
    br_model, _, _, _ = panda_both
    qs = sample_panda_q(n=3)                              # (3, nq)
    vs = torch.stack([_random_v(br_model.nv, s) for s in (10, 11, 12)])
    as_ = torch.stack([_random_v(br_model.nv, s) for s in (20, 21, 22)])

    data_batch = br_model.create_data(batch_shape=(3,))
    tau_batch = br.rnea(br_model, data_batch, qs, vs, as_)

    for k in range(3):
        data_k = br_model.create_data()
        tau_k = br.rnea(br_model, data_k, qs[k], vs[k], as_[k])
        assert torch.allclose(tau_batch[k], tau_k, atol=1e-12), f"mismatch at batch {k}"


def test_rnea_autograd_runs(panda_both):
    """Smoke: backward through RNEA closes without inf/nan."""
    br_model, _, _, _ = panda_both
    q = sample_panda_q(n=1)[0].clone().requires_grad_(True)
    v = _random_v(br_model.nv, seed=3).requires_grad_(True)
    a = _random_v(br_model.nv, seed=4).requires_grad_(True)

    data = br_model.create_data()
    tau = br.rnea(br_model, data, q, v, a)
    loss = tau.pow(2).sum()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_compute_coriolis_matrix_still_raises(panda_both):
    br_model, _, _, _ = panda_both
    q = sample_panda_q(n=1)[0]
    v = _random_v(br_model.nv, seed=5)
    data = br_model.create_data()
    with pytest.raises(NotImplementedError):
        br_compute_coriolis_matrix(br_model, data, q, v)
