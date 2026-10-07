"""CRBA parity — BetterRobot vs Pinocchio + structural acceptance.

Acceptance for P11-D3:

* ``M(q) @ a + bias_forces(q, v) == rnea(q, v, a)`` — internal consistency.
* ``M(q_neutral)`` is symmetric and positive-definite.
* ``M(q)`` matches ``pin.crba`` element-wise on Panda at fp64 (1e-10 rel
  tolerance; the measured gap is 1e-15).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from better_robot.dynamics.state import inverse_dynamics, mass_matrix
from better_robot.kinematics import forward_kinematics_state

from .conftest import sample_panda_q

pin = pytest.importorskip("pinocchio")

_RTOL = 1e-10


def test_crba_matches_pinocchio(panda_both):
    """``M(q)`` matches Pinocchio's ``crba`` on Panda."""
    br_model, pin_model, pin_data, _ = panda_both
    qs = sample_panda_q(8, seed=0)
    for i in range(qs.shape[0]):
        q = qs[i]
        M_br = mass_matrix(br_model, forward_kinematics_state(br_model, q)).detach().cpu().numpy()
        M_pin = np.asarray(pin.crba(pin_model, pin_data, q.numpy()))
        # Pinocchio fills only the upper triangle by default; mirror it.
        M_pin_full = np.triu(M_pin) + np.triu(M_pin, k=1).T
        np.testing.assert_allclose(M_br, M_pin_full, rtol=_RTOL, atol=_RTOL)


def test_crba_consistent_with_rnea(panda_both):
    """``M(q) @ a + b(q, v) == rnea(q, v, a)`` to fp64 noise."""
    br_model, _, _, _ = panda_both
    qs = sample_panda_q(4, seed=1)
    rng = torch.Generator().manual_seed(2)
    for i in range(qs.shape[0]):
        q = qs[i]
        v = (torch.rand(br_model.nv, generator=rng, dtype=torch.float64) - 0.5) * 2.0
        a = (torch.rand(br_model.nv, generator=rng, dtype=torch.float64) - 0.5) * 2.0

        M = mass_matrix(br_model, forward_kinematics_state(br_model, q))

        b = inverse_dynamics(br_model, forward_kinematics_state(br_model, q), v, torch.zeros_like(v))

        tau = inverse_dynamics(br_model, forward_kinematics_state(br_model, q), v, a)

        lhs = (M @ a.unsqueeze(-1)).squeeze(-1) + b
        torch.testing.assert_close(lhs, tau, rtol=1e-10, atol=1e-10)


def test_crba_neutral_is_spd(panda_both):
    """``M(q_neutral)`` is symmetric positive-definite."""
    br_model, _, _, _ = panda_both
    q = br_model.q_neutral.clamp(br_model.lower_pos_limit, br_model.upper_pos_limit)
    M = mass_matrix(br_model, forward_kinematics_state(br_model, q)).detach().cpu().double()
    # Symmetric:
    torch.testing.assert_close(M, M.transpose(-1, -2), rtol=1e-12, atol=1e-12)
    # Positive-definite (Cholesky succeeds):
    torch.linalg.cholesky(M)


def test_crba_batched(panda_both):
    """Batched CRBA matches a per-sample loop."""
    br_model, _, _, _ = panda_both
    qs = sample_panda_q(4, seed=3)  # (4, nq)
    M_batch = mass_matrix(br_model, forward_kinematics_state(br_model, qs))
    for k in range(qs.shape[0]):
        M_k = mass_matrix(br_model, forward_kinematics_state(br_model, qs[k]))
        torch.testing.assert_close(M_batch[k], M_k, rtol=1e-12, atol=1e-12)
