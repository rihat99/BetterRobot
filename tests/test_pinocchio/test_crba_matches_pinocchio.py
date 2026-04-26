"""CRBA parity — BetterRobot vs Pinocchio + structural acceptance.

Acceptance for P11-D3:

* ``M(q) @ a + bias_forces(q, v) == rnea(q, v, a)`` — internal consistency.
* ``M(q_neutral)`` is symmetric and positive-definite.
* ``M(q)`` matches ``pin.crba`` element-wise on Panda at fp64 (1e-5 rel
  tolerance, dominated by the URDF inertia round-trip through fp32 in
  ``io/parsers/urdf.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot.dynamics import bias_forces as br_bias_forces
from better_robot.dynamics import crba as br_crba
from better_robot.dynamics import rnea as br_rnea

from .conftest import sample_panda_q

pin = pytest.importorskip("pinocchio")

_RTOL = 1e-5


def test_crba_matches_pinocchio(panda_both):
    """``M(q)`` matches Pinocchio's ``crba`` on Panda."""
    br_model, pin_model, pin_data, _ = panda_both
    qs = sample_panda_q(8, seed=0)
    for i in range(qs.shape[0]):
        q = qs[i]
        data = br_model.create_data()
        M_br = br_crba(br_model, data, q).detach().cpu().numpy()
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

        d_M = br_model.create_data()
        M = br_crba(br_model, d_M, q)

        d_b = br_model.create_data()
        b = br_bias_forces(br_model, d_b, q, v)

        d_t = br_model.create_data()
        tau = br_rnea(br_model, d_t, q, v, a)

        lhs = (M @ a.unsqueeze(-1)).squeeze(-1) + b
        torch.testing.assert_close(lhs, tau, rtol=1e-10, atol=1e-10)


def test_crba_neutral_is_spd(panda_both):
    """``M(q_neutral)`` is symmetric positive-definite."""
    br_model, _, _, _ = panda_both
    q = br_model.q_neutral.clamp(br_model.lower_pos_limit, br_model.upper_pos_limit)
    data = br_model.create_data()
    M = br_crba(br_model, data, q).detach().cpu().double()
    # Symmetric:
    torch.testing.assert_close(M, M.transpose(-1, -2), rtol=1e-12, atol=1e-12)
    # Positive-definite (Cholesky succeeds):
    torch.linalg.cholesky(M)


def test_crba_batched(panda_both):
    """Batched CRBA matches a per-sample loop."""
    br_model, _, _, _ = panda_both
    qs = sample_panda_q(4, seed=3)                      # (4, nq)
    data_batch = br_model.create_data(batch_shape=(4,))
    M_batch = br_crba(br_model, data_batch, qs)
    for k in range(qs.shape[0]):
        d_k = br_model.create_data()
        M_k = br_crba(br_model, d_k, qs[k])
        torch.testing.assert_close(M_batch[k], M_k, rtol=1e-12, atol=1e-12)
