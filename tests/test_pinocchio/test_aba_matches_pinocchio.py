"""ABA parity — BetterRobot vs Pinocchio + the inverse-of-RNEA identity.

Acceptance for P11-D4:

* ``aba(q, v, rnea(q, v, a)) == a`` to fp64 ulp on Panda (fixed-base) and
  G1 (free-flyer).
* ``aba(q, v, τ)`` matches ``pin.aba`` on Panda. The looser 1e-4 atol is
  the URDF inertia round-trip noise (see comment in
  ``test_rnea_matches_pinocchio.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot.dynamics import aba as br_aba
from better_robot.dynamics import rnea as br_rnea

from .conftest import sample_panda_q

pin = pytest.importorskip("pinocchio")

_RTOL = 1e-4
_ATOL = 1e-4


def test_aba_matches_pinocchio(panda_both):
    br_model, pin_model, pin_data, _ = panda_both
    qs = sample_panda_q(8, seed=0)
    rng = torch.Generator().manual_seed(1)
    for i in range(qs.shape[0]):
        q = qs[i]
        v = (torch.rand(br_model.nv, generator=rng, dtype=torch.float64) - 0.5)
        tau = (torch.rand(br_model.nv, generator=rng, dtype=torch.float64) - 0.5)
        ddq_br = br_aba(br_model, br_model.create_data(), q, v, tau).detach().cpu().numpy()
        ddq_pin = np.asarray(pin.aba(pin_model, pin_data, q.numpy(), v.numpy(), tau.numpy()))
        np.testing.assert_allclose(ddq_br, ddq_pin, rtol=_RTOL, atol=_ATOL)


def test_aba_inverse_of_rnea_panda(panda_both):
    """``aba(q, v, rnea(q, v, a)) == a`` to fp64 ulp."""
    br_model, _, _, _ = panda_both
    qs = sample_panda_q(8, seed=2)
    rng = torch.Generator().manual_seed(3)
    for i in range(qs.shape[0]):
        q = qs[i]
        v = (torch.rand(br_model.nv, generator=rng, dtype=torch.float64) - 0.5) * 2.0
        a = (torch.rand(br_model.nv, generator=rng, dtype=torch.float64) - 0.5) * 2.0
        tau = br_rnea(br_model, br_model.create_data(), q, v, a)
        ddq = br_aba(br_model, br_model.create_data(), q, v, tau)
        torch.testing.assert_close(ddq, a, rtol=1e-10, atol=1e-10)


def test_aba_inverse_of_rnea_g1():
    """Same identity on the G1 humanoid (free-flyer base)."""
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description

    model = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
    rng = torch.Generator().manual_seed(0)
    q = model.q_neutral.clone()
    v = (torch.rand(model.nv, generator=rng, dtype=torch.float64) - 0.5) * 0.2
    a = (torch.rand(model.nv, generator=rng, dtype=torch.float64) - 0.5) * 0.2
    tau = br_rnea(model, model.create_data(), q, v, a)
    ddq = br_aba(model, model.create_data(), q, v, tau)
    torch.testing.assert_close(ddq, a, rtol=1e-9, atol=1e-9)


def test_aba_batched(panda_both):
    """Batched ABA matches a per-sample loop."""
    br_model, _, _, _ = panda_both
    qs = sample_panda_q(4, seed=4)
    rng = torch.Generator().manual_seed(5)
    vs = (torch.rand(4, br_model.nv, generator=rng, dtype=torch.float64) - 0.5)
    taus = (torch.rand(4, br_model.nv, generator=rng, dtype=torch.float64) - 0.5)

    ddq_batch = br_aba(br_model, br_model.create_data(batch_shape=(4,)), qs, vs, taus)
    for k in range(qs.shape[0]):
        ddq_k = br_aba(br_model, br_model.create_data(), qs[k], vs[k], taus[k])
        torch.testing.assert_close(ddq_batch[k], ddq_k, rtol=1e-12, atol=1e-12)
