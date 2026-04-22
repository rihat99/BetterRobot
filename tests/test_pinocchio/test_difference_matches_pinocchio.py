"""``Model.difference`` and ``Model.integrate`` parity with Pinocchio."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .conftest import sample_panda_q

pin = pytest.importorskip("pinocchio")


def test_difference_matches(panda_both):
    br_model, pin_model, _, _ = panda_both
    qs_a = sample_panda_q(n=8, seed=1)
    qs_b = sample_panda_q(n=8, seed=2)

    for i in range(8):
        q0, q1 = qs_a[i], qs_b[i]
        dv_br = br_model.difference(q0, q1).detach().cpu().double().numpy()
        dv_pin = pin.difference(pin_model, q0.numpy(), q1.numpy())
        np.testing.assert_allclose(dv_br, dv_pin, atol=1e-12, rtol=1e-10)


def test_integrate_matches(panda_both):
    br_model, pin_model, _, _ = panda_both
    qs = sample_panda_q(n=8, seed=3)
    rng = torch.Generator().manual_seed(4)

    for i in range(8):
        q = qs[i]
        v = (torch.rand(br_model.nv, generator=rng, dtype=torch.float64) - 0.5) * 0.5
        q_new_br = br_model.integrate(q, v).detach().cpu().double().numpy()
        q_new_pin = pin.integrate(pin_model, q.numpy(), v.numpy())
        np.testing.assert_allclose(q_new_br, q_new_pin, atol=1e-12, rtol=1e-10)


def test_integrate_difference_roundtrip(panda_both):
    """q ⊕ (q ⊖ q') should equal q'."""
    br_model, pin_model, _, _ = panda_both
    qs_a = sample_panda_q(n=8, seed=5)
    qs_b = sample_panda_q(n=8, seed=6)

    for i in range(8):
        q0, q1 = qs_a[i], qs_b[i]
        dv = br_model.difference(q0, q1)
        q1_reconstructed = br_model.integrate(q0, dv).detach().cpu().double().numpy()
        np.testing.assert_allclose(
            q1_reconstructed, q1.numpy(), atol=1e-12, rtol=1e-10
        )
