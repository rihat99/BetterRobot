"""Centroidal-dynamics acceptance tests for P11-D1 / P11-D5.

The whole-body convention here includes every body in
``model.body_inertias`` — including bodies attached via the synthetic
``root_joint`` (a fixed joint inserted by ``io.build_model``). Pinocchio
collapses fixed-joint bodies into the ``universe`` slot and excludes
that slot from ``pin.centerOfMass`` / ``pin.ccrba``, so a direct numeric
match would require fabricating a one-off Pinocchio model. We instead
test the invariants the centroidal API must satisfy under any valid
convention:

* ``h_g[:3] == total_mass · v_com`` — linear momentum invariant.
* ``h_g == A_g(q) · v`` — matrix–vector consistency.
* ``v == 0 ⇒ h_g == 0`` — kinematic invariance.

The integration test ``test_centroidal_momentum_consistency`` derives
``h_g`` independently by running RNEA at zero acceleration to extract
the per-body twists, then compares to ``ccrba``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot.dynamics import ccrba, center_of_mass
from better_robot.dynamics import compute_centroidal_map, compute_centroidal_momentum

from .conftest import sample_panda_q


def _panda(dtype=torch.float64):
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description
    return br.load(panda_description.URDF_PATH, dtype=dtype)


def test_hg_linear_equals_mass_times_vcom():
    """``h_g[:3] = m_total · v_com`` for any (q, v)."""
    model = _panda()
    qs = sample_panda_q(4, seed=0)
    rng = torch.Generator().manual_seed(1)
    total_mass = torch.tensor(
        sum(model.body_inertias[i, 0].item() for i in range(model.njoints)),
        dtype=torch.float64,
    )
    for i in range(qs.shape[0]):
        q = qs[i]
        v = (torch.rand(model.nv, generator=rng, dtype=torch.float64) - 0.5)
        data = model.create_data()
        com = center_of_mass(model, data, q, v=v)
        assert data.com_velocity is not None
        v_com = data.com_velocity
        _, h_g = ccrba(model, model.create_data(), q, v)
        torch.testing.assert_close(h_g[:3], total_mass * v_com, rtol=1e-10, atol=1e-10)


def test_ag_times_v_equals_hg():
    """``h_g = A_g(q) · v``."""
    model = _panda()
    qs = sample_panda_q(4, seed=2)
    rng = torch.Generator().manual_seed(3)
    for i in range(qs.shape[0]):
        q = qs[i]
        v = (torch.rand(model.nv, generator=rng, dtype=torch.float64) - 0.5)
        data = model.create_data()
        A_g = compute_centroidal_map(model, data, q)
        h_g_expected = (A_g @ v.unsqueeze(-1)).squeeze(-1)
        h_g = compute_centroidal_momentum(model, model.create_data(), q, v)
        torch.testing.assert_close(h_g, h_g_expected, rtol=1e-12, atol=1e-12)


def test_zero_velocity_gives_zero_momentum():
    """``v = 0 ⇒ h_g = 0`` for any ``q``."""
    model = _panda()
    qs = sample_panda_q(2, seed=4)
    for i in range(qs.shape[0]):
        q = qs[i]
        zero_v = torch.zeros(model.nv, dtype=torch.float64)
        _, h_g = ccrba(model, model.create_data(), q, zero_v)
        torch.testing.assert_close(
            h_g,
            torch.zeros(6, dtype=torch.float64),
            rtol=1e-12, atol=1e-12,
        )


def test_centroidal_g1_free_flyer():
    """G1 free-flyer: a unit base translation produces ``total_mass · ê_x`` linear momentum."""
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description

    model = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
    q = model.q_neutral.clone()
    # Free-flyer first 6 v slots: [vx, vy, vz, wx, wy, wz]; pure x-translation.
    v = torch.zeros(model.nv, dtype=torch.float64)
    v[0] = 1.0

    _, h_g = ccrba(model, model.create_data(), q, v)
    total_mass = sum(model.body_inertias[i, 0].item() for i in range(model.njoints))
    torch.testing.assert_close(
        h_g[0],
        torch.tensor(total_mass, dtype=torch.float64),
        rtol=1e-10, atol=1e-10,
    )
    # No rotation ⇒ angular momentum stays zero (body is rigid as a whole).
    # ~1e-8 fp noise from cross-product accumulation across 30+ joints.
    torch.testing.assert_close(
        h_g[3:],
        torch.zeros(3, dtype=torch.float64),
        rtol=1e-7, atol=1e-7,
    )
