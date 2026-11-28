"""Centroidal momentum: invariants every valid convention satisfies.

The whole-body convention here includes every body in
``model.body_inertias`` — including bodies attached via the synthetic
``root_joint`` (a fixed joint inserted by ``io.build_model``). Pinocchio
collapses fixed-joint bodies into the ``universe`` slot and excludes
that slot from ``pin.centerOfMass`` / ``pin.ccrba``, so a direct numeric
match would require fabricating a one-off Pinocchio model. The oracles are instead:

* ``h_g[:3] == total_mass · v_com``, with ``v_com`` the forward-mode derivative of ``center_of_mass``.
* A pure base translation of the free-flying G1 at unit speed gives ``total_mass · ê_x`` and no angular momentum.
"""

from __future__ import annotations

import torch

import better_robot as br
from better_robot.dynamics import center_of_mass, centroidal_momentum
from better_robot.kinematics import forward_kinematics
from tests.support.robots import g1_urdf, panda_urdf


def _panda(dtype=torch.float64):
    return br.load(panda_urdf(), dtype=dtype)


def _sample_q(model, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(count, model.nq, generator=generator, dtype=torch.float64) * 2.0 - 1.0


def test_hg_linear_equals_mass_times_vcom():
    """``h_g[:3] = m_total · v_com`` for any (q, v)."""
    model = _panda()
    qs = _sample_q(model, 4, seed=0)
    rng = torch.Generator().manual_seed(1)
    total_mass = torch.tensor(
        sum(model.body_inertias[i, 0].item() for i in range(model.njoints)),
        dtype=torch.float64,
    )
    for i in range(qs.shape[0]):
        q = qs[i]
        v = torch.rand(model.nv, generator=rng, dtype=torch.float64) - 0.5
        # Panda is all-revolute, so dq/dt = v and v_com is the directional derivative of the COM.
        _, v_com = torch.func.jvp(lambda q_: center_of_mass(model, forward_kinematics(model, q_)), (q,), (v,))
        h_g = centroidal_momentum(model, forward_kinematics(model, q), v=v)
        torch.testing.assert_close(h_g[:3], total_mass * v_com, rtol=1e-10, atol=1e-10)


def test_centroidal_g1_free_flyer():
    """G1 free-flyer: a unit base translation produces ``total_mass · ê_x`` linear momentum."""

    model = br.load(g1_urdf(), root_joint=br.model.JointFreeFlyer(), dtype=torch.float64)
    q = model.q_neutral.clone()
    # Free-flyer first 6 v slots: [vx, vy, vz, wx, wy, wz]; pure x-translation.
    v = torch.zeros(model.nv, dtype=torch.float64)
    v[0] = 1.0

    h_g = centroidal_momentum(model, forward_kinematics(model, q), v=v)
    total_mass = sum(model.body_inertias[i, 0].item() for i in range(model.njoints))
    torch.testing.assert_close(
        h_g[0],
        torch.tensor(total_mass, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    # No rotation ⇒ angular momentum stays zero (body is rigid as a whole).
    # ~1e-8 fp noise from cross-product accumulation across 30+ joints.
    torch.testing.assert_close(
        h_g[3:],
        torch.zeros(3, dtype=torch.float64),
        rtol=1e-7,
        atol=1e-7,
    )
