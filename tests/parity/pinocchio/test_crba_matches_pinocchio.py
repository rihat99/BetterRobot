"""CRBA parity: ``M(q)`` matches ``pin.crba`` element-wise on the Panda in fp64 (1e-10 relative tolerance; the
measured gap is 1e-15), with and without armature."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import torch

from better_robot.dynamics import inverse_dynamics, mass_matrix
from better_robot.kinematics import forward_kinematics

from .conftest import sample_panda_q


_RTOL = 1e-10


def test_crba_matches_pinocchio(panda_both):
    """``M(q)`` matches Pinocchio's ``crba`` on Panda."""
    br_model, pin_model, pin_data, _ = panda_both
    qs = sample_panda_q(8, seed=0)
    for i in range(qs.shape[0]):
        q = qs[i]
        M_br = mass_matrix(br_model, forward_kinematics(br_model, q)).detach().cpu().numpy()
        M_pin = np.asarray(pin.crba(pin_model, pin_data, q.numpy()))
        # Pinocchio fills only the upper triangle by default; mirror it.
        M_pin_full = np.triu(M_pin) + np.triu(M_pin, k=1).T
        np.testing.assert_allclose(M_br, M_pin_full, rtol=_RTOL, atol=_RTOL)


def test_armature_matches_pinocchio(panda_both):
    """``model.armature`` enters RNEA (``+ armature * a``) and CRBA (diagonal) as in Pinocchio."""
    br_model, pin_model, _, _ = panda_both
    armature = torch.linspace(0.01, 0.09, br_model.nv, dtype=torch.float64)
    br_model = br_model.with_values(armature=armature)
    pin_model = pin.Model(pin_model)
    pin_model.armature = armature.numpy()
    pin_data = pin_model.createData()
    rng = torch.Generator().manual_seed(4)
    q = sample_panda_q(1, seed=4)[0]
    v, a = torch.rand(2, br_model.nv, generator=rng, dtype=torch.float64) - 0.5
    state = forward_kinematics(br_model, q)
    tau_pin = pin.rnea(pin_model, pin_data, q.numpy(), v.numpy(), a.numpy())
    np.testing.assert_allclose(
        inverse_dynamics(br_model, state, v=v, a=a).numpy(),
        tau_pin,
        rtol=_RTOL,
        atol=_RTOL,
        err_msg="RNEA with armature",
    )
    M_pin = np.asarray(pin.crba(pin_model, pin_data, q.numpy()))
    M_pin_full = np.triu(M_pin) + np.triu(M_pin, k=1).T
    np.testing.assert_allclose(
        mass_matrix(br_model, state).numpy(), M_pin_full, rtol=_RTOL, atol=_RTOL, err_msg="CRBA with armature"
    )
