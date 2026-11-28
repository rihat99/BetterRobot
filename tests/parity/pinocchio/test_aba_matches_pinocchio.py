"""ABA parity: ``forward_dynamics(q, v, τ)`` matches ``pin.aba`` on the Panda in fp64."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import torch

from better_robot.dynamics import forward_dynamics
from better_robot.kinematics import forward_kinematics

from .conftest import sample_panda_q


_RTOL = 1e-10
_ATOL = 1e-10


def test_aba_matches_pinocchio(panda_both):
    br_model, pin_model, pin_data, _ = panda_both
    qs = sample_panda_q(8, seed=0)
    rng = torch.Generator().manual_seed(1)
    for i in range(qs.shape[0]):
        q = qs[i]
        v = torch.rand(br_model.nv, generator=rng, dtype=torch.float64) - 0.5
        tau = torch.rand(br_model.nv, generator=rng, dtype=torch.float64) - 0.5
        ddq_br = forward_dynamics(br_model, forward_kinematics(br_model, q), v=v, tau=tau).detach().cpu().numpy()
        ddq_pin = np.asarray(pin.aba(pin_model, pin_data, q.numpy(), v.numpy(), tau.numpy()))
        np.testing.assert_allclose(ddq_br, ddq_pin, rtol=_RTOL, atol=_ATOL)
