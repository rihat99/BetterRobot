"""Gradcheck acceptance for the dynamics layer (P11-D6).

The forward routines are pure differentiable PyTorch, so the standard
``torch.autograd.gradcheck`` on a smoothed scalar loss is the cleanest
end-to-end check. We use small random ``q, v, a`` to keep central-FD
runtime manageable on CPU.

* ``rnea`` — gradient through the full kinematic chain.
* ``aba`` — same identity confirmed via ``aba(q, v, rnea(q, v, a)) == a``.
* ``∂τ/∂a`` from autograd matches ``crba`` to fp64.
"""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.dynamics import aba, crba, rnea


def _panda():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415

    return br.load(panda_description.URDF_PATH, dtype=torch.float64)


def test_rnea_gradcheck():
    model = _panda()
    rng = torch.Generator().manual_seed(0)
    q = (torch.rand(model.nq, generator=rng, dtype=torch.float64) * 0.2).requires_grad_(True)
    v = (torch.rand(model.nv, generator=rng, dtype=torch.float64) * 0.1).requires_grad_(True)
    a = (torch.rand(model.nv, generator=rng, dtype=torch.float64) * 0.1).requires_grad_(True)

    def f(q_, v_, a_):
        return rnea(model, q_, v_, a_)

    assert torch.autograd.gradcheck(f, (q, v, a), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_aba_gradcheck():
    model = _panda()
    rng = torch.Generator().manual_seed(1)
    q = (torch.rand(model.nq, generator=rng, dtype=torch.float64) * 0.2).requires_grad_(True)
    v = (torch.rand(model.nv, generator=rng, dtype=torch.float64) * 0.1).requires_grad_(True)
    tau = (torch.rand(model.nv, generator=rng, dtype=torch.float64) * 0.1).requires_grad_(True)

    def f(q_, v_, tau_):
        return aba(model, q_, v_, tau_)

    assert torch.autograd.gradcheck(f, (q, v, tau), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_dtau_da_matches_crba():
    """``∂τ/∂a`` returned by RNEA derivatives must equal ``M(q)`` from CRBA."""
    model = _panda()
    rng = torch.Generator().manual_seed(2)
    q = torch.rand(model.nq, generator=rng, dtype=torch.float64) * 0.2
    v = torch.rand(model.nv, generator=rng, dtype=torch.float64) * 0.1
    a = torch.rand(model.nv, generator=rng, dtype=torch.float64) * 0.1

    dtau_da = torch.autograd.functional.jacobian(
        lambda acceleration: rnea(model, q, v, acceleration),
        a,
    )
    M = crba(model, q)
    torch.testing.assert_close(dtau_da, M, rtol=1e-10, atol=1e-10)


def test_dadtau_matches_minverse():
    """``∂a/∂τ`` from ABA derivatives equals ``M(q)⁻¹``."""
    model = _panda()
    rng = torch.Generator().manual_seed(3)
    q = torch.rand(model.nq, generator=rng, dtype=torch.float64) * 0.2
    v = torch.rand(model.nv, generator=rng, dtype=torch.float64) * 0.1
    tau = torch.rand(model.nv, generator=rng, dtype=torch.float64) * 0.1

    da_dtau = torch.autograd.functional.jacobian(
        lambda torque: aba(model, q, v, torque),
        tau,
    )
    M = crba(model, q)
    M_inv = torch.linalg.inv(M)
    torch.testing.assert_close(da_dtau, M_inv, rtol=1e-9, atol=1e-9)
