"""fp64 ``gradcheck`` on the pure-PyTorch SE3/SO3 backend.

Skipped automatically when CUDA isn't part of the goal — these run on
CPU. Each tested op accepts fp64 inputs and is numerically stable across
``torch.autograd.gradcheck`` finite differences.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.lie import _torch_native_backend as tn


def _rand_quat(dtype):
    q = torch.randn(2, 4, dtype=dtype, requires_grad=True)
    # gradcheck needs differentiable input; we don't normalise in-place.
    return q


@pytest.mark.parametrize("op", [tn.se3_log, tn.se3_inverse])
def test_se3_op_gradcheck(op) -> None:
    rng = torch.Generator().manual_seed(0)
    t = torch.randn(2, 3, generator=rng, dtype=torch.float64) * 0.2
    q = torch.randn(2, 4, generator=rng, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    a = torch.cat([t, q], dim=-1).requires_grad_(True)
    assert torch.autograd.gradcheck(op, (a,), atol=1e-6, rtol=1e-5)


def test_se3_exp_gradcheck() -> None:
    rng = torch.Generator().manual_seed(1)
    xi = (torch.randn(2, 6, generator=rng, dtype=torch.float64) * 0.2).requires_grad_(True)
    assert torch.autograd.gradcheck(tn.se3_exp, (xi,), atol=1e-6, rtol=1e-5)


def test_se3_compose_gradcheck() -> None:
    rng = torch.Generator().manual_seed(2)
    def _make():
        t = torch.randn(2, 3, generator=rng, dtype=torch.float64) * 0.1
        q = torch.randn(2, 4, generator=rng, dtype=torch.float64)
        q = q / q.norm(dim=-1, keepdim=True)
        return torch.cat([t, q], dim=-1).requires_grad_(True)

    a, b = _make(), _make()
    assert torch.autograd.gradcheck(tn.se3_compose, (a, b), atol=1e-6, rtol=1e-5)


def test_se3_act_gradcheck() -> None:
    rng = torch.Generator().manual_seed(3)
    t = torch.randn(2, 3, generator=rng, dtype=torch.float64) * 0.1
    q = torch.randn(2, 4, generator=rng, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    T = torch.cat([t, q], dim=-1).requires_grad_(True)
    p = torch.randn(2, 3, generator=rng, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(tn.se3_act, (T, p), atol=1e-6, rtol=1e-5)


def test_so3_exp_gradcheck() -> None:
    rng = torch.Generator().manual_seed(4)
    omega = (torch.randn(2, 3, generator=rng, dtype=torch.float64) * 0.2).requires_grad_(True)
    assert torch.autograd.gradcheck(tn.so3_exp, (omega,), atol=1e-6, rtol=1e-5)


def test_so3_log_gradcheck() -> None:
    rng = torch.Generator().manual_seed(5)
    q = torch.randn(2, 4, generator=rng, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    q.requires_grad_(True)
    assert torch.autograd.gradcheck(tn.so3_log, (q,), atol=1e-6, rtol=1e-5)
