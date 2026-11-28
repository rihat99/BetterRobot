"""fp64 ``gradcheck`` on the pure-PyTorch SE3/SO3 backend.

They run on CPU. Each tested op accepts fp64 inputs and is numerically stable across
``torch.autograd.gradcheck`` finite differences.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.lie import se3, so3


def _random_poses(generator: torch.Generator, translation_scale: float) -> torch.Tensor:
    """``(2, 7)`` float64 leaf poses: translations ``N(0, translation_scale^2)``, unit quaternions of normal draws."""
    t = torch.randn(2, 3, generator=generator, dtype=torch.float64) * translation_scale
    q = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    return torch.cat([t, q / q.norm(dim=-1, keepdim=True)], dim=-1).requires_grad_(True)


@pytest.mark.parametrize("op", [se3.log, se3.inverse])
def test_se3_op_gradcheck(op) -> None:
    a = _random_poses(torch.Generator().manual_seed(0), 0.2)
    assert torch.autograd.gradcheck(op, (a,), atol=1e-6, rtol=1e-5)


def test_se3_exp_gradcheck() -> None:
    rng = torch.Generator().manual_seed(1)
    xi = (torch.randn(2, 6, generator=rng, dtype=torch.float64) * 0.2).requires_grad_(True)
    assert torch.autograd.gradcheck(se3.exp, (xi,), atol=1e-6, rtol=1e-5)


def test_se3_compose_gradcheck() -> None:
    rng = torch.Generator().manual_seed(2)
    a, b = _random_poses(rng, 0.1), _random_poses(rng, 0.1)
    assert torch.autograd.gradcheck(se3.compose, (a, b), atol=1e-6, rtol=1e-5)


def test_se3_act_gradcheck() -> None:
    rng = torch.Generator().manual_seed(3)
    T = _random_poses(rng, 0.1)
    p = torch.randn(2, 3, generator=rng, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(se3.act, (T, p), atol=1e-6, rtol=1e-5)


def test_so3_exp_gradcheck() -> None:
    rng = torch.Generator().manual_seed(4)
    omega = (torch.randn(2, 3, generator=rng, dtype=torch.float64) * 0.2).requires_grad_(True)
    assert torch.autograd.gradcheck(so3.exp, (omega,), atol=1e-6, rtol=1e-5)


def test_so3_log_gradcheck() -> None:
    rng = torch.Generator().manual_seed(5)
    q = torch.randn(2, 4, generator=rng, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    q.requires_grad_(True)
    assert torch.autograd.gradcheck(so3.log, (q,), atol=1e-6, rtol=1e-5)
