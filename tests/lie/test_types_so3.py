"""Tests for :class:`better_robot.lie.types.SO3` typed value class."""

from __future__ import annotations

import pytest
import torch

from better_robot.lie import so3 as _so3
from better_robot.lie.types import SO3


def _rand_so3(batch=(4,), dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(1)
    w = torch.randn(*batch, 3, dtype=dtype) * 0.5
    return _so3.exp(w)


def _atol(dtype) -> float:
    return 1e-6 if dtype == torch.float32 else 1e-12


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compose_matches_functional(dtype):
    A = _rand_so3((4,), dtype)
    B = _rand_so3((4,), dtype)
    out = SO3(A) @ SO3(B)
    assert torch.allclose(out.tensor, _so3.compose(A, B), atol=_atol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_act_on_point_matches_functional(dtype):
    R = _rand_so3((4,), dtype)
    p = torch.randn(4, 3, dtype=dtype)
    out = SO3(R) @ p
    assert torch.allclose(out, _so3.act(R, p), atol=_atol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse_log_exp_match_functional(dtype):
    R = _rand_so3((4,), dtype)
    assert torch.allclose(SO3(R).inverse().tensor, _so3.inverse(R), atol=_atol(dtype))
    w = SO3(R).log()
    assert torch.allclose(w, _so3.log(R), atol=_atol(dtype))
    assert torch.allclose(SO3.exp(w).tensor, _so3.exp(w), atol=_atol(dtype))


def test_scalar_mul_raises():
    R = SO3(_rand_so3((1,), torch.float32))
    with pytest.raises(TypeError):
        _ = R * 2.0


def test_to_matrix_matches_functional():
    R = _rand_so3((3,), torch.float64)
    mat = SO3(R).to_matrix()
    assert torch.allclose(mat, _so3.to_matrix(R), atol=1e-12)
