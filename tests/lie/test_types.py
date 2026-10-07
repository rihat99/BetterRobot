"""Tests for the typed value classes :class:`better_robot.lie.types.SO3` and ``SE3``.

Each method must agree with the functional API in ``lie.so3`` / ``lie.se3`` to within
``1e-6`` (fp32) / ``1e-12`` (fp64) on randomised batches. The only
operator overloaded is ``@`` (composition / point action); ``*`` raises.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.lie import se3 as _se3
from better_robot.lie import so3 as _so3
from better_robot.lie.types import SE3, SO3

# (functional module, typed class, RNG seed, tangent dim)
_GROUPS = [
    pytest.param((_so3, SO3, 1, 3), id="SO3"),
    pytest.param((_se3, SE3, 0, 6), id="SE3"),
]


def _rand(group, batch=(4,), dtype=torch.float64) -> torch.Tensor:
    functional, _, seed, dim = group
    torch.manual_seed(seed)
    tangent = torch.randn(*batch, dim, dtype=dtype) * 0.5
    return functional.exp(tangent)


def _atol(dtype) -> float:
    return 1e-6 if dtype == torch.float32 else 1e-12


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("group", _GROUPS)
def test_compose_matches_functional(group, dtype):
    functional, typed, _, _ = group
    A = _rand(group, (4,), dtype)
    B = _rand(group, (4,), dtype)
    out = typed(A) @ typed(B)
    assert torch.allclose(out.tensor, functional.compose(A, B), atol=_atol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("group", _GROUPS)
def test_act_on_point_matches_functional(group, dtype):
    functional, typed, _, _ = group
    X = _rand(group, (4,), dtype)
    p = torch.randn(4, 3, dtype=dtype)
    out = typed(X) @ p
    assert torch.allclose(out, functional.act(X, p), atol=_atol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("group", _GROUPS)
def test_inverse_log_exp_match_functional(group, dtype):
    functional, typed, _, _ = group
    X = _rand(group, (4,), dtype)
    assert torch.allclose(typed(X).inverse().tensor, functional.inverse(X), atol=_atol(dtype))
    tangent = typed(X).log()
    assert torch.allclose(tangent, functional.log(X), atol=_atol(dtype))
    assert torch.allclose(typed.exp(tangent).tensor, functional.exp(tangent), atol=_atol(dtype))


@pytest.mark.parametrize("group", _GROUPS)
def test_scalar_mul_raises(group):
    _, typed, _, _ = group
    X = typed(_rand(group, (1,), torch.float32))
    with pytest.raises(TypeError):
        _ = X * 2.0
    with pytest.raises(TypeError):
        _ = 2.0 * X


def test_so3_to_matrix_matches_functional():
    R = _rand(_GROUPS[0].values[0], (3,), torch.float64)
    mat = SO3(R).to_matrix()
    assert torch.allclose(mat, _so3.to_matrix(R), atol=1e-12)


def test_se3_translation_rotation_accessors():
    T = _rand(_GROUPS[1].values[0], (2, 3), torch.float64)
    se3_obj = SE3(T)
    assert torch.equal(se3_obj.translation, T[..., :3])
    assert torch.equal(se3_obj.rotation.tensor, T[..., 3:7])
