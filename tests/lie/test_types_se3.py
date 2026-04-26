"""Tests for :class:`better_robot.lie.types.SE3` typed value class.

Each method must agree with the functional API in ``lie.se3`` to within
``1e-6`` (fp32) / ``1e-12`` (fp64) on randomised batches. The only
operator overloaded is ``@`` (composition / point action); ``*`` raises.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.lie import se3 as _se3
from better_robot.lie.types import SE3, Pose


def _rand_se3(batch=(4,), dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(0)
    xi = torch.randn(*batch, 6, dtype=dtype) * 0.5
    return _se3.exp(xi)


def _atol(dtype) -> float:
    return 1e-6 if dtype == torch.float32 else 1e-12


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compose_matches_functional(dtype):
    A = _rand_se3((4,), dtype)
    B = _rand_se3((4,), dtype)
    out = SE3(A) @ SE3(B)
    assert torch.allclose(out.tensor, _se3.compose(A, B), atol=_atol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_act_on_point_matches_functional(dtype):
    T = _rand_se3((4,), dtype)
    p = torch.randn(4, 3, dtype=dtype)
    out = SE3(T) @ p
    assert torch.allclose(out, _se3.act(T, p), atol=_atol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse_log_exp_match_functional(dtype):
    T = _rand_se3((4,), dtype)
    inv = SE3(T).inverse()
    assert torch.allclose(inv.tensor, _se3.inverse(T), atol=_atol(dtype))
    xi = SE3(T).log()
    assert torch.allclose(xi, _se3.log(T), atol=_atol(dtype))
    re = SE3.exp(xi)
    assert torch.allclose(re.tensor, _se3.exp(xi), atol=_atol(dtype))


def test_scalar_mul_raises():
    T = SE3(_rand_se3((1,), torch.float32))
    with pytest.raises(TypeError):
        _ = T * 2.0
    with pytest.raises(TypeError):
        _ = 2.0 * T


def test_pose_alias_is_se3():
    assert Pose is SE3
    T = Pose.identity()
    assert isinstance(T, SE3)


def test_translation_rotation_accessors():
    T = _rand_se3((2, 3), torch.float64)
    se3_obj = SE3(T)
    assert torch.equal(se3_obj.translation, T[..., :3])
    assert torch.equal(se3_obj.rotation.tensor, T[..., 3:7])


def test_top_level_export_works():
    import better_robot
    assert better_robot.lie.SE3 is SE3
