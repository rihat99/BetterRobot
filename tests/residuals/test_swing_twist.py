"""Swing/twist limits for spherical-joint robot configurations."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.lie import so3
from better_robot.optim.problem import Problem
from better_robot.optim.variables import RobotVariable
from better_robot.residuals import SwingTwistLimitResidual


def _two_spherical_model():
    builder = ModelBuilder("two_spherical")
    base = builder.add_body("base")
    first = builder.add_body("first", mass=1.0)
    second = builder.add_body("second", mass=1.0)
    builder.add_spherical("first_ball", parent=base, child=first)
    builder.add_spherical("second_ball", parent=first, child=second)
    return build_model(builder.finalize()).to(dtype=torch.float64)


def _with_rotation(model, q: torch.Tensor, joint_id: int, rotation: torch.Tensor) -> torch.Tensor:
    result = q.clone()
    start = model.idx_qs[joint_id]
    result[..., start : start + 4] = rotation
    return result


def _residual(
    model,
    value: torch.Tensor | None = None,
    joint_ids=(2,),
) -> tuple[RobotVariable, SwingTwistLimitResidual]:
    q = RobotVariable(model, model.q_neutral if value is None else value, name="q")
    item = SwingTwistLimitResidual(
        q,
        joint_ids,
        torch.tensor([0.0, 0.0, 1.0], dtype=q.tensor.dtype),
        swing_max=0.4,
        twist_range=(-0.3, 0.5),
    )
    return q, item


def test_inside_and_each_one_sided_violation() -> None:
    model = _two_spherical_model()
    q, item = _residual(model)
    neutral = model.q_neutral
    cases = (
        (
            _with_rotation(model, neutral, 2, so3.exp(torch.tensor([0.2, 0.0, 0.1], dtype=torch.float64))),
            torch.zeros(3, dtype=torch.float64),
        ),
        (
            _with_rotation(model, neutral, 2, so3.exp(torch.tensor([0.8, 0.0, 0.0], dtype=torch.float64))),
            torch.tensor([0.4, 0.0, 0.0], dtype=torch.float64),
        ),
        (
            _with_rotation(model, neutral, 2, so3.exp(torch.tensor([0.0, 0.0, -0.7], dtype=torch.float64))),
            torch.tensor([0.0, 0.4, 0.0], dtype=torch.float64),
        ),
        (
            _with_rotation(model, neutral, 2, so3.exp(torch.tensor([0.0, 0.0, 0.9], dtype=torch.float64))),
            torch.tensor([0.0, 0.0, 0.4], dtype=torch.float64),
        ),
    )

    for value, expected in cases:
        q.tensor = value
        torch.testing.assert_close(item.error(), expected)


def test_quaternion_double_cover_and_batches() -> None:
    model = _two_spherical_model()
    value = model.q_neutral
    value = _with_rotation(model, value, 2, so3.exp(torch.tensor([0.7, 0.1, 0.8], dtype=torch.float64)))
    value = _with_rotation(model, value, 3, so3.exp(torch.tensor([-0.6, 0.2, -0.5], dtype=torch.float64)))
    negative = value.clone()
    for joint_id in (2, 3):
        start = model.idx_qs[joint_id]
        negative[start : start + 4] *= -1.0

    _q, scalar = _residual(model, value, joint_ids=(2, 3))
    expected = scalar.error()
    _batched_q, batched = _residual(
        model,
        torch.stack((value, negative)).reshape(1, 2, model.nq),
        joint_ids=(2, 3),
    )
    actual = batched.error()

    assert actual.shape == (1, 2, 6)
    torch.testing.assert_close(actual[0, 0], expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual[0, 1], expected, atol=1e-12, rtol=1e-12)


def test_named_block_tangent_ad_matches_finite_difference_and_is_sparse() -> None:
    model = _two_spherical_model()
    value = _with_rotation(
        model,
        model.q_neutral,
        3,
        so3.exp(torch.tensor([0.7, -0.2, 0.8], dtype=torch.float64)),
    )
    _q, item = _residual(model, value, joint_ids=(3,))
    problem = Problem([item])

    ad = problem.dense_jacobian(strategy="jacrev")
    finite_difference = problem.jacobian_blocks(strategy="finite_difference", fd_eps=1e-6)[(item.name, "q")]

    torch.testing.assert_close(ad, finite_difference, atol=2e-6, rtol=2e-5)
    first_start = model.idx_vs[2]
    second_start = model.idx_vs[3]
    torch.testing.assert_close(ad[..., :first_start], torch.zeros_like(ad[..., :first_start]))
    torch.testing.assert_close(ad[..., first_start:second_start], torch.zeros_like(ad[..., first_start:second_start]))
    assert torch.count_nonzero(ad[..., second_start : second_start + 3]) > 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_pure_pi_swing_uses_finite_zero_twist_convention(dtype: torch.dtype) -> None:
    model = _two_spherical_model().to(dtype=dtype)
    value = _with_rotation(
        model,
        model.q_neutral,
        2,
        so3.exp(torch.tensor([math.pi, 0.0, 0.0], dtype=dtype)),
    )
    q = RobotVariable(model, value, name="q")
    item = SwingTwistLimitResidual(
        q,
        (2,),
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        swing_max=0.4,
        twist_range=(-0.3, 0.5),
    )
    problem = Problem([item])

    value = item.error()
    gradient = problem.gradient()["q"]
    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(value[1:], torch.zeros(2, dtype=dtype))


def test_constructor_rejects_non_spherical_or_ambiguous_limits() -> None:
    model = _two_spherical_model()
    q = RobotVariable(model, name="q")
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="spherical joints only"):
        SwingTwistLimitResidual(q, (1,), axis, 0.4, (-0.3, 0.5))
    with pytest.raises(ValueError, match="unit norm"):
        SwingTwistLimitResidual(q, (2,), axis * 2.0, 0.4, (-0.3, 0.5))
    with pytest.raises(ValueError, match="non-wrapping"):
        SwingTwistLimitResidual(q, (2,), axis, 0.4, (2.5, -2.5))
