"""Swing/twist limits for spherical-joint robot configurations."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.lie import so3
from better_robot.optim import Problem, ResidualItem, RobotConfig, VarSpec
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


def _residual(model, joint_ids=(2,)) -> SwingTwistLimitResidual:
    return SwingTwistLimitResidual(
        model,
        joint_ids,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        swing_max=0.4,
        twist_range=(-0.3, 0.5),
    )


def test_inside_and_each_one_sided_violation() -> None:
    model = _two_spherical_model()
    residual = _residual(model)
    neutral = model.q_neutral

    inside = _with_rotation(model, neutral, 2, so3.exp(torch.tensor([0.2, 0.0, 0.1], dtype=torch.float64)))
    swing = _with_rotation(model, neutral, 2, so3.exp(torch.tensor([0.8, 0.0, 0.0], dtype=torch.float64)))
    lower = _with_rotation(model, neutral, 2, so3.exp(torch.tensor([0.0, 0.0, -0.7], dtype=torch.float64)))
    upper = _with_rotation(model, neutral, 2, so3.exp(torch.tensor([0.0, 0.0, 0.9], dtype=torch.float64)))

    torch.testing.assert_close(residual({"q": inside}), torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(residual({"q": swing}), torch.tensor([0.4, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(residual({"q": lower}), torch.tensor([0.0, 0.4, 0.0], dtype=torch.float64))
    torch.testing.assert_close(residual({"q": upper}), torch.tensor([0.0, 0.0, 0.4], dtype=torch.float64))


def test_quaternion_double_cover_and_batches() -> None:
    model = _two_spherical_model()
    residual = _residual(model, joint_ids=(2, 3))
    q = model.q_neutral
    q = _with_rotation(model, q, 2, so3.exp(torch.tensor([0.7, 0.1, 0.8], dtype=torch.float64)))
    q = _with_rotation(model, q, 3, so3.exp(torch.tensor([-0.6, 0.2, -0.5], dtype=torch.float64)))
    q_negative = q.clone()
    for joint_id in (2, 3):
        start = model.idx_qs[joint_id]
        q_negative[start : start + 4] *= -1.0

    expected = residual({"q": q})
    torch.testing.assert_close(residual({"q": q_negative}), expected, atol=1e-12, rtol=1e-12)
    batched = torch.stack((q, q_negative)).reshape(1, 2, model.nq)
    actual = residual({"q": batched})
    assert actual.shape == (1, 2, 6)
    torch.testing.assert_close(actual[0, 0], expected)
    torch.testing.assert_close(actual[0, 1], expected)


def test_named_block_tangent_ad_matches_finite_difference_and_is_sparse() -> None:
    model = _two_spherical_model()
    residual = _residual(model, joint_ids=(3,))
    q = _with_rotation(
        model,
        model.q_neutral,
        3,
        so3.exp(torch.tensor([0.7, -0.2, 0.8], dtype=torch.float64)),
    )
    problem = Problem(
        vars=(VarSpec("q", (model.nq,), RobotConfig(model)),),
        residuals=(ResidualItem(residual.name, residual),),
    )

    ad = problem.jacobian_blocks({"q": q}, strategy="jacrev")[(residual.name, "q")]
    fd = problem.jacobian_blocks(
        {"q": q},
        strategy="finite_difference",
        fd_eps=1e-6,
    )[(residual.name, "q")]

    torch.testing.assert_close(ad, fd, atol=2e-6, rtol=2e-5)
    first_start = model.idx_vs[2]
    second_start = model.idx_vs[3]
    torch.testing.assert_close(ad[..., :first_start], torch.zeros_like(ad[..., :first_start]))
    torch.testing.assert_close(
        ad[..., first_start:second_start],
        torch.zeros_like(ad[..., first_start:second_start]),
    )
    assert torch.count_nonzero(ad[..., second_start : second_start + 3]) > 0
    assert residual.jacobian({"q": q}) is None


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_pure_pi_swing_uses_finite_zero_twist_convention(dtype: torch.dtype) -> None:
    model = _two_spherical_model().to(dtype=dtype)
    residual = SwingTwistLimitResidual(
        model,
        (2,),
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        swing_max=0.4,
        twist_range=(-0.3, 0.5),
    )
    q = _with_rotation(
        model,
        model.q_neutral,
        2,
        so3.exp(torch.tensor([math.pi, 0.0, 0.0], dtype=dtype)),
    )
    spec = VarSpec("q", (model.nq,), RobotConfig(model))
    problem = Problem(vars=(spec,), residuals=(ResidualItem(residual.name, residual),))

    value = residual({"q": q})
    gradient = problem.gradient({"q": q})["q"]
    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(value[1:], torch.zeros(2, dtype=dtype))


def test_constructor_rejects_non_spherical_or_ambiguous_limits() -> None:
    model = _two_spherical_model()
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="spherical joints only"):
        SwingTwistLimitResidual(model, (1,), axis, 0.4, (-0.3, 0.5))
    with pytest.raises(ValueError, match="unit norm"):
        SwingTwistLimitResidual(model, (2,), axis * 2.0, 0.4, (-0.3, 0.5))
    with pytest.raises(ValueError, match="non-wrapping"):
        SwingTwistLimitResidual(model, (2,), axis, 0.4, (2.5, -2.5))
