"""Per-joint tangent-space rotation priors."""

from __future__ import annotations

import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.optim import Problem, ResidualItem, RobotConfig, VarSpec
from better_robot.residuals import JointRotationPrior


def _model():
    builder = ModelBuilder("rotation_prior")
    base = builder.add_body("base")
    first = builder.add_body("first", mass=1.0)
    second = builder.add_body("second", mass=1.0)
    builder.add_spherical("first_ball", parent=base, child=first)
    builder.add_spherical("second_ball", parent=first, child=second)
    return build_model(builder.finalize()).to(dtype=torch.float64)


def test_joint_weights_expand_over_tangent_slices_and_batch() -> None:
    model = _model()
    joint_weights = torch.tensor([0.0, 0.0, 2.0, 0.25], dtype=torch.float64)
    prior = JointRotationPrior(model, model.q_neutral, joint_weights)
    tangent = torch.tensor([0.2, -0.1, 0.3, -0.4, 0.5, 0.1], dtype=torch.float64)
    q = model.integrate(model.q_neutral, tangent)
    batched = torch.stack((q, q)).reshape(1, 2, model.nq)

    expected_weight = torch.tensor([2.0, 2.0, 2.0, 0.25, 0.25, 0.25], dtype=torch.float64)
    expected = tangent * expected_weight
    torch.testing.assert_close(prior({"q": q}), expected, atol=1e-12, rtol=1e-12)
    actual = prior({"q": batched})
    assert actual.shape == (1, 2, model.nv)
    torch.testing.assert_close(actual, expected.expand(1, 2, model.nv), atol=1e-12, rtol=1e-12)


def test_explicit_tangent_weights_match_manual_difference() -> None:
    model = _model()
    weights = torch.linspace(0.1, 0.6, model.nv, dtype=torch.float64)
    prior = JointRotationPrior(model, model.q_neutral, weights)
    tangent = torch.linspace(-0.25, 0.3, model.nv, dtype=torch.float64)
    q = model.integrate(model.q_neutral, tangent)

    expected = model.difference(model.q_neutral, q) * weights
    torch.testing.assert_close(prior({"q": q}), expected)


def test_named_block_ad_matches_finite_difference_away_from_mean() -> None:
    model = _model()
    weights = torch.linspace(0.2, 0.9, model.nv, dtype=torch.float64)
    prior = JointRotationPrior(model, model.q_neutral, weights)
    tangent = torch.tensor([0.3, -0.2, 0.15, -0.1, 0.25, 0.2], dtype=torch.float64)
    q = model.integrate(model.q_neutral, tangent)
    problem = Problem(
        vars=(VarSpec("q", (model.nq,), RobotConfig(model)),),
        residuals=(ResidualItem(prior.name, prior),),
    )

    ad = problem.jacobian_blocks({"q": q}, strategy="jacrev")[(prior.name, "q")]
    fd = problem.jacobian_blocks(
        {"q": q},
        strategy="finite_difference",
        fd_eps=1e-6,
    )[(prior.name, "q")]
    torch.testing.assert_close(ad, fd, atol=2e-6, rtol=2e-5)


def test_identity_prior_has_finite_fp32_tangent_gradient() -> None:
    model = _model().to(dtype=torch.float32)
    prior = JointRotationPrior(model, model.q_neutral, torch.ones(model.njoints))
    problem = Problem(
        vars=(VarSpec("q", (model.nq,), RobotConfig(model)),),
        residuals=(ResidualItem(prior.name, prior),),
    )

    jacobian = problem.dense_jacobian({"q": model.q_neutral}, strategy="jacrev")
    gradient = problem.gradient({"q": model.q_neutral})["q"]
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
