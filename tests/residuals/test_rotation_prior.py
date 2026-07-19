"""Per-joint tangent-space rotation priors."""

from __future__ import annotations

import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.optim import Problem, RobotVariable
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
    tangent = torch.tensor([0.2, -0.1, 0.3, -0.4, 0.5, 0.1], dtype=torch.float64)
    tensor = model.integrate(model.q_neutral, tangent)
    q = RobotVariable(model, tensor, name="q")
    prior = JointRotationPrior(q, model.q_neutral, joint_weights)
    batched = torch.stack((tensor, tensor)).reshape(1, 2, model.nq)

    expected_weight = torch.tensor([2.0, 2.0, 2.0, 0.25, 0.25, 0.25], dtype=torch.float64)
    expected = tangent * expected_weight
    torch.testing.assert_close(prior.error(), expected, atol=1e-12, rtol=1e-12)
    q.tensor = batched
    actual = prior.error()
    assert actual.shape == (1, 2, model.nv)
    torch.testing.assert_close(actual, expected.expand(1, 2, model.nv), atol=1e-12, rtol=1e-12)


def test_explicit_tangent_weights_match_manual_difference() -> None:
    model = _model()
    weights = torch.linspace(0.1, 0.6, model.nv, dtype=torch.float64)
    tangent = torch.linspace(-0.25, 0.3, model.nv, dtype=torch.float64)
    tensor = model.integrate(model.q_neutral, tangent)
    q = RobotVariable(model, tensor, name="q")
    prior = JointRotationPrior(q, model.q_neutral, weights)

    expected = model.difference(model.q_neutral, tensor) * weights
    torch.testing.assert_close(prior.error(), expected)


def test_named_block_ad_matches_finite_difference_away_from_mean() -> None:
    model = _model()
    weights = torch.linspace(0.2, 0.9, model.nv, dtype=torch.float64)
    tangent = torch.tensor([0.3, -0.2, 0.15, -0.1, 0.25, 0.2], dtype=torch.float64)
    q = RobotVariable(model, model.integrate(model.q_neutral, tangent), name="q")
    prior = JointRotationPrior(q, model.q_neutral, weights)
    problem = Problem([prior])

    ad = problem.jacobian_blocks(strategy="jacrev")[(prior.name, "q")]
    fd = problem.jacobian_blocks(
        strategy="finite_difference",
        fd_eps=1e-6,
    )[(prior.name, "q")]
    torch.testing.assert_close(ad, fd, atol=2e-6, rtol=2e-5)


def test_identity_prior_has_finite_fp32_tangent_gradient() -> None:
    model = _model().to(dtype=torch.float32)
    q = RobotVariable(model, model.q_neutral, name="q")
    prior = JointRotationPrior(q, model.q_neutral, torch.ones(model.njoints))
    problem = Problem([prior])

    jacobian = problem.dense_jacobian(strategy="jacrev")
    gradient = problem.gradient()["q"]
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
