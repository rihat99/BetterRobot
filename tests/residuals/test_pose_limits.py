"""Object-referenced pose and limit residual contracts."""

from __future__ import annotations

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics
from better_robot.optim import Problem, RobotVariable, Variable
from better_robot.residuals import (
    JointPositionLimit,
    JointVelocityLimit,
    OrientationResidual,
    PoseResidual,
    PositionResidual,
    TimeIndexedResidual,
)
from better_robot.residuals.nodes import RobotState
from better_robot.residuals.structure import TemporalPattern


@pytest.fixture(scope="module")
def model():
    placement = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    builder = ModelBuilder("pose_limits")
    base = builder.add_body("base", mass=1.0)
    first = builder.add_body("first", mass=1.0)
    second = builder.add_body("second", mass=1.0)
    builder.add_revolute_z("joint_1", parent=base, child=first, origin=placement, lower=-0.5, upper=0.5)
    builder.add_revolute_y("joint_2", parent=first, child=second, origin=placement, lower=-0.6, upper=0.6)
    builder.add_frame("tip", parent_body=second, placement=placement)
    return build_model(builder.finalize(), dtype=torch.float64)


def _target(model, q: torch.Tensor) -> torch.Tensor:
    data = forward_kinematics(model, q, compute_frames=True)
    return data.frame_pose_world[..., model.frame_id("tip"), :]


def test_pose_holds_robot_and_static_target_references(model) -> None:
    tangent = torch.tensor([0.2, -0.15], dtype=torch.float64)
    q = RobotVariable(model, model.integrate(model.q_neutral, tangent), name="q")
    target = Variable(_target(model, model.q_neutral), name="target", trainable=False)
    residual = PoseResidual(q, frame="tip", target=target, weight=0.25)

    assert residual.variables == (q, target)
    assert residual.nodes[0].q is q
    torch.testing.assert_close(residual.weighted_error(), residual.error() * 0.25)
    problem = Problem([residual])
    problem.error()
    assert problem.variables == {"q": q, "target": target}


@pytest.mark.parametrize("residual_type", [PoseResidual, PositionResidual, OrientationResidual])
def test_kinematic_analytic_jacobians_match_tangent_finite_difference(model, residual_type) -> None:
    tangent = torch.tensor([0.1, -0.08], dtype=torch.float64)
    q = RobotVariable(model, model.integrate(model.q_neutral, tangent), name="q")
    residual = residual_type(q, frame="tip", target=_target(model, model.q_neutral))
    problem = Problem([residual])

    analytic = problem.jacobian_blocks(strategy="analytic")[(residual.name, q.name)]
    finite_difference = problem.jacobian_blocks(
        strategy="finite_difference",
        fd_eps=1e-6,
    )[(residual.name, q.name)]

    torch.testing.assert_close(analytic, finite_difference, rtol=2e-5, atol=2e-7)


def test_pose_and_time_adapter_support_trajectory_knots(model) -> None:
    horizon = 5
    tangent = torch.linspace(-0.1, 0.1, horizon * model.nv, dtype=torch.float64).reshape(horizon, model.nv)
    tensor = model.integrate(model.q_neutral.expand(horizon, -1), tangent)
    q = RobotVariable(model, tensor, name="q", time_axis=0)
    goal = _target(model, tensor[-1])
    pose = PoseResidual(RobotState(q), frame="tip", target=goal, knot=-1)

    assert pose.temporal_structure(q) == TemporalPattern(1, 6, horizon - 1, (0,))
    assert pose.error().shape == (6,)
    assert pose.jacobian()[0].shape == (6, horizon * model.nv)

    inner = PositionResidual(q, frame="tip", target=_target(model, tensor[2]))
    indexed = TimeIndexedResidual(inner, 2)
    assert indexed.temporal_structure("q") == TemporalPattern(1, 3, 2, (0,))
    torch.testing.assert_close(indexed.error(), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0.0)
    assert indexed.jacobian()[0].shape == (3, horizon * model.nv)


def test_joint_position_limit_uses_base_weight_once(model) -> None:
    tensor = model.q_neutral.clone()
    tensor[0] = 0.7
    tensor[1] = -0.9
    q = RobotVariable(model, tensor, name="q")
    residual = JointPositionLimit(q, weight=0.4)

    expected = torch.tensor([0.0, 0.3, 0.2, 0.0], dtype=torch.float64)
    torch.testing.assert_close(residual.error(), expected, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(residual.weighted_error(), expected * 0.4, atol=1e-12, rtol=0.0)
    jacobian = residual.jacobian()[0]
    assert jacobian.shape == (2 * model.nq, model.nv)
    assert torch.equal(jacobian[1], torch.tensor([0.0, -1.0], dtype=torch.float64))
    assert torch.equal(jacobian[2], torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_joint_velocity_limit_holds_velocity_and_static_limit(model) -> None:
    velocity = Variable(torch.tensor([-0.8, 0.35], dtype=torch.float64), name="velocity")
    limit = Variable(torch.tensor([0.5, 0.3], dtype=torch.float64), name="limit", trainable=False)
    residual = JointVelocityLimit(velocity, limit, weight=0.2)

    expected = torch.tensor([0.3, 0.0, 0.0, 0.05], dtype=torch.float64)
    assert residual.variables == (velocity, limit)
    torch.testing.assert_close(residual.error(), expected)
    torch.testing.assert_close(residual.weighted_error(), expected * 0.2)
    assert residual.jacobian() is None


def test_pose_rejects_invalid_frame_and_trainable_target(model) -> None:
    q = RobotVariable(model, model.q_neutral, name="q")
    target = Variable(_target(model, model.q_neutral), name="target")

    with pytest.raises(ValueError, match="trainable=False"):
        PoseResidual(q, frame="tip", target=target)
    with pytest.raises(ValueError, match=r"\[0,"):
        PoseResidual(q, frame_id=model.nframes, target=target.tensor)
