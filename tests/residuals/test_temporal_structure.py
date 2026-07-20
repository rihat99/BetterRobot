"""Object-referenced temporal-structure contracts for trajectory residuals."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim import Problem, Residual, RobotVariable, Variable
from better_robot.residuals import (
    AccelerationResidual,
    ContactConsistencyResidual,
    PositionResidual,
    ReferenceTrajectoryResidual,
    RestResidual,
    TimeIndexedResidual,
    VelocityResidual,
)
from better_robot.residuals._temporal_jacobian import dense_temporal_jacobian
from better_robot.residuals.structure import TemporalPattern


@pytest.fixture(scope="module")
def two_joint_model():
    pose = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    builder = ModelBuilder("temporal_structure")
    builder.add_body("base", mass=1.0)
    builder.add_body("first", mass=1.0)
    builder.add_body("second", mass=1.0)
    builder.add_revolute_z(
        "joint_1",
        parent="base",
        child="first",
        origin=pose,
        lower=-math.pi,
        upper=math.pi,
    )
    builder.add_revolute_y(
        "joint_2",
        parent="first",
        child="second",
        origin=pose,
        lower=-math.pi,
        upper=math.pi,
    )
    builder.add_frame("first_tip", parent_body="first", placement=pose)
    builder.add_frame("second_tip", parent_body="second", placement=pose)
    return build_model(builder.finalize(), dtype=torch.float64)


def _trajectory(model, batch_shape=(2, 3), horizon: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(8)
    neutral = model.q_neutral.expand(*batch_shape, horizon, model.nq).clone()
    tangent = (
        torch.randn(
            *batch_shape,
            horizon,
            model.nv,
            dtype=neutral.dtype,
            generator=generator,
        )
        * 0.03
    )
    return model.integrate(neutral, tangent)


def _robot_trajectory(
    model,
    tensor: torch.Tensor,
    *,
    name: str = "q",
) -> RobotVariable:
    return RobotVariable(model, tensor, name=name, time_axis=0)


def _trajectory_residual(model, q: RobotVariable, kind: str):
    if kind == "velocity":
        return VelocityResidual(q, dt=0.1, weight=0.7)
    if kind == "acceleration":
        return AccelerationResidual(q, dt=0.1, weight=0.4)
    return ReferenceTrajectoryResidual(
        q,
        model.q_neutral.expand(5, -1).clone(),
        weight=0.3,
        weight_per_frame=torch.linspace(0.5, 1.0, 5, dtype=q.tensor.dtype),
    )


@pytest.mark.parametrize(
    ("kind", "expected"),
    (
        ("velocity", TemporalPattern(3, 2, 1, (-1, 1))),
        ("acceleration", TemporalPattern(3, 2, 1, (-1, 0, 1))),
        ("reference", TemporalPattern(5, 2, 0, (0,))),
    ),
)
def test_smoothness_and_reference_named_blocks_match_dense_oracle(
    two_joint_model,
    kind: str,
    expected: TemporalPattern,
) -> None:
    model = two_joint_model
    q = _robot_trajectory(model, _trajectory(model))
    residual = _trajectory_residual(model, q, kind)

    assert residual.variables[0] is q
    assert residual.temporal_structure(q) == expected
    assert residual.temporal_structure("q") == expected
    assert residual.temporal_structure("other") is None
    result = residual.error()
    assert result.shape == (*q.tensor.shape[:-2], residual.dim)

    temporal = residual.temporal_jacobian_blocks(q)
    assert tuple(temporal) == expected.offsets
    assert all(
        block.shape == (*q.tensor.shape[:-2], expected.rows, expected.row_width, model.nv)
        for block in temporal.values()
    )
    densified = dense_temporal_jacobian(expected, temporal, horizon=5)
    torch.testing.assert_close(residual.jacobian()[0], densified)


def test_reference_trajectory_uses_static_variable_and_central_weight(two_joint_model) -> None:
    model = two_joint_model
    q = _robot_trajectory(model, _trajectory(model, batch_shape=()))
    reference = Variable(q.tensor.clone(), name="reference", trainable=False)
    frame_weight = torch.linspace(0.5, 1.0, q.time_length, dtype=q.tensor.dtype)
    residual = ReferenceTrajectoryResidual(q, reference, weight=0.3, weight_per_frame=frame_weight)

    assert residual.variables == (q, reference)
    assert residual.temporal_structure(q) == TemporalPattern(q.time_length, model.nv, 0, (0,))
    torch.testing.assert_close(residual.error(), torch.zeros_like(residual.error()))
    torch.testing.assert_close(residual.weighted_error(), residual.error() * 0.3)
    assert residual.jacobian()[0].shape == (q.time_length * model.nv, q.free_dim)


def test_rest_residual_reads_static_reference_object(two_joint_model) -> None:
    model = two_joint_model
    tangent = torch.tensor([0.2, -0.1], dtype=torch.float64)
    q = RobotVariable(model, model.integrate(model.q_neutral, tangent), name="q")
    rest = Variable(model.q_neutral.clone(), name="rest", trainable=False)
    residual = RestResidual(q, rest, weight=0.5)

    torch.testing.assert_close(residual.error(), tangent, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(residual.weighted_error(), tangent * 0.5, atol=1e-12, rtol=1e-12)
    assert residual.jacobian()[0].shape == (model.nv, q.free_dim)


def test_constant_temporal_blocks_honor_create_graph(two_joint_model) -> None:
    model = two_joint_model
    q = _robot_trajectory(model, _trajectory(model, batch_shape=()).requires_grad_())
    residual = VelocityResidual(q, dt=0.1)
    block = residual.temporal_jacobian_blocks(q)[-1]

    assert block.requires_grad
    gradient = torch.autograd.grad(block.sum(), q.tensor)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(q.tensor))


def test_time_indexed_named_mapping_slices_arbitrary_batches(two_joint_model) -> None:
    model = two_joint_model
    q = _robot_trajectory(model, _trajectory(model))
    frame_id = model.frame_id("second_tip")
    target = forward_kinematics(model, model.q_neutral, compute_frames=True).frame_pose_world[frame_id]
    inner = PositionResidual(q, frame_id=frame_id, target=target)
    residual = TimeIndexedResidual(inner, 2)

    pattern = TemporalPattern(1, 3, 2, (0,))
    assert residual.temporal_structure(q) == pattern
    assert residual.temporal_structure("q") == pattern
    result = residual.error()
    assert result.shape == (*q.tensor.shape[:-2], 3)
    temporal = residual.temporal_jacobian_blocks(q)
    assert temporal[0].shape == (*q.tensor.shape[:-2], 1, 3, model.nv)
    dense = dense_temporal_jacobian(pattern, temporal, horizon=5)
    torch.testing.assert_close(residual.jacobian()[0], dense)


def test_contact_named_blocks_match_dense(two_joint_model) -> None:
    model = two_joint_model
    q = _robot_trajectory(model, _trajectory(model))
    frames = (model.frame_id("first_tip"), model.frame_id("second_tip"))
    weights = torch.tensor(
        [[1.0, 0.0], [1.0, 0.5], [0.7, 1.0], [0.2, 1.0], [0.0, 0.8]],
        dtype=q.tensor.dtype,
    )
    residual = ContactConsistencyResidual(q, frames, weights, dt=0.05, weight=0.3)
    pattern = TemporalPattern(4, 6, 0, (0, 1))

    assert residual.variables == ()
    assert residual.temporal_structure(q) == pattern
    result = residual.error()
    assert result.shape == (*q.tensor.shape[:-2], residual.dim)
    temporal = residual.temporal_jacobian_blocks(q)
    assert all(block.shape == (*q.tensor.shape[:-2], 4, 6, model.nv) for block in temporal.values())
    dense = dense_temporal_jacobian(pattern, temporal, horizon=5)
    analytic = residual.jacobian()
    assert analytic is not None
    torch.testing.assert_close(analytic[0], dense)


def test_contact_temporal_blocks_match_tangent_finite_difference(two_joint_model) -> None:
    model = two_joint_model
    q_tensor = _trajectory(model, batch_shape=())
    q = _robot_trajectory(model, q_tensor)
    frames = (model.frame_id("first_tip"), model.frame_id("second_tip"))
    weights = torch.linspace(0.2, 1.0, 10, dtype=q.tensor.dtype).reshape(5, 2)
    residual = ContactConsistencyResidual(q, frames, weights, dt=0.1, weight=0.6)
    problem = Problem([residual])
    analytic = problem.dense_jacobian(strategy="analytic")
    finite_difference = torch.zeros_like(analytic)
    epsilon = 1e-6
    for knot in range(5):
        for coordinate in range(model.nv):
            delta = torch.zeros(5, model.nv, dtype=q.tensor.dtype)
            delta[knot, coordinate] = epsilon
            problem.update({q.name: model.integrate(q_tensor, delta)})
            plus = problem.error()
            problem.update({q.name: model.integrate(q_tensor, -delta)})
            minus = problem.error()
            finite_difference[:, knot * model.nv + coordinate] = (plus - minus) / (2.0 * epsilon)
    problem.update({q.name: q_tensor})

    torch.testing.assert_close(analytic, finite_difference, rtol=2e-5, atol=2e-7)


def test_contact_horizon_validation_is_eager(two_joint_model) -> None:
    model = two_joint_model
    short_q = _robot_trajectory(
        model,
        model.q_neutral.expand(1, model.nq).clone(),
        name="short_q",
    )
    with pytest.raises(ValueError, match="at least two timesteps"):
        ContactConsistencyResidual(
            short_q,
            (model.frame_id("first_tip"),),
            torch.ones(1, 1),
            dt=0.1,
        )


def test_temporal_horizon_validation_is_eager_and_variable_owned(two_joint_model) -> None:
    model = two_joint_model
    short = _robot_trajectory(model, model.q_neutral.expand(2, -1).clone(), name="short")
    with pytest.raises(ValueError, match="at least 3"):
        VelocityResidual(short, dt=0.1)
    with pytest.raises(ValueError, match="at least 3"):
        AccelerationResidual(short, dt=0.1)
    with pytest.raises(ValueError, match="at least one timestep"):
        _robot_trajectory(model, torch.empty(0, model.nq), name="empty")

    q = _robot_trajectory(model, model.q_neutral.expand(3, -1).clone())
    target = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=q.tensor.dtype)
    inner = PositionResidual(q, frame_id=0, target=target)
    with pytest.raises(ValueError, match="must index trajectory length"):
        TimeIndexedResidual(inner, 3)


def test_custom_time_local_residual_falls_back_to_ad(two_joint_model) -> None:
    model = two_joint_model
    q = _robot_trajectory(model, _trajectory(model, batch_shape=()))

    class QuadraticResidual(Residual):
        def __init__(self, variable: RobotVariable, t_idx: int) -> None:
            self.q = variable
            self.t_idx = t_idx
            super().__init__(variable, dim=1, name="quadratic")

        def error(self) -> torch.Tensor:
            return self.q.tensor[..., self.t_idx, :1].square()

    residual = QuadraticResidual(q, 2)
    problem = Problem([residual])
    auto = problem.jacobian_blocks(strategy="auto")[(residual.name, q.name)]
    finite_difference = problem.jacobian_blocks(
        strategy="finite_difference",
        fd_eps=1e-6,
    )[(residual.name, q.name)]

    torch.testing.assert_close(auto, finite_difference, rtol=2e-5, atol=2e-7)
    with pytest.raises(TypeError, match="built-in knot error and Jacobian primitives"):
        TimeIndexedResidual(residual, 2)
