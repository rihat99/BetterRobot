"""Direct-variable temporal and regularization residual contracts."""

from __future__ import annotations

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.optim import RobotVariable, Variable
from better_robot.residuals import (
    AccelerationResidual,
    ReferenceTrajectoryResidual,
    RestResidual,
    VelocityResidual,
)
from better_robot.residuals._temporal_jacobian import dense_temporal_jacobian
from better_robot.residuals.structure import TemporalPattern


@pytest.fixture(scope="module")
def model():
    placement = torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    builder = ModelBuilder("temporal_v2")
    base = builder.add_body("base", mass=1.0)
    first = builder.add_body("first", mass=1.0)
    second = builder.add_body("second", mass=1.0)
    builder.add_revolute_z("joint_1", parent=base, child=first, origin=placement)
    builder.add_revolute_y("joint_2", parent=first, child=second, origin=placement)
    return build_model(builder.finalize(), dtype=torch.float64)


def _trajectory(model) -> RobotVariable:
    horizon = 5
    tangent = torch.linspace(-0.08, 0.08, horizon * model.nv, dtype=torch.float64).reshape(horizon, model.nv)
    tensor = model.integrate(model.q_neutral.expand(horizon, -1), tangent)
    return RobotVariable(model, tensor, name="q", time_axis=0)


@pytest.mark.parametrize(
    ("residual_type", "expected"),
    [
        (VelocityResidual, TemporalPattern(3, 2, 1, (-1, 1))),
        (AccelerationResidual, TemporalPattern(3, 2, 1, (-1, 0, 1))),
    ],
)
def test_smoothness_hooks_accept_variable_objects_and_names(model, residual_type, expected) -> None:
    q = _trajectory(model)
    residual = residual_type(q, dt=0.1)

    assert residual.temporal_structure(q) == expected
    assert residual.temporal_structure("q") == expected
    assert residual.temporal_structure("other") is None
    blocks = residual.temporal_jacobian_blocks(q)
    assert tuple(blocks) == expected.offsets
    dense = dense_temporal_jacobian(expected, blocks, horizon=q.time_length)
    torch.testing.assert_close(residual.jacobian()[0], dense)


def test_reference_trajectory_uses_static_variable_and_central_weight(model) -> None:
    q = _trajectory(model)
    reference = Variable(q.tensor.clone(), name="reference", trainable=False)
    frame_weight = torch.linspace(0.5, 1.0, q.time_length, dtype=q.tensor.dtype)
    residual = ReferenceTrajectoryResidual(q, reference, weight=0.3, weight_per_frame=frame_weight)

    assert residual.variables == (q, reference)
    assert residual.temporal_structure(q) == TemporalPattern(q.time_length, model.nv, 0, (0,))
    torch.testing.assert_close(residual.error(), torch.zeros_like(residual.error()))
    torch.testing.assert_close(residual.weighted_error(), residual.error() * 0.3)
    assert residual.jacobian()[0].shape == (q.time_length * model.nv, q.free_dim)


def test_rest_residual_reads_static_reference_object(model) -> None:
    tangent = torch.tensor([0.2, -0.1], dtype=torch.float64)
    q = RobotVariable(model, model.integrate(model.q_neutral, tangent), name="q")
    rest = Variable(model.q_neutral.clone(), name="rest", trainable=False)
    residual = RestResidual(q, rest, weight=0.5)

    torch.testing.assert_close(residual.error(), tangent, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(residual.weighted_error(), tangent * 0.5, atol=1e-12, rtol=1e-12)
    assert residual.jacobian()[0].shape == (model.nv, q.free_dim)
