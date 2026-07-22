"""Object-referenced trajectory smoothness residuals."""

from __future__ import annotations

from functools import partial
import math

import pytest
import torch

import better_robot as br
from better_robot.data_model.joint_models import JointComposite, JointPX, JointRX
from better_robot.io import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.optim import AutodiffFallbackWarning, LevenbergMarquardt, Problem, RobotVariable
from better_robot.optim.temporal import LinearizationReason
from better_robot.residuals import SmoothnessResidual, VelocityResidual


@pytest.fixture(scope="module")
def panda_model():
    panda_description = pytest.importorskip("robot_descriptions.panda_description")
    return br.load(panda_description.URDF_PATH, dtype=torch.float32)


@pytest.fixture(scope="module")
def scalar_model():
    builder = ModelBuilder("scalar_smoothness")
    base = builder.add_body("base")
    middle = builder.add_body("middle")
    tip = builder.add_body("tip")
    builder.add_revolute_z("first", parent=base, child=middle)
    builder.add_prismatic_x("second", parent=middle, child=tip)
    return build_model(builder.finalize(), dtype=torch.float32)


@pytest.fixture(scope="module")
def spherical_model():
    builder = ModelBuilder("spherical_smoothness")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_spherical("ball", parent=base, child=tip)
    return build_model(builder.finalize(), dtype=torch.float32)


def _variable(model, tensor: torch.Tensor) -> RobotVariable:
    return RobotVariable(model, tensor, name="q", time_axis=0)


def _scalar_trajectory(model, *, horizon: int = 7, batch_shape: tuple[int, ...] = ()) -> torch.Tensor:
    values = torch.linspace(-0.25, 0.4, horizon)
    tangent = torch.stack((values.square() + 0.1 * values, values.sin()), dim=-1)
    tangent = tangent.expand(*batch_shape, horizon, model.nv).clone()
    return model.integrate(model.q_neutral.expand_as(tangent), tangent)


@pytest.mark.parametrize("order", (2, 3, 4))
def test_orders_match_hand_computed_forward_stencils(scalar_model, order: int) -> None:
    dt = 0.2
    tensor = _scalar_trajectory(scalar_model)
    q = _variable(scalar_model, tensor)
    coefficients = [float((-1) ** (order - index) * math.comb(order, index)) for index in range(order + 1)]
    expected = (
        sum(
            coefficient * tensor[index : tensor.shape[0] - order + index]
            for index, coefficient in enumerate(coefficients)
        )
        / dt**order
    )

    actual = SmoothnessResidual(q, order=order, dt=dt).error().reshape_as(expected)

    torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-5)


def test_order_two_is_bitwise_legacy_acceleration_on_panda(panda_model) -> None:
    generator = torch.Generator().manual_seed(3)
    tensor = panda_model.q_neutral.expand(6, -1).clone()
    tensor += torch.randn(tensor.shape, generator=generator) * 0.02
    q = _variable(panda_model, tensor)
    forward = panda_model.difference(tensor[1:-1], tensor[2:])
    backward = panda_model.difference(tensor[:-2], tensor[1:-1])
    expected = ((forward - backward) / 0.1**2).reshape(-1)

    actual = SmoothnessResidual(q, order=2, dt=0.1).error()

    assert torch.equal(actual, expected)


def test_coordinate_weight_repeats_per_knot_row(scalar_model) -> None:
    q = _variable(scalar_model, _scalar_trajectory(scalar_model, batch_shape=(2, 3)))
    coordinate_weight = torch.tensor([0.25, 1.5])
    residual = SmoothnessResidual(q, order=3, dt=0.1, coordinate_weight=coordinate_weight)
    rows = residual.error().reshape(2, 3, q.time_length - 3, scalar_model.nv)

    torch.testing.assert_close(
        residual.weighted_error().reshape_as(rows),
        rows * coordinate_weight,
    )
    coordinate_weight.add_(0.5)
    torch.testing.assert_close(
        residual.weighted_error().reshape_as(rows),
        rows * coordinate_weight,
    )
    problem = Problem([residual])
    dense = problem.dense_jacobian(strategy="analytic")
    structured = problem.structured_normal()
    torch.testing.assert_close(structured.normal.densify(), dense.mT @ dense)


def test_velocity_preserves_arbitrary_batch_axes(scalar_model) -> None:
    q = _variable(scalar_model, _scalar_trajectory(scalar_model, horizon=6, batch_shape=(2, 3)))
    residual = VelocityResidual(q, dt=0.1, row_weight=0.25)

    assert residual.error().shape == (2, 3, 4 * scalar_model.nv)
    torch.testing.assert_close(residual.weighted_error(), residual.error() * 0.25)


@pytest.mark.parametrize("strategy", ("jacrev", "jacfwd"))
@pytest.mark.parametrize("order", (2, 3, 4))
def test_scalar_analytic_blocks_match_explicit_ad(scalar_model, order: int, strategy: str) -> None:
    q = _variable(scalar_model, _scalar_trajectory(scalar_model, horizon=7))
    problem = Problem([SmoothnessResidual(q, order=order, dt=0.1)])

    analytic = problem.dense_jacobian(strategy="analytic")
    autodiff = problem.dense_jacobian(strategy=strategy)

    torch.testing.assert_close(analytic, autodiff, atol=3.0e-4, rtol=3.0e-5)


def test_scalar_composite_keeps_analytic_blocks() -> None:
    builder = ModelBuilder("composite_scalar_smoothness")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_joint("compound", kind=JointComposite((JointRX(), JointPX())), parent=base, child=tip)
    model = build_model(builder.finalize(), dtype=torch.float32)
    tangent = torch.linspace(-0.2, 0.3, 12).reshape(6, model.nv)
    q = _variable(model, model.integrate(model.q_neutral.expand(6, -1), tangent))
    residual = SmoothnessResidual(q, order=2, dt=0.1)

    assert callable(residual.temporal_jacobian_blocks)
    assert residual.jacobian() is not None


@pytest.mark.parametrize(
    "factory",
    (
        pytest.param(partial(VelocityResidual, dt=0.1), id="velocity"),
        pytest.param(partial(SmoothnessResidual, order=2, dt=0.1), id="smoothness"),
    ),
)
def test_spherical_models_warn_and_route_dense(spherical_model, factory) -> None:
    tangent = torch.tensor(
        [
            [0.35, -0.20, 0.12],
            [-0.15, 0.28, 0.22],
            [0.25, 0.18, -0.30],
            [-0.32, 0.10, 0.27],
            [0.18, -0.35, 0.15],
            [-0.22, 0.30, -0.16],
        ]
    )
    tensor = spherical_model.integrate(spherical_model.q_neutral.expand(6, -1), tangent)
    residual = factory(_variable(spherical_model, tensor))
    problem = Problem([residual])
    optimizer = LevenbergMarquardt(problem)

    assert residual.jacobian() is None
    assert not callable(getattr(residual, "temporal_jacobian_blocks", None))
    assert problem.temporal_analysis.reason is LinearizationReason.MISSING_TEMPORAL_BLOCKS
    with pytest.warns(
        AutodiffFallbackWarning,
        match="spherical_smoothness.*spherical.*dense autodiff",
    ) as caught:
        decision = optimizer.resolve_linearization(problem)
        problem.dense_jacobian()
    assert len(caught) == 1
    assert decision.used == "dense"


def test_constructor_validates_order_horizon_weight_and_trajectory(scalar_model) -> None:
    trajectory = _variable(scalar_model, _scalar_trajectory(scalar_model, horizon=5))
    point = RobotVariable(scalar_model, scalar_model.q_neutral.clone(), name="point")

    with pytest.raises(ValueError, match="order must be one of"):
        SmoothnessResidual(trajectory, order=1, dt=0.1)
    with pytest.raises(ValueError, match="at least 5 timesteps"):
        SmoothnessResidual(_variable(scalar_model, trajectory.tensor[:4]), order=4, dt=0.1)
    with pytest.raises(ValueError, match="exact shape"):
        SmoothnessResidual(trajectory, order=2, dt=0.1, coordinate_weight=torch.ones(2, scalar_model.nv))
    with pytest.raises(ValueError, match="time_axis=0"):
        SmoothnessResidual(point, order=2, dt=0.1)
    with pytest.raises(ValueError, match="finite and positive"):
        VelocityResidual(trajectory, dt=0.0)
