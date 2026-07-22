"""Object-referenced trajectory smoothness residuals."""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.optim import RobotVariable
from better_robot.residuals import AccelerationResidual, VelocityResidual


@pytest.fixture(scope="module")
def panda_model():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415

    return br.load(panda_description.URDF_PATH, dtype=torch.float64)


def _variable(model, tensor: torch.Tensor) -> RobotVariable:
    return RobotVariable(model, tensor, name="q", time_axis=0)


def test_acceleration_zero_on_linear_trajectory(panda_model) -> None:
    horizon = 10
    q0 = panda_model.q_neutral.double()
    q1 = q0.clone()
    q1[0] = 0.5
    alpha = torch.linspace(0, 1, horizon, dtype=torch.float64).unsqueeze(1)
    q = _variable(panda_model, q0 * (1 - alpha) + q1 * alpha)

    error = AccelerationResidual(q, dt=0.1).error()

    assert error.abs().max() < 1e-12


def test_acceleration_nonzero_on_perturbed_trajectory(panda_model) -> None:
    generator = torch.Generator().manual_seed(42)
    tensor = panda_model.q_neutral.double().expand(10, -1).clone()
    tensor += torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * 0.1
    residual = AccelerationResidual(_variable(panda_model, tensor), dt=0.05)

    assert torch.linalg.vector_norm(residual.error()) > 1.0


def test_velocity_preserves_arbitrary_batch_axes(panda_model) -> None:
    tensor = panda_model.q_neutral.double().expand(2, 3, 6, -1).clone()
    residual = VelocityResidual(_variable(panda_model, tensor), dt=0.1, row_weight=0.25)

    assert residual.error().shape == (2, 3, 4 * panda_model.nv)
    torch.testing.assert_close(residual.error(), torch.zeros_like(residual.error()))
    torch.testing.assert_close(residual.weighted_error(), residual.error() * 0.25)


@pytest.mark.parametrize("residual_type", [VelocityResidual, AccelerationResidual])
def test_analytic_jacobian_matches_tangent_finite_difference(panda_model, residual_type) -> None:
    generator = torch.Generator().manual_seed(3)
    horizon = 6
    tensor = panda_model.q_neutral.double().expand(horizon, -1).clone()
    tensor += torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * 0.02
    q = _variable(panda_model, tensor)
    residual = residual_type(q, dt=0.1)
    analytic = residual.jacobian()[0]
    finite_difference = torch.zeros_like(analytic)
    epsilon = 1e-6

    for knot in range(horizon):
        for coordinate in range(panda_model.nv):
            delta = torch.zeros(horizon, panda_model.nv, dtype=tensor.dtype)
            delta[knot, coordinate] = epsilon
            q.tensor = panda_model.integrate(tensor, delta)
            plus = residual.error()
            q.tensor = panda_model.integrate(tensor, -delta)
            minus = residual.error()
            finite_difference[:, knot * panda_model.nv + coordinate] = (plus - minus) / (2.0 * epsilon)
    q.tensor = tensor

    torch.testing.assert_close(analytic, finite_difference, atol=1e-6, rtol=1e-4)


def test_autograd_through_acceleration_difference(panda_model) -> None:
    horizon = 6
    initial = panda_model.q_neutral.double().expand(horizon, -1).clone()
    delta = torch.zeros(horizon, panda_model.nv, dtype=torch.float64, requires_grad=True)
    q = _variable(panda_model, panda_model.integrate(initial, delta))

    loss = AccelerationResidual(q, dt=0.1).error().square().sum()
    gradient = torch.autograd.grad(loss, delta)[0]

    assert torch.isfinite(gradient).all()


def test_constructor_derives_and_validates_trajectory_structure(panda_model) -> None:
    short = _variable(panda_model, panda_model.q_neutral.expand(2, -1).clone())
    point = RobotVariable(panda_model, panda_model.q_neutral.clone(), name="point")

    with pytest.raises(ValueError, match="at least 3"):
        VelocityResidual(short, dt=0.1)
    with pytest.raises(ValueError, match="time_axis=0"):
        AccelerationResidual(point, dt=0.1)
    with pytest.raises(ValueError, match="finite and positive"):
        VelocityResidual(_variable(panda_model, panda_model.q_neutral.expand(4, -1).clone()), dt=0.0)
