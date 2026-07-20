"""Named variable blocks, reduced coordinates, and dense assembly."""

from __future__ import annotations

import pytest
import torch

from better_robot.lie import so3
from better_robot.optim import LevenbergMarquardt, Problem, Residual, SO3Variable, Variable, residual

_POSITION_TARGET = torch.tensor([1.0, -2.0, 0.5])
_ROTATION_TARGET = torch.tensor([0.2, -0.1, 0.3])


def _identity_rotation(*batch_shape: int) -> torch.Tensor:
    value = torch.zeros(*batch_shape, 4)
    value[..., 3] = 1.0
    return value


class _PositionResidual(Residual):
    def __init__(self, x: Variable, *, name: str = "position", weight=1.0, group_size: int = 1) -> None:
        self.x = x
        super().__init__(x, dim=3, name=name, weight=weight, group_size=group_size)

    def error(self) -> torch.Tensor:
        return self.x.tensor - _POSITION_TARGET.to(self.x.tensor)


class _AttitudeResidual(Residual):
    def __init__(self, rotation: SO3Variable) -> None:
        self.rotation = rotation
        super().__init__(rotation, dim=3, name="attitude")

    def error(self) -> torch.Tensor:
        return so3.log(self.rotation.tensor) - _ROTATION_TARGET.to(self.rotation.tensor)


class _ParameterResidual(Residual):
    def __init__(self, x: Variable, target: Variable, *extra: Variable) -> None:
        self.x, self.target = x, target
        super().__init__(x, target, *extra, dim=3, name="parameter")

    def error(self) -> torch.Tensor:
        return self.x.tensor - self.target.tensor


class _PromotingResidual(_PositionResidual):
    def error(self) -> torch.Tensor:
        return self.x.tensor.double()


class _PromotingAnalyticResidual(_PositionResidual):
    def jacobian(self):
        return (torch.eye(3, device=self.x.tensor.device, dtype=torch.float64),)


class _DetachedAnalyticResidual(_PositionResidual):
    def error(self) -> torch.Tensor:
        return self.x.tensor.square()

    def jacobian(self):
        return (torch.diag_embed(2.0 * self.x.tensor).detach(),)


class _PerElementInvalidResidual(Residual):
    def __init__(self, x: Variable) -> None:
        self.x = x
        super().__init__(x, dim=1, name="validity")

    def error(self) -> torch.Tensor:
        return torch.where(self.x.tensor > 0, torch.full_like(self.x.tensor, torch.nan), self.x.tensor)


def _two_block_problem(x_value: torch.Tensor, rotation_value: torch.Tensor):
    x = Variable(x_value, name="x", batch_ndim=max(0, x_value.ndim - 1))
    rotation = SO3Variable(rotation_value, name="rotation")
    return x, rotation, Problem([_PositionResidual(x), _AttitudeResidual(rotation)])


def test_builder_solves_the_public_line_fit_example() -> None:
    x_data = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y_data = torch.tensor([1.0, 3.0, 5.0, 7.0])
    theta = Variable(torch.zeros(2), name="theta")

    @residual(theta, dim=4)
    def fit(value: torch.Tensor) -> torch.Tensor:
        return value[0:1] * x_data + value[1:2] - y_data

    problem = Problem()
    assert problem.add_residual(fit) is fit
    LevenbergMarquardt(problem, max_iterations=20).optimize()
    assert fit.name == "fit"
    torch.testing.assert_close(theta.tensor, torch.tensor([2.0, 1.0]), rtol=1e-4, atol=1e-4)


def test_two_block_shapes_gradient_and_structural_zeros() -> None:
    x_value = torch.tensor([0.5, -1.0, 2.0])
    x, rotation, problem = _two_block_problem(x_value, _identity_rotation())
    output = problem.error()
    gradient = problem.gradient()
    blocks = problem.jacobian_blocks()
    torch.testing.assert_close(output, torch.cat((x_value - _POSITION_TARGET, -_ROTATION_TARGET)))
    torch.testing.assert_close(gradient["x"], x_value - _POSITION_TARGET)
    torch.testing.assert_close(gradient["rotation"], -_ROTATION_TARGET)
    assert x.tensor.shape == (3,) and rotation.tensor.shape == (4,)
    assert set(blocks) == {("position", "x"), ("attitude", "rotation")}
    assert blocks[("position", "x")].shape == blocks[("attitude", "rotation")].shape == (3, 3)


def test_two_block_batched_evaluation_matches_sequential() -> None:
    x_values = torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.0, -0.5], [-0.2, 0.3, 0.7]])
    _x, _rotation, problem = _two_block_problem(x_values, _identity_rotation(3))
    output, gradient = problem.error(), problem.gradient()
    for index in range(3):
        _sx, _sr, sequential = _two_block_problem(x_values[index], _identity_rotation())
        torch.testing.assert_close(output[index], sequential.error())
        for name, value in sequential.gradient().items():
            torch.testing.assert_close(gradient[name][index], value)


def test_dense_assembly_has_exact_declared_offsets() -> None:
    x = Variable(torch.tensor([0.2, 8.0, -0.7]), name="x")
    rotation = SO3Variable(_identity_rotation(), name="rotation")
    problem = Problem([_AttitudeResidual(rotation), _PositionResidual(x)])
    problem._freeze()
    assert problem.row_offsets == {"attitude": slice(0, 3), "position": slice(3, 6)}
    assert problem.column_offsets == {"rotation": slice(0, 3), "x": slice(3, 6)}
    expected = torch.zeros(6, 6)
    expected[:3, :3] = torch.eye(3)
    expected[3:, 3:] = torch.eye(3)
    torch.testing.assert_close(problem.dense_jacobian(), expected)


def test_tensor_weight_must_match_working_dtype() -> None:
    x = Variable(torch.ones(2, 3), name="x", batch_ndim=1)
    problem = Problem([_PositionResidual(x, weight=torch.ones(2, dtype=torch.float64))])
    with pytest.raises(ValueError, match="must preserve working dtype/device"):
        problem.error()


def test_static_variables_are_harvested_and_graph_visible() -> None:
    target_tensor = torch.tensor([0.2, -0.3, 0.5], requires_grad=True)
    x = Variable(torch.tensor([0.7, -0.1, 0.8]), name="x")
    target = Variable(target_tensor, name="target", trainable=False)
    problem = Problem([_ParameterResidual(x, target)])
    gradient = problem.gradient(create_graph=True)["x"]
    assert problem.variables["target"] is target and not target.trainable
    torch.testing.assert_close(torch.autograd.grad(gradient.sum(), target_tensor)[0], -torch.ones(3))


def test_unused_gradient_block_stays_connected_to_static_variable() -> None:
    target_tensor = torch.tensor([0.2, -0.3, 0.5], requires_grad=True)
    x = Variable(torch.tensor([0.7, -0.1, 0.8]), name="x")
    unused = Variable(torch.tensor([2.0]), name="unused")
    target = Variable(target_tensor, name="target", trainable=False)
    problem = Problem([_ParameterResidual(x, target, unused)])
    unused_gradient = problem.gradient(create_graph=True)["unused"]
    torch.testing.assert_close(unused_gradient, torch.zeros(1))
    torch.testing.assert_close(torch.autograd.grad(unused_gradient.sum(), target_tensor)[0], torch.zeros(3))


def test_working_dtype_fails_fast() -> None:
    x = Variable(torch.ones(3), name="x")
    promoting = Problem([_PromotingResidual(x)])
    with pytest.raises(ValueError, match="must preserve working dtype/device"):
        promoting.error()
    for strategy in ("jacrev", "jacfwd", "finite_difference"):
        with pytest.raises(ValueError, match="must preserve working dtype/device"):
            promoting.jacobian_blocks(strategy=strategy)

    analytic = Problem([_PromotingAnalyticResidual(x)])
    with pytest.raises(ValueError, match="Analytic block.*working dtype/device"):
        analytic.jacobian_blocks(strategy="analytic")

    graph_x = Variable(torch.ones(3, requires_grad=True), name="graph_x")
    detached = Problem([_DetachedAnalyticResidual(graph_x, name="detached_analytic")])
    with pytest.raises(ValueError, match="cannot honor create_graph=True"):
        detached.jacobian_blocks(strategy="analytic", create_graph=True)


def test_robust_group_size_must_partition_static_dimension() -> None:
    x = Variable(torch.zeros(3))
    with pytest.raises(ValueError, match="positive divisor"):
        _PositionResidual(x, group_size=2)
    item = _PositionResidual(x, group_size=3)
    assert item.group_size == 3


def test_invalid_batch_element_uses_nan_rows_without_raising() -> None:
    x = Variable(torch.tensor([[-1.0], [1.0]]), name="x", batch_ndim=1)
    output = Problem([_PerElementInvalidResidual(x)]).error()
    torch.testing.assert_close(output[0], torch.tensor([-1.0]))
    assert torch.isnan(output[1]).all()
