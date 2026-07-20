"""Contracts for residuals and square-root-information weights."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim.variables import Variable
from better_robot.residuals.base import DiagonalWeight, Difference, Residual, ScaleWeight, residual


class _SquareResidual(Residual):
    def __init__(self, variable: Variable, **kwargs) -> None:
        self.variable = variable
        super().__init__(variable, dim=2, **kwargs)

    def error(self) -> torch.Tensor:
        return self.variable.tensor.square()

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        return (torch.diag_embed(2.0 * self.variable.tensor),)


def test_scale_weight_multiplies_error_and_jacobian_rows_identically() -> None:
    variable = Variable(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), batch_ndim=1)
    item = _SquareResidual(variable, weight=ScaleWeight(torch.tensor([2.0, 3.0])))
    raw = item.error()
    blocks = item.jacobian()
    assert blocks is not None

    torch.testing.assert_close(item.weighted_error(), raw * torch.tensor([[2.0], [3.0]]))
    torch.testing.assert_close(
        item.weight.apply_jacobian(blocks)[0],
        blocks[0] * torch.tensor([2.0, 3.0])[:, None, None],
    )


def test_diagonal_weight_multiplies_the_same_rows() -> None:
    variable = Variable(torch.tensor([1.0, 2.0]))
    item = _SquareResidual(variable, weight=DiagonalWeight(torch.tensor([2.0, 4.0])))
    block = item.jacobian()
    assert block is not None

    torch.testing.assert_close(item.weighted_error(), torch.tensor([2.0, 16.0]))
    torch.testing.assert_close(
        item.weight.apply_jacobian(block)[0],
        torch.tensor([[4.0, 0.0], [0.0, 16.0]]),
    )


def test_python_zero_is_inactive_but_tensor_zero_stays_graph_visible() -> None:
    assert ScaleWeight(0.0).is_inactive()
    assert not ScaleWeight(torch.tensor(0.0)).is_inactive()


def test_residual_decorator_supports_decorator_and_direct_forms() -> None:
    variable = Variable(torch.tensor([2.0]), name="x")

    @residual(variable, dim=1)
    def shifted(x: torch.Tensor) -> torch.Tensor:
        return x - 1.0

    direct = residual(lambda x: x + 1.0, variable, dim=1, name="direct")

    torch.testing.assert_close(shifted.error(), torch.tensor([1.0]))
    torch.testing.assert_close(direct.error(), torch.tensor([3.0]))
    assert shifted.variables == direct.variables == (variable,)


def test_difference_uses_the_variable_geometry() -> None:
    variable = Variable(torch.tensor([2.0, -1.0]))
    item = Difference(variable, torch.tensor([0.5, 0.25]))

    torch.testing.assert_close(item.error(), torch.tensor([1.5, -1.25]))


@pytest.mark.parametrize("group_size", (0, 3))
def test_group_size_must_be_a_positive_divisor(group_size: int) -> None:
    variable = Variable(torch.ones(2))
    with pytest.raises(ValueError, match="positive divisor"):
        _SquareResidual(variable, group_size=group_size)


def test_weight_shape_and_working_type_are_validated_at_application() -> None:
    variable = Variable(torch.ones(2))
    item = _SquareResidual(variable, weight=torch.ones(3))
    with pytest.raises(ValueError, match="ScaleWeight tensor must be scalar"):
        item.weighted_error()

    item.weight = DiagonalWeight(torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="preserve working dtype/device"):
        item.weighted_error()
