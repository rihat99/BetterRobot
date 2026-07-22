"""Scalar objective adapter contracts."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim import LevenbergMarquardt, Problem, ScalarCost, TorchOptimizer, Variable
from better_robot.residuals import Node


class _ScaleNode(Node):
    def __init__(self, variable: Variable, scale: float) -> None:
        self.variable = variable
        self.scale = scale
        super().__init__(variable)

    def compute(self) -> torch.Tensor:
        return self.variable.tensor * self.scale


def test_scalar_cost_rejects_non_variable_and_non_node_reads() -> None:
    with pytest.raises(TypeError, match="Variables or Nodes, got NoneType"):
        ScalarCost(lambda value: value, None)  # type: ignore[arg-type]


def test_scalar_cost_objective_is_exact_weight_times_function_with_node_read() -> None:
    value = Variable(torch.tensor([[1.0, -2.0], [0.5, 3.0]]), batch_ndim=1, name="value")
    node = _ScaleNode(value, 0.5)
    weight = torch.tensor([0.25, 1.5])
    item = ScalarCost(lambda scaled: scaled.square().sum(dim=-1) + 0.3, node, weight=weight, name="energy")
    problem = Problem([item])

    expected_function = node.value().square().sum(dim=-1) + 0.3
    torch.testing.assert_close(problem.objective(), weight * expected_function)


def test_scalar_cost_gradcheck_away_from_zero() -> None:
    variable = Variable(torch.tensor([0.7, -1.2]), name="value")
    problem = Problem([ScalarCost(lambda value: value.square().sum() + 0.4, variable, weight=1.7)])

    def objective(value: torch.Tensor) -> torch.Tensor:
        return problem.objective({"value": value})

    value = variable.tensor.detach().requires_grad_(True)
    assert torch.autograd.gradcheck(objective, (value,), eps=1e-3, atol=1e-2, rtol=1e-2)


def test_scalar_cost_exact_zero_has_finite_zero_gradient() -> None:
    tensor = torch.zeros(2, requires_grad=True)
    variable = Variable(tensor, name="value")
    problem = Problem([ScalarCost(lambda value: value.square().sum(), variable)])

    objective = problem.objective()
    objective.backward()

    torch.testing.assert_close(objective, torch.tensor(0.0))
    assert tensor.grad is not None
    torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))
    assert bool(torch.isfinite(tensor.grad).all())


@pytest.mark.parametrize("optimizer_kind", ("lm", "torch"))
def test_scalar_cost_works_with_least_squares_and_torch_optimizers(optimizer_kind: str) -> None:
    variable = Variable(torch.tensor([0.0]), name="value")
    problem = Problem([ScalarCost(lambda value: (value - 2.0).square().sum(), variable)])
    if optimizer_kind == "lm":
        optimizer = LevenbergMarquardt(problem, max_iterations=12, tolerance=1e-6)
    else:
        optimizer = TorchOptimizer(problem, torch.optim.SGD, lr=0.5, max_iterations=4, tolerance=1e-6)

    optimizer.optimize()

    torch.testing.assert_close(variable.tensor, torch.tensor([2.0]), atol=2e-4, rtol=0.0)


def test_scalar_cost_is_implicit_ineligible_with_actionable_error() -> None:
    variable = Variable(torch.tensor([0.0]), name="value")
    problem = Problem([ScalarCost(lambda value: (value - 2.0).square().sum(), variable, name="scalar_prior")])

    with pytest.raises(ValueError, match="ScalarCost residual 'scalar_prior'.*detached optimization"):
        LevenbergMarquardt(problem, max_iterations=4).optimize(differentiate="implicit")
