"""Robust grouped objective/gradient semantics for named-block problems."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from better_robot.optim import Problem, ResidualItem, VarSpec
from better_robot.optim.kernels import Cauchy, Huber, L2, Tukey


class _GroupedResidual:
    name = "grouped"
    reads = ("x", "target")
    dim = 4

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]


class _NonzeroAtOriginKernel:
    def __init__(self) -> None:
        self.rho_calls = 0

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        self.rho_calls += 1
        return squared_norm + 7.0

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(squared_norm)


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(L2(), id="l2"),
        pytest.param(Huber(delta=0.7), id="huber"),
        pytest.param(Cauchy(c=0.9), id="cauchy"),
        pytest.param(Tukey(c=2.5), id="tukey"),
    ],
)
def test_public_objective_and_gradient_use_grouped_kernel_convention(kernel: object) -> None:
    dtype = torch.float64
    target = torch.tensor(
        [[-0.4, 0.2, 0.8, -0.3], [0.7, -0.6, 0.1, 0.9]],
        dtype=dtype,
    )
    x = torch.tensor(
        [[0.3, -0.5, 1.2, 0.4], [-0.1, 0.5, -0.8, 0.2]],
        dtype=dtype,
        requires_grad=True,
    )
    residual_weight = 1.7
    problem = Problem(
        vars=(VarSpec("x", (4,)),),
        residuals=(
            ResidualItem(
                "grouped",
                _GroupedResidual(),
                weight=residual_weight,
                kernel=kernel,
                group_size=2,
            ),
        ),
        parameters={"target": target},
    )

    weighted = residual_weight * (x - target)
    groups = weighted.reshape(2, 2, 2)
    squared_norm = groups.square().sum(dim=-1)
    expected_cost = kernel.rho(squared_norm).sum(dim=-1)
    group_weight = kernel.weight(squared_norm).repeat_interleave(2, dim=-1)
    expected_gradient = residual_weight * group_weight * weighted

    objective = problem.objective({"x": x})
    gradient = problem.gradient({"x": x})["x"]
    coordinate_gradient = torch.autograd.grad(objective.sum(), x)[0]

    torch.testing.assert_close(objective, expected_cost, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(coordinate_gradient, expected_gradient, rtol=1e-11, atol=1e-11)


def test_inactive_residual_skips_kernel_with_nonzero_origin() -> None:
    kernel = _NonzeroAtOriginKernel()
    problem = Problem(
        vars=(VarSpec("x", (4,)),),
        residuals=(ResidualItem("grouped", _GroupedResidual(), kernel=kernel, group_size=2),),
        parameters={"target": torch.zeros(4)},
    )
    values = {"x": torch.ones(4)}
    weights = {"grouped": 0.0}

    torch.testing.assert_close(problem.objective(values, weights=weights), torch.tensor(0.0))
    torch.testing.assert_close(problem.gradient(values, weights=weights)["x"], torch.zeros(4))
    assert kernel.rho_calls == 0
