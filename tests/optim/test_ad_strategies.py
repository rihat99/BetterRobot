"""Parity and graph contracts for blockwise Jacobian strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from better_robot.optim import Problem, ResidualItem, VarSpec


class _PolynomialResidual:
    name = "polynomial"
    reads = ("x",)
    dim = 2

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        x = ctx["x"]
        return torch.stack((x[..., 0].square() + 3.0 * x[..., 1], x[..., 1] * x[..., 2]), dim=-1)

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        x = ctx["x"]
        zero = torch.zeros_like(x[0])
        full = torch.stack(
            (
                torch.stack((2.0 * x[0], 3.0 + zero, zero)),
                torch.stack((zero, x[2], x[1])),
            )
        )
        indices = ctx.free_indices("x").to(device=x.device)
        return {"x": full.index_select(-1, indices)}


class _CubicResidual:
    name = "cubic"
    reads = ("x",)
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"].pow(3)


class _ExplodingAnalyticResidual:
    name = "exploding"
    reads = ("x",)
    dim = 2

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"].square()

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        del ctx
        raise RuntimeError("analytic failure sentinel")


class _ExternalScaleResidual:
    name = "external_scale"
    reads = ("x", "scale")
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] * ctx["scale"]


class _AffineExternalResidual:
    name = "affine_external"
    reads = ("x", "offset")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] + ctx["offset"]


def _polynomial_problem(*, masked: bool = False) -> Problem:
    mask = torch.tensor([True, False, True]) if masked else None
    return Problem(
        vars=(VarSpec("x", (3,), mask=mask),),
        residuals=(ResidualItem("polynomial", _PolynomialResidual()),),
    )


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
def test_forced_ad_matches_analytic(strategy: str) -> None:
    problem = _polynomial_problem()
    values = {"x": torch.tensor([0.7, -0.4, 1.2])}

    analytic = problem.jacobian_blocks(values, strategy="analytic")[("polynomial", "x")]
    forced = problem.jacobian_blocks(values, strategy=strategy)[("polynomial", "x")]

    torch.testing.assert_close(forced, analytic, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("strategy", ["analytic", "jacrev", "jacfwd"])
def test_masked_blocks_have_only_reduced_columns(strategy: str) -> None:
    problem = _polynomial_problem(masked=True)
    values = {"x": torch.tensor([0.7, -0.4, 1.2])}

    block = problem.jacobian_blocks(values, strategy=strategy)[("polynomial", "x")]

    assert block.shape == (2, 2)
    torch.testing.assert_close(block, torch.tensor([[1.4, 0.0], [0.0, -0.4]]))


def test_create_graph_supports_a_function_of_jacobian_second_derivative() -> None:
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("cubic", _CubicResidual()),),
    )
    x = torch.tensor([2.0], requires_grad=True)

    block = problem.jacobian_blocks(
        {"x": x},
        strategy="jacrev",
        create_graph=True,
    )[("cubic", "x")]
    first = torch.autograd.grad(block.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]

    torch.testing.assert_close(block, torch.tensor([[12.0]]))
    torch.testing.assert_close(first, torch.tensor([12.0]))
    torch.testing.assert_close(second, torch.tensor([6.0]))


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
def test_create_graph_keeps_constant_ad_block_connected(strategy: str) -> None:
    offset = torch.tensor([0.4], requires_grad=True)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("affine_external", _AffineExternalResidual()),),
        parameters={"offset": offset},
        differentiable_parameters=("offset",),
    )
    x = torch.tensor([0.7], requires_grad=True)

    block = problem.jacobian_blocks(
        {"x": x},
        strategy=strategy,
        create_graph=True,
    )[("affine_external", "x")]
    x_vjp, offset_vjp = torch.autograd.grad(block.sum(), (x, offset))

    torch.testing.assert_close(block, torch.ones(1, 1))
    torch.testing.assert_close(x_vjp, torch.zeros_like(x))
    torch.testing.assert_close(offset_vjp, torch.zeros_like(offset))


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
def test_batched_jacobian_matches_sequential(strategy: str) -> None:
    problem = _polynomial_problem()
    values = {"x": torch.tensor([[0.7, -0.4, 1.2], [-0.2, 0.5, 1.7], [1.1, 0.3, -0.8]])}

    batched = problem.jacobian_blocks(values, strategy=strategy)[("polynomial", "x")]
    sequential = torch.stack(
        [
            problem.jacobian_blocks({"x": values["x"][index]}, strategy=strategy)[("polynomial", "x")]
            for index in range(values["x"].shape[0])
        ]
    )

    assert batched.shape == (3, 2, 3)
    torch.testing.assert_close(batched, sequential, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
def test_multi_axis_batched_jacobian_matches_sequential(strategy: str) -> None:
    problem = _polynomial_problem()
    values = {"x": torch.linspace(-0.7, 1.1, 18).reshape(2, 3, 3)}

    batched = problem.jacobian_blocks(values, strategy=strategy)[("polynomial", "x")]
    sequential = torch.stack(
        [
            problem.jacobian_blocks({"x": row}, strategy=strategy)[("polynomial", "x")]
            for row in values["x"].reshape(-1, 3)
        ]
    ).reshape(2, 3, 2, 3)

    assert batched.shape == (2, 3, 2, 3)
    torch.testing.assert_close(batched, sequential, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
def test_shared_external_vector_is_not_misread_as_a_batch_axis(strategy: str) -> None:
    scale = torch.tensor([2.0, 3.0, 4.0])
    problem = Problem(
        vars=(VarSpec("x", (3,)),),
        residuals=(ResidualItem("external_scale", _ExternalScaleResidual()),),
        parameters={"scale": scale},
    )
    values = {"x": torch.ones(3, 3)}  # batch size intentionally equals scale length

    block = problem.jacobian_blocks(values, strategy=strategy)[("external_scale", "x")]
    expected = torch.diag(scale).expand(3, -1, -1)

    torch.testing.assert_close(block, expected)


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
def test_batched_tensor_residual_weights_scale_each_jacobian(strategy: str) -> None:
    problem = _polynomial_problem()
    values = {"x": torch.tensor([[0.7, -0.4, 1.2], [-0.2, 0.5, 1.7], [1.1, 0.3, -0.8]])}
    weights = torch.tensor([0.5, 1.25, 2.0])

    unweighted = problem.jacobian_blocks(values, strategy=strategy)[("polynomial", "x")]
    weighted = problem.jacobian_blocks(
        values,
        weights={"polynomial": weights},
        strategy=strategy,
    )[("polynomial", "x")]

    torch.testing.assert_close(weighted, unweighted * weights[:, None, None])


def test_finite_difference_is_explicit_and_never_a_silent_fallback() -> None:
    problem = Problem(
        vars=(VarSpec("x", (2,)),),
        residuals=(ResidualItem("exploding", _ExplodingAnalyticResidual()),),
    )
    values = {"x": torch.tensor([0.7, -1.2])}

    with pytest.raises(RuntimeError, match="analytic failure sentinel"):
        problem.jacobian_blocks(values, strategy="auto")

    expected = torch.diag(2.0 * values["x"])
    torch.testing.assert_close(
        problem.jacobian_blocks(values, strategy="jacrev")[("exploding", "x")],
        expected,
    )
    torch.testing.assert_close(
        problem.jacobian_blocks(
            values,
            strategy="finite_difference",
            fd_eps=1e-3,
        )[("exploding", "x")],
        expected,
        atol=2e-4,
        rtol=2e-4,
    )
    with pytest.raises(ValueError, match="graph-free debug strategy"):
        problem.jacobian_blocks(
            values,
            strategy="finite_difference",
            create_graph=True,
        )


def test_finite_difference_releases_primal_and_weight_graphs() -> None:
    problem = Problem(
        vars=(VarSpec("x", (2,)),),
        residuals=(ResidualItem("exploding", _ExplodingAnalyticResidual()),),
    )
    values = {"x": torch.tensor([0.7, -1.2], requires_grad=True)}
    weight = torch.tensor(1.5, requires_grad=True)

    block = problem.jacobian_blocks(
        values,
        weights={"exploding": weight},
        strategy="finite_difference",
    )[("exploding", "x")]
    dense = problem.dense_jacobian(
        values,
        weights={"exploding": weight},
        strategy="finite_difference",
    )

    assert block.requires_grad is False
    assert block.grad_fn is None
    assert dense.requires_grad is False
    assert dense.grad_fn is None
