"""Parity and graph contracts for blockwise v2 Jacobian strategies."""

from __future__ import annotations

import warnings

import pytest
import torch

from better_robot.optim import AutodiffFallbackWarning, Problem, Residual, Variable


class _PolynomialResidual(Residual):
    def __init__(self, x: Variable, *, weight=1.0) -> None:
        self.x = x
        super().__init__(x, dim=2, weight=weight, name="polynomial")

    def error(self) -> torch.Tensor:
        x = self.x.tensor
        return torch.stack((x[..., 0].square() + 3.0 * x[..., 1], x[..., 1] * x[..., 2]), dim=-1)

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        x = self.x.tensor
        zero = torch.zeros_like(x[..., 0])
        full = torch.stack(
            (
                torch.stack((2.0 * x[..., 0], 3.0 + zero, zero), dim=-1),
                torch.stack((zero, x[..., 2], x[..., 1]), dim=-1),
            ),
            dim=-2,
        )
        return (full,)


class _CubicResidual(Residual):
    def __init__(self, x: Variable) -> None:
        self.x = x
        super().__init__(x, dim=1, name="cubic")

    def error(self) -> torch.Tensor:
        return self.x.tensor.pow(3)


class _ExplodingAnalyticResidual(Residual):
    def __init__(self, x: Variable, *, weight=1.0) -> None:
        self.x = x
        super().__init__(x, dim=2, weight=weight, name="exploding")

    def error(self) -> torch.Tensor:
        return self.x.tensor.square()

    def jacobian(self):
        raise RuntimeError("analytic failure sentinel")


class _ProductResidual(Residual):
    def __init__(self, x: Variable, static: Variable, *, name: str) -> None:
        self.x, self.static = x, static
        super().__init__(x, static, dim=x.shape[-1] if x.shape else 1, name=name)

    def error(self) -> torch.Tensor:
        return self.x.tensor * self.static.tensor


class _AffineResidual(_ProductResidual):
    def error(self) -> torch.Tensor:
        return self.x.tensor + self.static.tensor


def _polynomial_problem(value: torch.Tensor, *, weight=1.0):
    x = Variable(value, name="x", batch_ndim=max(0, value.ndim - 1))
    item = _PolynomialResidual(x, weight=weight)
    return x, item, Problem([item])


def test_auto_autodiff_fallback_warning_names_residual_and_transform() -> None:
    x = Variable(torch.tensor([0.7]), name="x")
    problem = Problem([_CubicResidual(x)])

    with pytest.warns(
        AutodiffFallbackWarning,
        match=r"_CubicResidual residual 'cubic'.*torch\.func\.jacrev.*explicit Jacobian strategy",
    ):
        problem.jacobian_blocks()


def test_auto_autodiff_fallback_warns_once_per_problem_residual() -> None:
    x = Variable(torch.tensor([0.7]), name="x")
    problem = Problem([_CubicResidual(x)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AutodiffFallbackWarning)
        problem.jacobian_blocks()
        problem.jacobian_blocks()

    assert sum(issubclass(item.category, AutodiffFallbackWarning) for item in caught) == 1


@pytest.mark.parametrize("strategy", ("jacrev", "jacfwd", "finite_difference"))
def test_explicit_autodiff_strategies_do_not_warn(strategy: str) -> None:
    x = Variable(torch.tensor([0.7]), name="x")
    problem = Problem([_CubicResidual(x)])

    with warnings.catch_warnings():
        warnings.simplefilter("error", AutodiffFallbackWarning)
        problem.jacobian_blocks(strategy=strategy)


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
def test_forced_ad_matches_analytic(strategy: str) -> None:
    _x, _item, problem = _polynomial_problem(torch.tensor([0.7, -0.4, 1.2]))
    analytic = problem.jacobian_blocks(strategy="analytic")[("polynomial", "x")]
    forced = problem.jacobian_blocks(strategy=strategy)[("polynomial", "x")]
    torch.testing.assert_close(forced, analytic, atol=1e-6, rtol=1e-6)


def test_create_graph_supports_a_function_of_jacobian_second_derivative() -> None:
    x = Variable(torch.tensor([2.0], requires_grad=True), name="x")
    problem = Problem([_CubicResidual(x)])
    block = problem.jacobian_blocks(strategy="jacrev", create_graph=True)[("cubic", "x")]
    first = torch.autograd.grad(block.sum(), x.tensor, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x.tensor)[0]
    torch.testing.assert_close(block, torch.tensor([[12.0]]))
    torch.testing.assert_close(first, torch.tensor([12.0]))
    torch.testing.assert_close(second, torch.tensor([6.0]))


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
def test_create_graph_keeps_constant_ad_block_connected(strategy: str) -> None:
    x = Variable(torch.tensor([0.7], requires_grad=True), name="x")
    offset = Variable(torch.tensor([0.4], requires_grad=True), name="offset", trainable=False)
    problem = Problem([_AffineResidual(x, offset, name="affine_external")])
    block = problem.jacobian_blocks(strategy=strategy, create_graph=True)[("affine_external", "x")]
    x_vjp, offset_vjp = torch.autograd.grad(block.sum(), (x.tensor, offset.tensor))
    torch.testing.assert_close(block, torch.ones(1, 1))
    torch.testing.assert_close(x_vjp, torch.zeros_like(x.tensor))
    torch.testing.assert_close(offset_vjp, torch.zeros_like(offset.tensor))


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
@pytest.mark.parametrize("batch_shape", [(3,), (2, 3)])
def test_batched_jacobian_matches_sequential(strategy: str, batch_shape: tuple[int, ...]) -> None:
    values = torch.linspace(-0.7, 1.1, 3 * torch.tensor(batch_shape).prod().item()).reshape(*batch_shape, 3)
    _x, _item, problem = _polynomial_problem(values)
    batched = problem.jacobian_blocks(strategy=strategy)[("polynomial", "x")]
    sequential = torch.stack(
        [
            _polynomial_problem(row)[2].jacobian_blocks(strategy=strategy)[("polynomial", "x")]
            for row in values.reshape(-1, 3)
        ]
    ).reshape(*batch_shape, 2, 3)
    torch.testing.assert_close(batched, sequential, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
def test_shared_static_vector_is_not_misread_as_a_batch_axis(strategy: str) -> None:
    x = Variable(torch.ones(3, 3), name="x", batch_ndim=1)
    scale = Variable(torch.tensor([2.0, 3.0, 4.0]), name="scale", trainable=False)
    problem = Problem([_ProductResidual(x, scale, name="external_scale")])
    block = problem.jacobian_blocks(strategy=strategy)[("external_scale", "x")]
    torch.testing.assert_close(block, torch.diag(scale.tensor).expand(3, -1, -1))


@pytest.mark.parametrize("strategy", ["jacrev", "jacfwd"])
def test_batched_tensor_weights_scale_each_jacobian(strategy: str) -> None:
    values = torch.tensor([[0.7, -0.4, 1.2], [-0.2, 0.5, 1.7], [1.1, 0.3, -0.8]])
    weights = torch.tensor([0.5, 1.25, 2.0])
    _x, item, problem = _polynomial_problem(values)
    unweighted = problem.jacobian_blocks(strategy=strategy)[("polynomial", "x")]
    item.weight = weights
    weighted = problem.jacobian_blocks(strategy=strategy)[("polynomial", "x")]
    torch.testing.assert_close(weighted, unweighted * weights[:, None, None])


def test_finite_difference_is_explicit_and_never_a_silent_fallback() -> None:
    x = Variable(torch.tensor([0.7, -1.2]), name="x")
    problem = Problem([_ExplodingAnalyticResidual(x)])
    with pytest.raises(RuntimeError, match="analytic failure sentinel"):
        problem.jacobian_blocks(strategy="auto")
    expected = torch.diag(2.0 * x.tensor)
    torch.testing.assert_close(problem.jacobian_blocks(strategy="jacrev")[("exploding", "x")], expected)
    torch.testing.assert_close(
        problem.jacobian_blocks(strategy="finite_difference", fd_eps=1e-3)[("exploding", "x")],
        expected,
        atol=2e-4,
        rtol=2e-4,
    )
    with pytest.raises(ValueError, match="finite_difference must be graph-free"):
        problem.jacobian_blocks(strategy="finite_difference", create_graph=True)


def test_finite_difference_releases_primal_and_weight_graphs() -> None:
    x = Variable(torch.tensor([0.7, -1.2], requires_grad=True), name="x")
    item = _ExplodingAnalyticResidual(x, weight=torch.tensor(1.5, requires_grad=True))
    problem = Problem([item])
    block = problem.jacobian_blocks(strategy="finite_difference")[("exploding", "x")]
    dense = problem.dense_jacobian(strategy="finite_difference")
    assert not block.requires_grad and block.grad_fn is None
    assert not dense.requires_grad and dense.grad_fn is None
