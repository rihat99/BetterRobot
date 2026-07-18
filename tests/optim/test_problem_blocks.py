"""Contract tests for named variable blocks and dense-v1 assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from better_robot.lie import so3
from better_robot.optim import (
    Euclidean,
    LevenbergMarquardt,
    Problem,
    ResidualItem,
    SO3Manifold,
    VarSpec,
)


_POSITION_TARGET = torch.tensor([1.0, -2.0, 0.5])
_ROTATION_TARGET = torch.tensor([0.2, -0.1, 0.3])


def _like(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return reference.to(device=value.device, dtype=value.dtype)


def _identity_rotation(*batch_shape: int) -> torch.Tensor:
    result = torch.zeros(*batch_shape, 4)
    result[..., 3] = 1.0
    return result


class _PositionResidual:
    name = "position"
    reads = ("x",)
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        x = ctx["x"]
        return x - _like(x, _POSITION_TARGET)


class _AttitudeResidual:
    name = "attitude"
    reads = ("rotation",)
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        rotation = ctx["rotation"]
        return so3.log(rotation) - _like(rotation, _ROTATION_TARGET)


class _MaskedPositionResidual:
    name = "masked_position"
    reads = ("x",)
    dim = 2

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"][..., (0, 2)]


class _ParameterResidual:
    name = "parameter"
    reads = ("x", "target")
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]


class _UndeclaredReadResidual:
    name = "undeclared"
    reads = ("x",)
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["rotation"][..., :3]


class _PromotingResidual:
    name = "promoting"
    reads = ("x",)
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"].double()


class _PromotingAnalyticResidual(_PositionResidual):
    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {"x": torch.eye(3, device=ctx["x"].device, dtype=torch.float64)}


class _DetachedAnalyticResidual:
    name = "detached_analytic"
    reads = ("x",)
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"].square()

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {"x": torch.diag(2.0 * ctx["x"]).detach()}


class _PerElementInvalidResidual:
    name = "validity"
    reads = ("x",)
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        x = ctx["x"]
        return torch.where(x > 0, torch.full_like(x, torch.nan), x)


def _two_block_problem(*, residuals=None, x_spec: VarSpec | None = None) -> Problem:
    return Problem(
        vars=(
            x_spec or VarSpec("x", (3,), manifold=Euclidean()),
            VarSpec("rotation", (4,), manifold=SO3Manifold()),
        ),
        residuals=residuals
        or (
            ResidualItem("position", _PositionResidual()),
            ResidualItem("attitude", _AttitudeResidual()),
        ),
    )


def test_builder_solves_the_public_line_fit_example() -> None:
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 3.0, 5.0, 7.0])

    def fit(ctx):
        m, c = ctx["theta"][..., 0:1], ctx["theta"][..., 1:2]
        return m * x + c - y

    problem = Problem()
    problem.add_variable("theta", shape=(2,))
    item = problem.add_residual(fit, dim=4)
    values, _state = LevenbergMarquardt().run({"theta": torch.zeros(2)}, problem)

    assert item.name == "fit"
    assert item.residual.reads == ("theta",)
    torch.testing.assert_close(values["theta"], torch.tensor([2.0, 1.0]), rtol=1e-4, atol=1e-4)


def test_two_block_shapes_gradient_and_structural_zeros() -> None:
    problem = _two_block_problem()
    values = {
        "x": torch.tensor([0.5, -1.0, 2.0]),
        "rotation": _identity_rotation(),
    }

    residual = problem.residual(values)
    gradient = problem.gradient(values)
    blocks = problem.jacobian_blocks(values)

    assert residual.shape == (6,)
    torch.testing.assert_close(
        residual,
        torch.cat((values["x"] - _POSITION_TARGET, -_ROTATION_TARGET)),
    )
    assert gradient["x"].shape == (3,)
    assert gradient["rotation"].shape == (3,)
    torch.testing.assert_close(gradient["x"], values["x"] - _POSITION_TARGET)
    torch.testing.assert_close(gradient["rotation"], -_ROTATION_TARGET)

    assert set(blocks) == {("position", "x"), ("attitude", "rotation")}
    assert ("position", "rotation") not in blocks
    assert ("attitude", "x") not in blocks
    assert blocks[("position", "x")].shape == (3, 3)
    assert blocks[("attitude", "rotation")].shape == (3, 3)


def test_two_block_batched_evaluation_matches_sequential() -> None:
    problem = _two_block_problem()
    values = {
        "x": torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.0, -0.5], [-0.2, 0.3, 0.7]]),
        "rotation": _identity_rotation(3),
    }

    residual = problem.residual(values)
    gradient = problem.gradient(values)

    assert residual.shape == (3, 6)
    assert gradient["x"].shape == (3, 3)
    assert gradient["rotation"].shape == (3, 3)
    for index in range(3):
        sequential_values = {name: value[index] for name, value in values.items()}
        torch.testing.assert_close(residual[index], problem.residual(sequential_values))
        sequential_gradient = problem.gradient(sequential_values)
        for name in gradient:
            torch.testing.assert_close(gradient[name][index], sequential_gradient[name])


def test_mask_gather_expand_scale_and_reduced_normal_matrix() -> None:
    spec = VarSpec(
        "x",
        (3,),
        mask=torch.tensor([True, False, True]),
        scale=torch.tensor([2.0, 3.0, 4.0]),
    )
    full = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])

    assert spec.tangent_dim == 3
    assert spec.free_dim == 2
    torch.testing.assert_close(spec.free_indices, torch.tensor([0, 2]))
    torch.testing.assert_close(spec.free_scale, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(spec.gather_tangent(full), full[:, (0, 2)])
    torch.testing.assert_close(
        spec.expand_tangent(spec.gather_tangent(full)),
        torch.tensor([[1.0, 0.0, 3.0], [-1.0, 0.0, -3.0]]),
    )

    problem = Problem(
        vars=(spec,),
        residuals=(ResidualItem("masked_position", _MaskedPositionResidual()),),
    )
    jacobian = problem.dense_jacobian({"x": torch.tensor([0.3, 9.0, -0.4])})
    normal = jacobian.mT @ jacobian
    torch.testing.assert_close(normal, torch.eye(2))
    assert torch.linalg.matrix_rank(normal).item() == 2

    # The old zero-column convention retained the fixed coordinate and made
    # the same normal system singular. Elimination produces the 2x2 system above.
    zero_column_jacobian = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    zero_column_normal = zero_column_jacobian.mT @ zero_column_jacobian
    assert torch.linalg.matrix_rank(zero_column_normal).item() == 2
    assert zero_column_normal.shape == (3, 3)
    assert torch.linalg.det(zero_column_normal) == 0


@pytest.mark.parametrize(
    ("scale", "error", "message"),
    [
        (torch.ones(2), ValueError, "must have tangent shape"),
        (torch.tensor([1, 2, 3]), TypeError, "floating dtype"),
    ],
)
def test_scale_validation(scale: torch.Tensor, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        VarSpec("x", (3,), scale=scale)


def test_dense_assembly_has_exact_declared_offsets() -> None:
    x_spec = VarSpec("x", (3,), mask=torch.tensor([True, False, True]))
    problem = _two_block_problem(
        x_spec=x_spec,
        residuals=(
            ResidualItem("attitude", _AttitudeResidual()),
            ResidualItem("masked_position", _MaskedPositionResidual()),
        ),
    )
    values = {"x": torch.tensor([0.2, 8.0, -0.7]), "rotation": _identity_rotation()}

    assert problem.row_offsets["attitude"] == slice(0, 3)
    assert problem.row_offsets["masked_position"] == slice(3, 5)
    assert problem.column_offsets["x"] == slice(0, 2)
    assert problem.column_offsets["rotation"] == slice(2, 5)

    expected = torch.zeros(5, 5)
    expected[:3, 2:] = torch.eye(3)
    expected[3:, :2] = torch.eye(2)
    torch.testing.assert_close(problem.dense_jacobian(values), expected)


def test_tensor_weight_must_match_working_dtype() -> None:
    problem = Problem(
        vars=(VarSpec("x", (3,)),),
        residuals=(ResidualItem("position", _PositionResidual()),),
    )

    with pytest.raises(ValueError, match="must preserve working dtype/device"):
        problem.residual(
            {"x": torch.ones(2, 3, dtype=torch.float32)},
            weights={"position": torch.ones(2, dtype=torch.float64)},
        )


def test_external_tensor_parameters_are_enumerated_and_graph_visible() -> None:
    target = torch.tensor([0.2, -0.3, 0.5], requires_grad=True)
    problem = Problem(
        vars=(VarSpec("x", (3,)),),
        residuals=(ResidualItem("parameter", _ParameterResidual()),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    values = {"x": torch.tensor([0.7, -0.1, 0.8])}

    assert tuple(problem.external_parameters) == ("target",)
    assert tuple(problem.differentiable_external_parameters) == ("target",)
    gradient = problem.gradient(values, create_graph=True)["x"]
    target_vjp = torch.autograd.grad(gradient.sum(), target)[0]

    torch.testing.assert_close(gradient, values["x"] - target)
    torch.testing.assert_close(target_vjp, -torch.ones(3))


def test_unused_gradient_block_stays_connected_to_declared_external_parameter() -> None:
    target = torch.tensor([0.2, -0.3, 0.5], requires_grad=True)
    problem = Problem(
        vars=(VarSpec("x", (3,)), VarSpec("unused", (1,))),
        residuals=(ResidualItem("parameter", _ParameterResidual()),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    values = {"x": torch.tensor([0.7, -0.1, 0.8]), "unused": torch.tensor([2.0])}

    unused_gradient = problem.gradient(values, create_graph=True)["unused"]
    target_vjp = torch.autograd.grad(unused_gradient.sum(), target)[0]

    torch.testing.assert_close(unused_gradient, torch.zeros(1))
    torch.testing.assert_close(target_vjp, torch.zeros_like(target))


def test_reads_declare_structure_without_policing_access() -> None:
    values = {"x": torch.ones(3), "rotation": _identity_rotation()}
    undeclared = _two_block_problem(residuals=(ResidualItem("undeclared", _UndeclaredReadResidual()),))
    torch.testing.assert_close(undeclared.residual(values), torch.zeros(3))
    blocks = undeclared.jacobian_blocks(values, strategy="jacrev")
    assert set(blocks) == {("undeclared", "x")}
    torch.testing.assert_close(blocks[("undeclared", "x")], torch.zeros(3, 3))


def test_working_dtype_fails_fast() -> None:

    promoting = Problem(
        vars=(VarSpec("x", (3,)),),
        residuals=(ResidualItem("promoting", _PromotingResidual()),),
    )
    with pytest.raises(ValueError, match="must preserve working dtype/device"):
        promoting.residual({"x": torch.ones(3)})
    for strategy in ("jacrev", "jacfwd", "finite_difference"):
        with pytest.raises(ValueError, match="must preserve working dtype/device"):
            promoting.jacobian_blocks({"x": torch.ones(3)}, strategy=strategy)

    analytic = Problem(
        vars=(VarSpec("x", (3,)),),
        residuals=(ResidualItem("position", _PromotingAnalyticResidual()),),
    )
    with pytest.raises(ValueError, match="Analytic block.*working dtype/device"):
        analytic.jacobian_blocks({"x": torch.ones(3)}, strategy="analytic")

    detached = Problem(
        vars=(VarSpec("x", (3,)),),
        residuals=(ResidualItem("detached_analytic", _DetachedAnalyticResidual()),),
    )
    with pytest.raises(ValueError, match="cannot honor create_graph=True"):
        detached.jacobian_blocks(
            {"x": torch.ones(3, requires_grad=True)},
            strategy="analytic",
            create_graph=True,
        )
    with pytest.raises(ValueError, match="cannot honor create_graph=True"):
        detached.jacobian_blocks(
            {"x": torch.ones(3, requires_grad=True)},
            weights={"detached_analytic": torch.tensor(2.0, requires_grad=True)},
            strategy="analytic",
            create_graph=True,
        )


def test_robust_group_size_must_partition_static_dimension() -> None:
    with pytest.raises(ValueError, match="positive divisor"):
        ResidualItem("position", _PositionResidual(), group_size=2)
    assert ResidualItem("position", _PositionResidual(), group_size=3).group_size == 3


def test_invalid_batch_element_uses_nan_rows_without_raising() -> None:
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("validity", _PerElementInvalidResidual()),),
    )

    residual = problem.residual({"x": torch.tensor([[-1.0], [1.0]])})

    assert residual.shape == (2, 1)
    torch.testing.assert_close(residual[0], torch.tensor([-1.0]))
    assert torch.isnan(residual[1]).all()
