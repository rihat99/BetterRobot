"""Exact first-order implicit-differentiation contract tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, get_type_hints

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.lie import se3, so3
from better_robot.optim import TemporalPattern
from better_robot.optim.blocks import (
    Bounds,
    LevenbergMarquardt,
    LMStatus,
    Problem,
    ResidualItem,
    RobotConfig,
    SE3Manifold,
    SO3Manifold,
    VarSpec,
)
from better_robot.optim.blocks.implicit import (
    ImplicitDiffConfig,
    ImplicitDifferentiationError,
    attach_implicit_gradients,
)
from better_robot.optim.kernels import Huber


class _TargetResidual:
    name = "target"
    reads = ("x", "target")

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]


def _target_problem(target: torch.Tensor) -> Problem:
    residual = _TargetResidual(target.shape[-1])
    return Problem(
        vars=(VarSpec("x", (target.shape[-1],)),),
        residuals=(ResidualItem(residual.name, residual),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )


def _solve_and_attach(problem: Problem, initial: torch.Tensor) -> tuple[torch.Tensor, Any]:
    solver = LevenbergMarquardt(max_iter=30, gtol=1e-7, xtol=1e-12, ftol=1e-12)
    values, state = solver.run({"x": initial}, problem)
    attached = attach_implicit_gradients(values, state, problem, default_kernel=solver.kernel)
    return attached["x"], state


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quadratic_implicit_gradient_matches_closed_form(dtype: torch.dtype) -> None:
    target = torch.tensor([0.3, -0.7, 1.2], dtype=dtype, requires_grad=True)
    solution, state = _solve_and_attach(_target_problem(target), torch.zeros_like(target))

    gradient = torch.autograd.grad(0.5 * solution.square().sum(), target)[0]

    tolerance = 2e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(solution, target, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(gradient, target, atol=tolerance, rtol=tolerance)
    assert int(state.status) == int(LMStatus.CONVERGED)


def test_public_solve_is_detached_by_default_and_implicit_only_by_opt_in() -> None:
    dtype = torch.float64
    target = torch.tensor([0.4, -0.2], dtype=dtype, requires_grad=True)
    initial = torch.zeros(2, dtype=dtype, requires_grad=True)
    problem = _target_problem(target)
    solver = LevenbergMarquardt(max_iter=30, gtol=1e-9, linearization="dense")

    detached, _ = solver.solve({"x": initial}, problem)
    implicit, _ = solver.solve({"x": initial}, problem, differentiate="implicit")
    target_gradient, initial_gradient = torch.autograd.grad(
        implicit["x"].sum(),
        (target, initial),
        allow_unused=True,
    )

    assert not detached["x"].requires_grad
    assert implicit["x"].requires_grad
    torch.testing.assert_close(target_gradient, torch.ones_like(target), atol=2e-10, rtol=2e-10)
    assert initial_gradient is None


def test_public_solve_runtime_annotations_resolve() -> None:
    hints = get_type_hints(LevenbergMarquardt.solve)

    assert hints["implicit_config"] == ImplicitDiffConfig | None


def test_implicit_solve_rejects_optimized_external_parameter_identity_collision() -> None:
    shared = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    problem = _target_problem(shared)
    solver = LevenbergMarquardt(max_iter=5, gtol=1e-9)

    with pytest.raises(ValueError, match="distinct tensor roles"):
        solver.solve({"x": shared}, problem, differentiate="implicit")

    values, state = solver.run({"x": torch.zeros_like(shared)}, problem)
    colliding_problem = _target_problem(values["x"])
    with pytest.raises(ValueError, match="distinct tensor roles"):
        attach_implicit_gradients(values, state, colliding_problem)


def test_declared_differentiable_parameter_must_reach_terminal_optimality() -> None:
    target = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    unused = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    residual = _TargetResidual(1)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem(residual.name, residual),),
        parameters={"target": target, "unused": unused},
        differentiable_parameters=("target", "unused"),
    )
    solver = LevenbergMarquardt(max_iter=20, gtol=1e-9)
    values, _state = solver.solve(
        {"x": torch.zeros_like(target)},
        problem,
        differentiate="implicit",
    )

    with pytest.raises(ImplicitDifferentiationError, match="disconnected.*unused"):
        values["x"].sum().backward()


def test_implicit_backward_is_first_order_only() -> None:
    target = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    solution, _state = _solve_and_attach(_target_problem(target), torch.zeros_like(target))

    first = torch.autograd.grad(solution.sum(), target, create_graph=True)[0]

    assert not first.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(first.sum(), target)


class _SharedTargetResidual:
    name = "shared_target"
    reads = ("x", "shared")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["shared"]


def test_shared_scalar_parameter_vjp_reduces_over_multi_axis_batch() -> None:
    dtype = torch.float64
    shared = torch.tensor(0.25, dtype=dtype, requires_grad=True)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("shared_target", _SharedTargetResidual()),),
        parameters={"shared": shared},
        differentiable_parameters=("shared",),
    )
    solver = LevenbergMarquardt(max_iter=30, gtol=1e-9, linearization="dense")
    values, _state = solver.solve(
        {"x": torch.zeros(2, 3, 1, dtype=dtype)},
        problem,
        differentiate="implicit",
    )

    gradient = torch.autograd.grad(values["x"].sum(), shared)[0]

    torch.testing.assert_close(gradient, torch.tensor(6.0, dtype=dtype), atol=2e-10, rtol=2e-10)


def test_batched_implicit_gradient_matches_independent_solutions() -> None:
    dtype = torch.float64
    target = torch.linspace(-0.8, 0.9, 48, dtype=dtype).reshape(16, 3).requires_grad_()
    weights = torch.linspace(0.2, 1.7, 48, dtype=dtype).reshape(16, 3)
    solution, _state = _solve_and_attach(_target_problem(target), torch.zeros_like(target))

    batched = torch.autograd.grad((weights * solution).sum(), target)[0]

    torch.testing.assert_close(batched, weights, atol=2e-10, rtol=2e-10)
    for index in range(16):
        sequential_target = target[index].detach().clone().requires_grad_()
        sequential, _ = _solve_and_attach(_target_problem(sequential_target), torch.zeros(3, dtype=dtype))
        sequential_gradient = torch.autograd.grad((weights[index] * sequential).sum(), sequential_target)[0]
        torch.testing.assert_close(batched[index], sequential_gradient, atol=2e-10, rtol=2e-10)


class _RotationTargetResidual:
    name = "rotation_target"
    reads = ("rotation", "target")
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return so3.log(ctx["rotation"]) - ctx["target"]


def test_manifold_output_cotangent_is_mapped_through_the_local_chart() -> None:
    dtype = torch.float64
    target = torch.tensor([0.2, -0.1, 0.3], dtype=dtype, requires_grad=True)
    problem = Problem(
        vars=(VarSpec("rotation", (4,), manifold=SO3Manifold()),),
        residuals=(ResidualItem("rotation_target", _RotationTargetResidual()),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    solver = LevenbergMarquardt(max_iter=30, gtol=1e-9, xtol=1e-12, ftol=1e-12)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype)
    values, state = solver.run({"rotation": identity}, problem)
    attached = attach_implicit_gradients(values, state, problem, default_kernel=solver.kernel)

    gradient = torch.autograd.grad(so3.log(attached["rotation"]).sum(), target)[0]

    torch.testing.assert_close(so3.log(attached["rotation"]), target, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(gradient, torch.ones_like(target), atol=2e-9, rtol=2e-9)


def test_exact_pi_so3_terminal_is_rejected_as_a_log_branch_cut() -> None:
    dtype = torch.float64
    target = torch.tensor([torch.pi, 0.0, 0.0], dtype=dtype, requires_grad=True)
    problem = Problem(
        vars=(VarSpec("rotation", (4,), manifold=SO3Manifold()),),
        residuals=(ResidualItem("rotation_target", _RotationTargetResidual()),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    exact = so3.exp(target.detach())
    solver = LevenbergMarquardt(max_iter=2, gtol=1e-10)
    values, state = solver.run({"rotation": exact}, problem)
    assert int(state.status) == int(LMStatus.CONVERGED)
    attached = attach_implicit_gradients(values, state, problem)

    with pytest.raises(ImplicitDifferentiationError, match="absolute-pi principal-log branch cut"):
        so3.log(attached["rotation"]).sum().backward()


class _PoseTargetResidual:
    name = "pose_target"
    reads = ("pose", "target")
    dim = 6

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return se3.log(ctx["pose"]) - ctx["target"]


def test_exact_pi_se3_terminal_is_rejected_as_a_log_branch_cut() -> None:
    dtype = torch.float64
    target = torch.tensor([0.0, 0.0, 0.0, torch.pi, 0.0, 0.0], dtype=dtype, requires_grad=True)
    problem = Problem(
        vars=(VarSpec("pose", (7,), manifold=SE3Manifold()),),
        residuals=(ResidualItem("pose_target", _PoseTargetResidual()),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    solver = LevenbergMarquardt(max_iter=2, gtol=1e-10)
    values, state = solver.run({"pose": se3.exp(target.detach())}, problem)
    assert int(state.status) == int(LMStatus.CONVERGED)
    attached = attach_implicit_gradients(values, state, problem)

    with pytest.raises(ImplicitDifferentiationError, match="absolute-pi principal-log branch cut"):
        se3.log(attached["pose"]).sum().backward()


class _RobotTangentResidual:
    name = "robot_tangent"
    reads = ("q", "target")

    def __init__(self, model: Any) -> None:
        self.model = model
        self.dim = model.nv

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        neutral = self.model.q_neutral.to(dtype=ctx["q"].dtype, device=ctx["q"].device)
        return self.model.difference(neutral, ctx["q"]) - ctx["target"]


@pytest.mark.parametrize("quaternion_event", [0, 1])
def test_robot_config_checks_every_quaternion_event_for_log_branch_cut(
    quaternion_event: int,
) -> None:
    dtype = torch.float64
    builder = ModelBuilder("implicit_branch_cuts")
    base = builder.add_body("base", mass=1.0)
    tip = builder.add_body("tip", mass=1.0)
    builder.add_free_flyer_root("floating", child=base)
    builder.add_spherical("ball", parent=base, child=tip)
    model = build_model(builder.finalize(), dtype=dtype)
    manifold = RobotConfig(model)
    quaternion_slices = tuple(
        unit_slice
        for unit_slice in manifold.unit_coordinate_slices
        if unit_slice.stop - unit_slice.start == 4
    )
    assert len(quaternion_slices) == 2

    exact = model.q_neutral.clone()
    exact[quaternion_slices[quaternion_event]] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        dtype=dtype,
    )
    target = model.difference(model.q_neutral, exact).detach().requires_grad_()
    residual = _RobotTangentResidual(model)
    problem = Problem(
        vars=(VarSpec("q", (model.nq,), manifold=manifold),),
        residuals=(ResidualItem(residual.name, residual),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    solver = LevenbergMarquardt(max_iter=2, gtol=1e-10)
    values, state = solver.run({"q": exact}, problem)
    assert int(state.status) == int(LMStatus.CONVERGED)
    attached = attach_implicit_gradients(values, state, problem)

    with pytest.raises(ImplicitDifferentiationError, match="absolute-pi principal-log branch cut"):
        model.difference(model.q_neutral, attached["q"]).sum().backward()


def test_robot_config_nq_ne_nv_maps_ambient_output_cotangent() -> None:
    dtype = torch.float64
    builder = ModelBuilder("implicit_free_flyer")
    base = builder.add_body("base", mass=1.0)
    builder.add_free_flyer_root("floating", child=base)
    model = build_model(builder.finalize(), dtype=dtype)
    assert model.nq == 7 and model.nv == 6
    target = torch.tensor([0.1, -0.05, 0.08, 0.02, -0.03, 0.04], dtype=dtype, requires_grad=True)
    residual = _RobotTangentResidual(model)
    problem = Problem(
        vars=(VarSpec("q", (model.nq,), manifold=RobotConfig(model)),),
        residuals=(ResidualItem(residual.name, residual),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    solver = LevenbergMarquardt(max_iter=30, gtol=1e-9, linearization="dense")
    values, _state = solver.solve(
        {"q": model.q_neutral.clone()},
        problem,
        differentiate="implicit",
    )

    tangent = model.difference(model.q_neutral, values["q"])
    gradient = torch.autograd.grad(tangent.sum(), target)[0]

    torch.testing.assert_close(tangent, target, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(gradient, torch.ones_like(target), atol=2e-9, rtol=2e-9)


def test_stable_active_bound_has_zero_sensitivity_while_free_coordinate_differentiates() -> None:
    dtype = torch.float64
    target = torch.tensor([2.0, 0.3], dtype=dtype, requires_grad=True)
    bounds = Bounds(
        lower=torch.tensor([-torch.inf, -torch.inf], dtype=dtype),
        upper=torch.tensor([1.0, torch.inf], dtype=dtype),
    )
    residual = _TargetResidual(2)
    problem = Problem(
        vars=(VarSpec("x", (2,), bounds=bounds),),
        residuals=(ResidualItem(residual.name, residual),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    solver = LevenbergMarquardt(max_iter=30, gtol=1e-7, xtol=1e-12, ftol=1e-12)
    values, state = solver.run({"x": torch.zeros(2, dtype=dtype)}, problem)
    attached = attach_implicit_gradients(values, state, problem, default_kernel=solver.kernel)

    gradient = torch.autograd.grad(attached["x"].sum(), target)[0]

    assert int(state.status) == int(LMStatus.STALLED_AT_BOUNDS)
    torch.testing.assert_close(state.active_mask, torch.tensor([True, False]))
    torch.testing.assert_close(gradient, torch.tensor([0.0, 1.0], dtype=dtype), atol=2e-9, rtol=2e-9)


class _HuberOutlierResidual:
    name = "outlier"
    reads = ("x", "scale")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["scale"] * ctx["x"] - 10.0


class _PriorResidual:
    name = "prior"
    reads = ("x",)
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"]


class _FixedTargetResidual:
    name = "fixed_target"
    reads = ("x",)
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - 1.0


def test_identity_only_item_weight_binding_is_rejected_as_disconnected() -> None:
    weight = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(
            ResidualItem("fixed_target", _FixedTargetResidual(), weight=weight),
            ResidualItem("prior", _PriorResidual(), weight=0.5),
        ),
        parameters={"weight": weight},
        differentiable_parameters=("weight",),
    )
    solver = LevenbergMarquardt(max_iter=30, gtol=1e-9)
    values, _state = solver.solve(
        {"x": torch.zeros(1, dtype=torch.float64)},
        problem,
        differentiate="implicit",
    )

    with pytest.raises(ImplicitDifferentiationError, match="disconnected.*weight"):
        values["x"].sum().backward()


def test_huber_backward_uses_exact_robust_optimality() -> None:
    dtype = torch.float64
    scale = torch.tensor(1.0, dtype=dtype, requires_grad=True)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(
            ResidualItem("outlier", _HuberOutlierResidual(), kernel=Huber(delta=1.0)),
            ResidualItem("prior", _PriorResidual(), weight=0.5),
        ),
        parameters={"scale": scale},
        differentiable_parameters=("scale",),
    )
    solver = LevenbergMarquardt(max_iter=100, gtol=1e-6, xtol=1e-12, ftol=1e-12)
    values, state = solver.run({"x": torch.zeros(1, dtype=dtype)}, problem)
    attached = attach_implicit_gradients(values, state, problem, default_kernel=solver.kernel)

    gradient = torch.autograd.grad(attached["x"].sum(), scale)[0]

    torch.testing.assert_close(attached["x"], torch.tensor([4.0], dtype=dtype), atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(gradient, torch.tensor(4.0, dtype=dtype), atol=2e-7, rtol=2e-7)


class _KinkResidual:
    name = "kink"
    reads = ("x",)
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"]


class _BalancingResidual:
    name = "balance"
    reads = ("x", "target")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]


def test_huber_kink_is_strictly_rejected() -> None:
    dtype = torch.float64
    target = torch.tensor([2.0], dtype=dtype, requires_grad=True)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(
            ResidualItem("kink", _KinkResidual(), kernel=Huber(delta=1.0)),
            ResidualItem("balance", _BalancingResidual()),
        ),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    solver = LevenbergMarquardt(max_iter=1, gtol=1e-9, linearization="dense")
    values, state = solver.run({"x": torch.ones(1, dtype=dtype)}, problem)
    assert int(state.status) == int(LMStatus.CONVERGED)
    attached = attach_implicit_gradients(values, state, problem)

    with pytest.raises(ImplicitDifferentiationError, match="nonsmooth"):
        attached["x"].sum().backward()


class _RankDeficientResidual:
    name = "rank_deficient"
    reads = ("x", "target")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"][..., :1] - ctx["target"]


def test_rank_deficient_backward_is_strictly_rejected() -> None:
    dtype = torch.float64
    target = torch.tensor([0.3], dtype=dtype, requires_grad=True)
    problem = Problem(
        vars=(VarSpec("x", (2,)),),
        residuals=(ResidualItem("rank_deficient", _RankDeficientResidual()),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    solver = LevenbergMarquardt(max_iter=30, gtol=1e-9, linearization="dense")
    values, state = solver.run({"x": torch.zeros(2, dtype=dtype)}, problem)
    attached = attach_implicit_gradients(values, state, problem)

    with pytest.raises(ImplicitDifferentiationError, match="singular"):
        attached["x"].sum().backward()


def test_mixed_batch_rejects_every_gradient_if_one_terminal_status_is_invalid() -> None:
    target = torch.tensor([[0.2], [0.4]], dtype=torch.float64, requires_grad=True)
    problem = _target_problem(target)
    solver = LevenbergMarquardt(max_iter=20, gtol=1e-9)
    values, state = solver.run({"x": torch.zeros_like(target)}, problem)
    bad_status = state.status.clone()
    bad_status[1] = LMStatus.MAXITER
    bad_state = state._replace(status=bad_status)
    attached = attach_implicit_gradients(values, bad_state, problem, default_kernel=solver.kernel)

    with pytest.raises(ImplicitDifferentiationError, match="invalid batch indices") as caught:
        attached["x"].sum().backward()

    assert caught.value.invalid_indices == ((1,),)
    assert caught.value.statuses == ("maxiter",)


@dataclass(frozen=True)
class _TemporalTargetResidual:
    horizon: int
    name: str = "temporal_target"
    reads: tuple[str, ...] = ("x", "target")

    @property
    def dim(self) -> int:
        return self.horizon

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        value = ctx["x"]
        return (value[..., :, 0] - ctx["target"]).reshape(*value.shape[:-2], self.horizon)

    def temporal_structure(self, variable_name: str) -> TemporalPattern | None:
        return TemporalPattern(self.horizon, 1, 0, (0,)) if variable_name == "x" else None

    def temporal_jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
        variable_name: str,
    ) -> Mapping[int, torch.Tensor]:
        assert variable_name == "x"
        value = ctx["x"]
        return {0: value.new_ones(*value.shape[:-2], self.horizon, 1, 1)}


def test_small_banded_forward_requires_explicit_dense_backward_opt_in_and_matches_dense() -> None:
    dtype = torch.float64
    horizon = 4
    target = torch.linspace(-0.3, 0.6, horizon, dtype=dtype, requires_grad=True)
    residual = _TemporalTargetResidual(horizon)
    problem = Problem(
        vars=(VarSpec("x", (horizon, 1), time_axis=0),),
        residuals=(ResidualItem(residual.name, residual),),
        parameters={"target": target},
        differentiable_parameters=("target",),
    )
    initial = {"x": torch.zeros(horizon, 1, dtype=dtype)}
    dense_solver = LevenbergMarquardt(max_iter=20, gtol=1e-9, linearization="dense")
    banded_solver = LevenbergMarquardt(max_iter=20, gtol=1e-9, linearization="structured")
    dense_values, dense_state = dense_solver.run(initial, problem)
    banded_values, banded_state = banded_solver.run(initial, problem)

    with pytest.raises(ValueError, match="allow_banded_dense_backward"):
        attach_implicit_gradients(
            banded_values,
            banded_state,
            problem,
            forward_linearization="banded",
        )

    dense = attach_implicit_gradients(dense_values, dense_state, problem)
    banded = attach_implicit_gradients(
        banded_values,
        banded_state,
        problem,
        forward_linearization="banded",
        config=ImplicitDiffConfig(allow_banded_dense_backward=True),
    )
    dense_gradient = torch.autograd.grad(dense["x"].sum(), target, retain_graph=True)[0]
    banded_gradient = torch.autograd.grad(banded["x"].sum(), target)[0]

    torch.testing.assert_close(banded["x"], dense["x"], atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(banded_gradient, dense_gradient, atol=2e-10, rtol=2e-10)


def test_matrix_free_and_large_dense_materialization_are_rejected_before_backward() -> None:
    target = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    problem = _target_problem(target)
    solver = LevenbergMarquardt(max_iter=5)
    values, state = solver.run({"x": torch.zeros_like(target)}, problem)

    with pytest.raises(ValueError, match="does not materialize a matrix-free forward"):
        attach_implicit_gradients(values, state, problem, forward_linearization="matrix_free")
    with pytest.raises(ValueError, match="configured cap is 2"):
        attach_implicit_gradients(values, state, problem, config=ImplicitDiffConfig(max_dense_tangent_dim=2))
