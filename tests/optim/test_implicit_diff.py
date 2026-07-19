"""Exact first-order implicit-differentiation contract tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import get_type_hints

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.lie import se3, so3
from better_robot.optim import (
    Bounds,
    LevenbergMarquardt,
    OptimizerInfo,
    OptimizerStatus,
    Problem,
    Residual,
    RobotVariable,
    SE3Variable,
    SO3Variable,
    TemporalPattern,
    Variable,
    residual,
)
from better_robot.optim.implicit import (
    ImplicitDiffConfig,
    ImplicitDifferentiationError,
)
from better_robot.optim.kernels import Huber


def _target_graph(
    target_tensor: torch.Tensor,
    initial: torch.Tensor,
    *,
    bounds: Bounds | None = None,
) -> tuple[Variable, Variable, Problem]:
    batch_ndim = max(initial.ndim - 1, 0)
    solution = Variable(initial, name="x", bounds=bounds, batch_ndim=batch_ndim)
    target = Variable(
        target_tensor,
        name="target",
        trainable=False,
        batch_ndim=batch_ndim,
    )

    @residual(solution, target, dim=initial.shape[-1], name="target_residual")
    def target_residual(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return value - desired

    return solution, target, Problem([target_residual])


def _solve_implicitly(target: torch.Tensor, initial: torch.Tensor) -> tuple[torch.Tensor, OptimizerInfo]:
    solution, _target, problem = _target_graph(target, initial)
    optimizer = LevenbergMarquardt(
        problem,
        max_iterations=30,
        tolerance=1e-7,
        step_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    info = optimizer.optimize(differentiate="implicit")
    return solution.tensor, info


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quadratic_implicit_gradient_matches_closed_form(dtype: torch.dtype) -> None:
    target = torch.tensor([0.3, -0.7, 1.2], dtype=dtype, requires_grad=True)
    solution, info = _solve_implicitly(target, torch.zeros_like(target))

    gradient = torch.autograd.grad(0.5 * solution.square().sum(), target)[0]

    tolerance = 2e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(solution, target, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(gradient, target, atol=tolerance, rtol=tolerance)
    assert int(info.status) == int(OptimizerStatus.CONVERGED)


def test_public_optimize_is_detached_by_default_and_implicit_only_by_opt_in() -> None:
    dtype = torch.float64
    target = torch.tensor([0.4, -0.2], dtype=dtype, requires_grad=True)
    initial = torch.zeros(2, dtype=dtype, requires_grad=True)
    detached, _detached_target, detached_problem = _target_graph(target, initial)
    implicit, _implicit_target, implicit_problem = _target_graph(target, initial)

    LevenbergMarquardt(
        detached_problem,
        max_iterations=30,
        tolerance=1e-9,
        linearization="dense",
    ).optimize()
    LevenbergMarquardt(
        implicit_problem,
        max_iterations=30,
        tolerance=1e-9,
        linearization="dense",
    ).optimize(differentiate="implicit")
    target_gradient, initial_gradient = torch.autograd.grad(
        implicit.tensor.sum(),
        (target, initial),
        allow_unused=True,
    )

    assert not detached.tensor.requires_grad
    assert implicit.tensor.requires_grad
    torch.testing.assert_close(target_gradient, torch.ones_like(target), atol=2e-10, rtol=2e-10)
    assert initial_gradient is None


def test_public_optimize_runtime_annotations_resolve() -> None:
    hints = get_type_hints(LevenbergMarquardt.optimize)

    assert hints["implicit_config"] == ImplicitDiffConfig | None


def test_object_owned_solve_rebases_shared_initial_and_static_tensor_roles() -> None:
    shared = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    solution, _target, problem = _target_graph(shared, shared)

    info = LevenbergMarquardt(problem, max_iterations=5, tolerance=1e-9).optimize(differentiate="implicit")
    gradient = torch.autograd.grad(solution.tensor.sum(), shared)[0]

    assert bool(info.converged)
    torch.testing.assert_close(solution.tensor, shared)
    torch.testing.assert_close(gradient, torch.ones_like(shared))


def test_graph_carrying_static_variable_must_reach_terminal_optimality() -> None:
    target_tensor = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    unused_tensor = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    solution = Variable(torch.zeros_like(target_tensor), name="x")
    target = Variable(target_tensor, name="target", trainable=False)
    unused = Variable(unused_tensor, name="unused", trainable=False)

    @residual(solution, target, unused, dim=1, name="target_with_unused")
    def target_with_unused(
        value: torch.Tensor,
        desired: torch.Tensor,
        ignored: torch.Tensor,
    ) -> torch.Tensor:
        del ignored
        return value - desired

    optimizer = LevenbergMarquardt(Problem([target_with_unused]), max_iterations=20, tolerance=1e-9)
    optimizer.optimize(differentiate="implicit")

    with pytest.raises(ImplicitDifferentiationError, match="disconnected.*unused"):
        solution.tensor.sum().backward()


def test_implicit_backward_is_first_order_only() -> None:
    target = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    solution, _info = _solve_implicitly(target, torch.zeros_like(target))

    first = torch.autograd.grad(solution.sum(), target, create_graph=True)[0]

    assert not first.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(first.sum(), target)


def test_shared_scalar_parameter_vjp_reduces_over_multi_axis_batch() -> None:
    dtype = torch.float64
    shared_tensor = torch.tensor(0.25, dtype=dtype, requires_grad=True)
    solution = Variable(torch.zeros(2, 3, 1, dtype=dtype), name="x", batch_ndim=2)
    shared = Variable(shared_tensor, name="shared", trainable=False)

    @residual(solution, shared, dim=1, name="shared_target")
    def shared_target(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return value - desired

    LevenbergMarquardt(
        Problem([shared_target]),
        max_iterations=30,
        tolerance=1e-9,
        linearization="dense",
    ).optimize(differentiate="implicit")
    gradient = torch.autograd.grad(solution.tensor.sum(), shared_tensor)[0]

    torch.testing.assert_close(gradient, torch.tensor(6.0, dtype=dtype), atol=2e-10, rtol=2e-10)


def test_batched_implicit_gradient_matches_independent_solutions() -> None:
    dtype = torch.float64
    target = torch.linspace(-0.8, 0.9, 48, dtype=dtype).reshape(16, 3).requires_grad_()
    weights = torch.linspace(0.2, 1.7, 48, dtype=dtype).reshape(16, 3)
    solution, _info = _solve_implicitly(target, torch.zeros_like(target))

    batched = torch.autograd.grad((weights * solution).sum(), target)[0]

    torch.testing.assert_close(batched, weights, atol=2e-10, rtol=2e-10)
    for index in range(16):
        sequential_target = target[index].detach().clone().requires_grad_()
        sequential, _ = _solve_implicitly(sequential_target, torch.zeros(3, dtype=dtype))
        sequential_gradient = torch.autograd.grad((weights[index] * sequential).sum(), sequential_target)[0]
        torch.testing.assert_close(batched[index], sequential_gradient, atol=2e-10, rtol=2e-10)


def test_manifold_output_cotangent_is_mapped_through_the_local_chart() -> None:
    dtype = torch.float64
    target_tensor = torch.tensor([0.2, -0.1, 0.3], dtype=dtype, requires_grad=True)
    rotation = SO3Variable(torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype), name="rotation")
    target = Variable(target_tensor, name="target", trainable=False)

    @residual(rotation, target, dim=3, name="rotation_target")
    def rotation_target(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return so3.log(value) - desired

    LevenbergMarquardt(
        Problem([rotation_target]),
        max_iterations=30,
        tolerance=1e-9,
        step_tolerance=1e-12,
        relative_tolerance=1e-12,
    ).optimize(differentiate="implicit")
    gradient = torch.autograd.grad(so3.log(rotation.tensor).sum(), target_tensor)[0]

    torch.testing.assert_close(so3.log(rotation.tensor), target_tensor, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(gradient, torch.ones_like(target_tensor), atol=2e-9, rtol=2e-9)


def test_exact_pi_so3_terminal_is_rejected_as_a_log_branch_cut() -> None:
    dtype = torch.float64
    target_tensor = torch.tensor([torch.pi, 0.0, 0.0], dtype=dtype, requires_grad=True)
    rotation = SO3Variable(so3.exp(target_tensor.detach()), name="rotation")
    target = Variable(target_tensor, name="target", trainable=False)

    @residual(rotation, target, dim=3, name="rotation_target")
    def rotation_target(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return so3.log(value) - desired

    info = LevenbergMarquardt(
        Problem([rotation_target]),
        max_iterations=2,
        tolerance=1e-10,
    ).optimize(differentiate="implicit")
    assert int(info.status) == int(OptimizerStatus.CONVERGED)

    with pytest.raises(ImplicitDifferentiationError, match="absolute-pi principal-log branch cut"):
        so3.log(rotation.tensor).sum().backward()


def test_exact_pi_se3_terminal_is_rejected_as_a_log_branch_cut() -> None:
    dtype = torch.float64
    target_tensor = torch.tensor(
        [0.0, 0.0, 0.0, torch.pi, 0.0, 0.0],
        dtype=dtype,
        requires_grad=True,
    )
    pose = SE3Variable(se3.exp(target_tensor.detach()), name="pose")
    target = Variable(target_tensor, name="target", trainable=False)

    @residual(pose, target, dim=6, name="pose_target")
    def pose_target(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return se3.log(value) - desired

    info = LevenbergMarquardt(
        Problem([pose_target]),
        max_iterations=2,
        tolerance=1e-10,
    ).optimize(differentiate="implicit")
    assert int(info.status) == int(OptimizerStatus.CONVERGED)

    with pytest.raises(ImplicitDifferentiationError, match="absolute-pi principal-log branch cut"):
        se3.log(pose.tensor).sum().backward()


@pytest.mark.parametrize("quaternion_event", [0, 1])
def test_robot_variable_checks_every_quaternion_event_for_log_branch_cut(
    quaternion_event: int,
) -> None:
    dtype = torch.float64
    builder = ModelBuilder("implicit_branch_cuts")
    base = builder.add_body("base", mass=1.0)
    tip = builder.add_body("tip", mass=1.0)
    builder.add_free_flyer_root("floating", child=base)
    builder.add_spherical("ball", parent=base, child=tip)
    model = build_model(builder.finalize(), dtype=dtype)
    layout = RobotVariable(model, model.q_neutral, name="layout", trainable=False)
    quaternion_slices = tuple(
        unit_slice for unit_slice in layout.unit_coordinate_slices if unit_slice.stop - unit_slice.start == 4
    )
    assert len(quaternion_slices) == 2

    exact = model.q_neutral.clone()
    exact[quaternion_slices[quaternion_event]] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        dtype=dtype,
    )
    target_tensor = model.difference(model.q_neutral, exact).detach().requires_grad_()
    q = RobotVariable(model, exact, name="q")
    target = Variable(target_tensor, name="target", trainable=False)

    @residual(q, target, dim=model.nv, name="robot_tangent")
    def robot_tangent(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        neutral = model.q_neutral.to(dtype=value.dtype, device=value.device)
        return model.difference(neutral, value) - desired

    info = LevenbergMarquardt(
        Problem([robot_tangent]),
        max_iterations=2,
        tolerance=1e-10,
    ).optimize(differentiate="implicit")
    assert int(info.status) == int(OptimizerStatus.CONVERGED)

    with pytest.raises(ImplicitDifferentiationError, match="absolute-pi principal-log branch cut"):
        model.difference(model.q_neutral, q.tensor).sum().backward()


def test_robot_variable_nq_ne_nv_maps_ambient_output_cotangent() -> None:
    dtype = torch.float64
    builder = ModelBuilder("implicit_free_flyer")
    base = builder.add_body("base", mass=1.0)
    builder.add_free_flyer_root("floating", child=base)
    model = build_model(builder.finalize(), dtype=dtype)
    assert model.nq == 7 and model.nv == 6
    target_tensor = torch.tensor(
        [0.1, -0.05, 0.08, 0.02, -0.03, 0.04],
        dtype=dtype,
        requires_grad=True,
    )
    q = RobotVariable(model, model.q_neutral.clone(), name="q")
    target = Variable(target_tensor, name="target", trainable=False)

    @residual(q, target, dim=model.nv, name="robot_tangent")
    def robot_tangent(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return model.difference(model.q_neutral, value) - desired

    LevenbergMarquardt(
        Problem([robot_tangent]),
        max_iterations=30,
        tolerance=1e-9,
        linearization="dense",
    ).optimize(differentiate="implicit")
    tangent = model.difference(model.q_neutral, q.tensor)
    gradient = torch.autograd.grad(tangent.sum(), target_tensor)[0]

    torch.testing.assert_close(tangent, target_tensor, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(gradient, torch.ones_like(target_tensor), atol=2e-9, rtol=2e-9)


def test_stable_active_bound_has_zero_sensitivity_while_free_coordinate_differentiates() -> None:
    dtype = torch.float64
    target_tensor = torch.tensor([2.0, 0.3], dtype=dtype, requires_grad=True)
    bounds = Bounds(
        lower=torch.tensor([-torch.inf, -torch.inf], dtype=dtype),
        upper=torch.tensor([1.0, torch.inf], dtype=dtype),
    )
    solution, _target, problem = _target_graph(
        target_tensor,
        torch.zeros(2, dtype=dtype),
        bounds=bounds,
    )
    info = LevenbergMarquardt(
        problem,
        max_iterations=30,
        tolerance=1e-7,
        step_tolerance=1e-12,
        relative_tolerance=1e-12,
    ).optimize(differentiate="implicit")
    gradient = torch.autograd.grad(solution.tensor.sum(), target_tensor)[0]

    assert int(info.status) == int(OptimizerStatus.STALLED_AT_BOUNDS)
    torch.testing.assert_close(gradient, torch.tensor([0.0, 1.0], dtype=dtype), atol=2e-9, rtol=2e-9)


def test_identity_only_weight_binding_is_rejected_as_disconnected() -> None:
    weight_tensor = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    solution = Variable(torch.zeros(1, dtype=torch.float64), name="x")
    weight = Variable(weight_tensor, name="weight", trainable=False)

    @residual(solution, weight, dim=1, weight=weight_tensor, name="fixed_target")
    def fixed_target(value: torch.Tensor, unused_weight: torch.Tensor) -> torch.Tensor:
        del unused_weight
        return value - 1.0

    @residual(solution, dim=1, weight=0.5, name="prior")
    def prior(value: torch.Tensor) -> torch.Tensor:
        return value

    LevenbergMarquardt(
        Problem([fixed_target, prior]),
        max_iterations=30,
        tolerance=1e-9,
    ).optimize(differentiate="implicit")

    with pytest.raises(ImplicitDifferentiationError, match="disconnected.*weight"):
        solution.tensor.sum().backward()


def test_huber_backward_uses_exact_robust_optimality() -> None:
    dtype = torch.float64
    scale_tensor = torch.tensor(1.0, dtype=dtype, requires_grad=True)
    solution = Variable(torch.zeros(1, dtype=dtype), name="x")
    scale = Variable(scale_tensor, name="scale", trainable=False)

    @residual(solution, scale, dim=1, kernel=Huber(delta=1.0), name="outlier")
    def outlier(value: torch.Tensor, multiplier: torch.Tensor) -> torch.Tensor:
        return multiplier * value - 10.0

    @residual(solution, dim=1, weight=0.5, name="prior")
    def prior(value: torch.Tensor) -> torch.Tensor:
        return value

    LevenbergMarquardt(
        Problem([outlier, prior]),
        max_iterations=100,
        tolerance=1e-6,
        step_tolerance=1e-12,
        relative_tolerance=1e-12,
    ).optimize(differentiate="implicit")
    gradient = torch.autograd.grad(solution.tensor.sum(), scale_tensor)[0]

    torch.testing.assert_close(solution.tensor, torch.tensor([4.0], dtype=dtype), atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(gradient, torch.tensor(4.0, dtype=dtype), atol=2e-7, rtol=2e-7)


def test_huber_kink_is_strictly_rejected() -> None:
    dtype = torch.float64
    target_tensor = torch.tensor([2.0], dtype=dtype, requires_grad=True)
    solution = Variable(torch.ones(1, dtype=dtype), name="x")
    target = Variable(target_tensor, name="target", trainable=False)

    @residual(solution, dim=1, kernel=Huber(delta=1.0), name="kink")
    def kink(value: torch.Tensor) -> torch.Tensor:
        return value

    @residual(solution, target, dim=1, name="balance")
    def balance(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return value - desired

    info = LevenbergMarquardt(
        Problem([kink, balance]),
        max_iterations=1,
        tolerance=1e-9,
        linearization="dense",
    ).optimize(differentiate="implicit")
    assert int(info.status) == int(OptimizerStatus.CONVERGED)

    with pytest.raises(ImplicitDifferentiationError, match="nonsmooth"):
        solution.tensor.sum().backward()


def test_rank_deficient_backward_is_strictly_rejected() -> None:
    dtype = torch.float64
    target_tensor = torch.tensor([0.3], dtype=dtype, requires_grad=True)
    solution = Variable(torch.zeros(2, dtype=dtype), name="x")
    target = Variable(target_tensor, name="target", trainable=False)

    @residual(solution, target, dim=1, name="rank_deficient")
    def rank_deficient(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return value[..., :1] - desired

    LevenbergMarquardt(
        Problem([rank_deficient]),
        max_iterations=30,
        tolerance=1e-9,
        linearization="dense",
    ).optimize(differentiate="implicit")

    with pytest.raises(ImplicitDifferentiationError, match="singular"):
        solution.tensor.sum().backward()


def test_mixed_batch_rejects_every_gradient_if_one_terminal_status_is_invalid() -> None:
    target_tensor = torch.tensor([[0.0], [0.4]], dtype=torch.float64, requires_grad=True)
    solution, _target, problem = _target_graph(target_tensor, torch.zeros_like(target_tensor))
    info = LevenbergMarquardt(problem, max_iterations=0, tolerance=1e-9).optimize(differentiate="implicit")
    torch.testing.assert_close(
        info.status,
        torch.tensor([OptimizerStatus.CONVERGED, OptimizerStatus.MAXITER], dtype=torch.int8),
    )

    with pytest.raises(ImplicitDifferentiationError, match="invalid batch indices") as caught:
        solution.tensor.sum().backward()

    assert caught.value.invalid_indices == ((1,),)
    assert caught.value.statuses == ("maxiter",)


class _TemporalTargetResidual(Residual):
    def __init__(self, x: Variable, target: Variable) -> None:
        self.x = x
        self.target = target
        self.horizon = x.time_length
        super().__init__(x, target, dim=self.horizon, name="temporal_target")

    def error(self) -> torch.Tensor:
        value = self.x.tensor
        return (value[..., :, 0] - self.target.tensor).reshape(*value.shape[:-2], self.horizon)

    def temporal_structure(self, variable: Variable | str) -> TemporalPattern | None:
        if variable is not self.x and variable != self.x.name:
            return None
        return TemporalPattern(self.horizon, 1, 0, (0,))

    def temporal_jacobian_blocks(self, variable: Variable | str) -> Mapping[int, torch.Tensor]:
        if variable is not self.x and variable != self.x.name:
            return {}
        value = self.x.tensor
        return {0: value.new_ones(*value.shape[:-2], self.horizon, 1, 1)}


def _temporal_problem(
    target_tensor: torch.Tensor,
) -> tuple[Variable, Problem]:
    horizon = target_tensor.shape[0]
    solution = Variable(
        torch.zeros(horizon, 1, dtype=target_tensor.dtype),
        name="x",
        time_axis=0,
    )
    target = Variable(target_tensor, name="target", trainable=False)
    return solution, Problem([_TemporalTargetResidual(solution, target)])


def test_small_banded_forward_requires_explicit_dense_backward_opt_in_and_matches_dense() -> None:
    dtype = torch.float64
    target = torch.linspace(-0.3, 0.6, 4, dtype=dtype, requires_grad=True)
    _rejected_solution, rejected_problem = _temporal_problem(target)
    rejected_optimizer = LevenbergMarquardt(
        rejected_problem,
        max_iterations=20,
        tolerance=1e-9,
        linearization="structured",
    )
    with pytest.raises(ValueError, match="allow_banded_dense_backward"):
        rejected_optimizer.optimize(differentiate="implicit")

    dense_solution, dense_problem = _temporal_problem(target)
    banded_solution, banded_problem = _temporal_problem(target)
    LevenbergMarquardt(
        dense_problem,
        max_iterations=20,
        tolerance=1e-9,
        linearization="dense",
    ).optimize(differentiate="implicit")
    LevenbergMarquardt(
        banded_problem,
        max_iterations=20,
        tolerance=1e-9,
        linearization="structured",
    ).optimize(
        differentiate="implicit",
        implicit_config=ImplicitDiffConfig(allow_banded_dense_backward=True),
    )
    dense_gradient = torch.autograd.grad(dense_solution.tensor.sum(), target, retain_graph=True)[0]
    banded_gradient = torch.autograd.grad(banded_solution.tensor.sum(), target)[0]

    torch.testing.assert_close(banded_solution.tensor, dense_solution.tensor, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(banded_gradient, dense_gradient, atol=2e-10, rtol=2e-10)


def test_large_dense_materialization_is_rejected_before_backward() -> None:
    target = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    _solution, _target, problem = _target_graph(target, torch.zeros_like(target))
    optimizer = LevenbergMarquardt(problem, max_iterations=5)

    with pytest.raises(ValueError, match="configured cap is 2"):
        optimizer.optimize(
            differentiate="implicit",
            implicit_config=ImplicitDiffConfig(max_dense_tangent_dim=2),
        )
