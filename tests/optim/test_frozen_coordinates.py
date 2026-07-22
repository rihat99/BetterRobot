"""Optimizer-facing regressions for frozen robot tangent groups."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from better_robot.data_model import Model
from better_robot.io import ModelBuilder, build_model
from better_robot.optim import (
    Bounds,
    ImplicitDiffConfig,
    LevenbergMarquardt,
    OptimizerStatus,
    Problem,
    Residual,
    RobotVariable,
    TemporalPattern,
    TorchOptimizer,
    Variable,
    residual,
)

_DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=[
            pytest.mark.cuda,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ],
    ),
]


def _two_joint_model(device: str = "cpu") -> Model:
    builder = ModelBuilder("frozen_optimizer")
    base = builder.add_body("base")
    middle = builder.add_body("middle")
    tip = builder.add_body("tip")
    builder.add_revolute_z(
        "free_joint",
        parent=base,
        child=middle,
        lower=-0.25,
        upper=0.25,
    )
    builder.add_revolute_y(
        "frozen_joint",
        parent=middle,
        child=tip,
        lower=-0.1,
        upper=0.1,
    )
    return build_model(builder.finalize(), device=torch.device(device))


class _TangentTargetResidual(Residual):
    def __init__(self, q: RobotVariable, target: Variable, *, name: str = "tangent_target") -> None:
        self.q = q
        self.target = target
        self.model = q.model
        super().__init__(q, target, dim=self.model.nv, name=name)

    def error(self) -> torch.Tensor:
        neutral = self.model.q_neutral.to(self.q.tensor)
        return self.model.difference(neutral, self.q.tensor) - self.target.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...] | None:
        identity = torch.eye(self.model.nv, dtype=self.q.tensor.dtype, device=self.q.tensor.device)
        return (identity,)


class _AutodiffTangentTargetResidual(_TangentTargetResidual):
    def jacobian(self) -> None:
        return None


class _TrajectoryTargetResidual(Residual):
    def __init__(self, q: RobotVariable, target: Variable) -> None:
        self.q = q
        self.target = target
        self.model = q.model
        self.horizon = q.time_length
        super().__init__(q, target, dim=self.horizon * self.model.nv, name="trajectory_target")

    def error(self) -> torch.Tensor:
        neutral = self.model.q_neutral.to(self.q.tensor).expand(self.horizon, -1)
        rows = self.model.difference(neutral, self.q.tensor) - self.target.tensor
        return rows.reshape(self.dim)

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        return (torch.eye(self.dim, dtype=self.q.tensor.dtype, device=self.q.tensor.device),)

    def temporal_structure(self, variable: RobotVariable | str) -> TemporalPattern | None:
        if variable is not self.q and variable != self.q.name:
            return None
        return TemporalPattern(self.horizon, self.model.nv, 0, (0,))

    def temporal_jacobian_blocks(self, variable: RobotVariable | str) -> Mapping[int, torch.Tensor]:
        if variable is not self.q and variable != self.q.name:
            return {}
        identity = torch.eye(self.model.nv, dtype=self.q.tensor.dtype, device=self.q.tensor.device)
        return {0: identity.expand(self.horizon, self.model.nv, self.model.nv)}


def _frozen_problem(
    model: Model,
    target_tensor: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
    bounds: bool = False,
) -> tuple[RobotVariable, Problem]:
    q = RobotVariable(
        model,
        model.q_neutral.clone() if initial is None else initial,
        name="q",
        bounds=bounds,
        frozen_groups=("frozen_joint",),
    )
    target = Variable(target_tensor, name="target", trainable=False)
    return q, Problem([_TangentTargetResidual(q, target)])


@pytest.mark.parametrize("device", _DEVICES)
def test_problem_reduces_full_analytic_blocks_and_optimizer_tangents(device: str) -> None:
    model = _two_joint_model(device)
    q, problem = _frozen_problem(model, torch.tensor([0.2, 0.8], device=device))

    block = problem.jacobian_blocks(strategy="analytic")[("tangent_target", "q")]
    moved = model.integrate(model.q_neutral, torch.tensor([0.3, 0.7], device=device))
    difference = problem.difference({"q": model.q_neutral}, {"q": moved})["q"]

    torch.testing.assert_close(block, torch.tensor([[1.0], [0.0]], device=device))
    torch.testing.assert_close(problem.gradient()["q"], torch.tensor([-0.2], device=device))
    torch.testing.assert_close(difference, torch.tensor([0.3], device=device))
    assert all(index.device == q.tensor.device for index in q.tangent_groups().values())
    assert q.difference(model.q_neutral).shape == (model.nv,)


@pytest.mark.parametrize("device", _DEVICES)
def test_lm_bounds_ignore_and_preserve_frozen_coordinate(device: str) -> None:
    model = _two_joint_model(device)
    initial = model.integrate(model.q_neutral, torch.tensor([0.0, 0.35], device=device))
    q, problem = _frozen_problem(
        model,
        torch.tensor([0.8, -0.4], device=device),
        initial=initial,
        bounds=True,
    )
    optimizer = LevenbergMarquardt(problem, max_iterations=40, tolerance=1e-6)

    info = optimizer.optimize()
    assert optimizer._state is not None
    tangent = model.difference(model.q_neutral, q.tensor)

    torch.testing.assert_close(tangent[0], torch.tensor(0.25, device=device), atol=2e-5, rtol=2e-5)
    assert torch.equal(tangent[1], torch.tensor(0.35, device=device))
    torch.testing.assert_close(optimizer._state.bound_state_index, torch.tensor([0], device=device))
    torch.testing.assert_close(optimizer._state.bound_lower, torch.tensor([-0.25], device=device))
    torch.testing.assert_close(optimizer._state.bound_upper, torch.tensor([0.25], device=device))
    torch.testing.assert_close(optimizer._state.bounded_mask, torch.tensor([True], device=device))
    assert int(info.status) == int(OptimizerStatus.STALLED_AT_BOUNDS)


def test_frozen_root_translation_makes_world_axis_bounds_irrelevant() -> None:
    builder = ModelBuilder("frozen_root_bounds")
    base = builder.add_body("base")
    builder.add_free_flyer_root("root", child=base)
    model = build_model(builder.finalize())
    lower = torch.full((model.nq,), -torch.inf)
    upper = torch.full((model.nq,), torch.inf)
    lower[:3], upper[:3] = -1.0, 1.0
    initial = model.q_neutral.clone()
    initial[:3] = torch.tensor([1.5, -1.5, 2.0])
    q = RobotVariable(
        model,
        initial.clone(),
        name="q",
        bounds=Bounds(lower, upper),
        frozen_groups=("root_lin",),
    )
    target = Variable(torch.zeros(model.nv), name="target", trainable=False)
    problem = Problem([_TangentTargetResidual(q, target)])

    info = LevenbergMarquardt(problem, max_iterations=2, tolerance=1e-6).optimize()

    assert torch.equal(q.tensor[:3], initial[:3])
    assert int(info.status) == int(OptimizerStatus.CONVERGED)


def test_two_phase_floating_warmup_hands_off_to_fresh_unfrozen_problem() -> None:
    builder = ModelBuilder("floating_warmup")
    base = builder.add_body("base")
    limb = builder.add_body("limb")
    tip = builder.add_body("tip")
    builder.add_free_flyer_root("root", child=base)
    builder.add_spherical("ball", parent=base, child=limb)
    builder.add_revolute_z("hinge", parent=limb, child=tip)
    model = build_model(builder.finalize())
    target_tensor = torch.tensor([0.1, -0.05, 0.08, 0.02, -0.03, 0.04, 0.05, -0.04, 0.03, 0.15])

    phase_one = RobotVariable(model, model.q_neutral.clone(), name="q", frozen_groups=("joints",))
    phase_one_target = Variable(target_tensor, name="target", trainable=False)
    phase_one_problem = Problem([_AutodiffTangentTargetResidual(phase_one, phase_one_target)])
    first_info = LevenbergMarquardt(
        phase_one_problem,
        max_iterations=40,
        tolerance=1e-6,
        jacobian_strategy="jacrev",
    ).optimize()
    first_tangent = model.difference(model.q_neutral, phase_one.tensor)
    groups = phase_one.tangent_groups()

    torch.testing.assert_close(first_tangent[groups["root"]], target_tensor[groups["root"]], atol=2e-5, rtol=2e-5)
    assert torch.equal(first_tangent[groups["joints"]], torch.zeros_like(first_tangent[groups["joints"]]))
    assert int(first_info.status) == int(OptimizerStatus.CONVERGED)

    phase_two = RobotVariable(model, phase_one.tensor.detach().clone(), name="q")
    phase_two_target = Variable(target_tensor, name="target", trainable=False)
    phase_two_problem = Problem([_AutodiffTangentTargetResidual(phase_two, phase_two_target)])
    second_info = LevenbergMarquardt(
        phase_two_problem,
        max_iterations=40,
        tolerance=1e-6,
        jacobian_strategy="jacrev",
    ).optimize()
    final_tangent = model.difference(model.q_neutral, phase_two.tensor)

    torch.testing.assert_close(final_tangent, target_tensor, atol=2e-5, rtol=2e-5)
    assert int(second_info.status) == int(OptimizerStatus.CONVERGED)


@pytest.mark.parametrize("device", _DEVICES)
def test_torch_optimizer_uses_only_free_tangent_buffers(device: str) -> None:
    model = _two_joint_model(device)
    initial = model.integrate(model.q_neutral, torch.tensor([0.0, 0.15], device=device))
    q, problem = _frozen_problem(model, torch.tensor([0.2, -0.4], device=device), initial=initial)
    optimizer = TorchOptimizer(
        problem,
        torch.optim.SGD,
        lr=0.2,
        max_iterations=80,
        tolerance=1e-6,
    )

    info = optimizer.optimize()
    tangent = model.difference(model.q_neutral, q.tensor)

    assert optimizer._buffers["q"].shape == (1,)
    torch.testing.assert_close(tangent[0], torch.tensor(0.2, device=device), atol=2e-5, rtol=2e-5)
    assert torch.equal(tangent[1], torch.tensor(0.15, device=device))
    assert bool(info.converged)


@pytest.mark.parametrize("device", _DEVICES)
def test_banded_and_dense_assembly_match_with_frozen_per_knot_group(device: str) -> None:
    model = _two_joint_model(device)
    horizon = 4
    target_tensor = torch.tensor(
        [[-0.2, 0.7], [-0.05, 0.6], [0.1, 0.5], [0.2, 0.4]],
        device=device,
    )
    initial_tangent = torch.tensor([[0.0, 0.15]], device=device).expand(horizon, -1).clone()
    initial = model.integrate(model.q_neutral.expand(horizon, -1), initial_tangent)
    outputs: dict[str, torch.Tensor] = {}

    for mode in ("dense", "structured"):
        q = RobotVariable(
            model,
            initial.clone(),
            name="q",
            time_axis=0,
            frozen_groups=("frozen_joint",),
        )
        target = Variable(target_tensor, name="target", trainable=False)
        problem = Problem([_TrajectoryTargetResidual(q, target)])
        dense = problem.dense_jacobian(strategy="analytic")
        structured = problem.structured_normal()

        assert problem.temporal_analysis.tangent_width == 1
        assert structured.normal.block_size == 1
        torch.testing.assert_close(structured.normal.densify(), dense.mT @ dense)

        LevenbergMarquardt(
            problem,
            max_iterations=30,
            tolerance=1e-6,
            linearization=mode,
        ).optimize()
        outputs[mode] = model.difference(model.q_neutral.expand(horizon, -1), q.tensor)

    torch.testing.assert_close(outputs["structured"], outputs["dense"], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(outputs["dense"][:, 0], target_tensor[:, 0], atol=2e-5, rtol=2e-5)
    assert torch.equal(outputs["dense"][:, 1], initial_tangent[:, 1])


@pytest.mark.parametrize("device", _DEVICES)
def test_implicit_backward_uses_only_free_solution_coordinates(device: str) -> None:
    model = _two_joint_model(device)
    target_tensor = torch.tensor([0.2, -0.4], device=device, requires_grad=True)
    initial = model.integrate(model.q_neutral, torch.tensor([0.0, 0.15], device=device))
    q, problem = _frozen_problem(model, target_tensor, initial=initial)

    info = LevenbergMarquardt(
        problem,
        max_iterations=30,
        tolerance=1e-6,
        step_tolerance=1e-12,
        relative_tolerance=1e-12,
    ).optimize(
        differentiate="implicit",
        implicit_config=ImplicitDiffConfig(max_dense_tangent_dim=1),
    )
    tangent = model.difference(model.q_neutral, q.tensor)
    gradient = torch.autograd.grad(tangent.sum(), target_tensor)[0]

    torch.testing.assert_close(tangent[0], target_tensor.detach()[0], atol=2e-5, rtol=2e-5)
    assert torch.equal(tangent[1], torch.tensor(0.15, device=device))
    torch.testing.assert_close(gradient, torch.tensor([1.0, 0.0], device=device), atol=2e-5, rtol=2e-5)
    assert int(info.status) == int(OptimizerStatus.CONVERGED)


def test_implicit_branch_cut_guard_ignores_fully_frozen_quaternion() -> None:
    builder = ModelBuilder("frozen_branch_cut")
    base_body = builder.add_body("base")
    builder.add_free_flyer_root("root", child=base_body)
    model = build_model(builder.finalize())
    base = model.q_neutral.clone()
    base[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    target_tensor = model.integrate(base, torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])).requires_grad_()
    q = RobotVariable(model, base, name="q", frozen_groups=("root_ang",))
    target = Variable(target_tensor, name="target", trainable=False)

    @residual(q, target, dim=model.nv, name="pose_target")
    def pose_target(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return model.difference(desired, value)

    info = LevenbergMarquardt(
        Problem([pose_target]),
        max_iterations=30,
        tolerance=1e-6,
        step_tolerance=1e-12,
        relative_tolerance=1e-12,
        jacobian_strategy="jacrev",
    ).optimize(differentiate="implicit")
    gradient = torch.autograd.grad(q.tensor[:3].sum(), target_tensor)[0]

    torch.testing.assert_close(q.tensor[:3], target_tensor[:3], atol=2e-5, rtol=2e-5)
    assert torch.equal(q.tensor[3:7], base[3:7])
    torch.testing.assert_close(gradient[:3], torch.ones(3), atol=2e-5, rtol=2e-5)
    assert bool(torch.isfinite(gradient).all())
    assert int(info.status) == int(OptimizerStatus.CONVERGED)
