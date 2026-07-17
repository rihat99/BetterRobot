"""Regression tests for evaluation-local IK state caching."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import better_robot.tasks.ik as ik_module
from better_robot.costs.stack import CostStack
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim.problem import LeastSquaresProblem
from better_robot.residuals.base import ResidualState


class _UncachedProblem(LeastSquaresProblem):
    """Pre-T1.11 behavior used only as the numerical/count baseline."""

    def _state_at(self, x: torch.Tensor) -> ResidualState:
        return self.state_factory(x)


class _DerivedResidual:
    name = "derived"
    dim = 1

    def __call__(self, state: ResidualState) -> torch.Tensor:
        return state.data.derived

    def jacobian(self, state: ResidualState) -> torch.Tensor:
        return 2.0 * state.variables.unsqueeze(-2)


def _make_arm():
    builder = ModelBuilder("cache_arm")
    builder.add_body("base", mass=1.0)
    for link_index in range(1, 4):
        builder.add_body(f"link{link_index}", mass=1.0)

    identity = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    offset = torch.tensor([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    builder.add_revolute_z(
        "j1", parent="base", child="link1", origin=identity, lower=-2.5, upper=2.5
    )
    builder.add_revolute_y(
        "j2", parent="link1", child="link2", origin=offset, lower=-2.0, upper=2.0
    )
    builder.add_revolute_y(
        "j3", parent="link2", child="link3", origin=offset, lower=-2.0, upper=2.0
    )
    return build_model(builder.finalize())


def _solve_fixed_iterations(model, target: torch.Tensor):
    return ik_module.solve_ik(
        model,
        {"body_link3": target},
        initial_q=torch.zeros(model.nq),
        cost_cfg=ik_module.IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=ik_module.OptimizerConfig(max_iter=5, tol=0.0),
    )


def test_solve_ik_reuses_fk_without_changing_solution(monkeypatch) -> None:
    model = _make_arm()
    q_target = torch.tensor([1.0, -0.8, 0.6])
    target = forward_kinematics(
        model, q_target, compute_frames=True
    ).frame_pose_world[model.frame_id("body_link3")].clone()

    fk_calls = 0
    original_fk = ik_module.forward_kinematics

    def counted_fk(*args, **kwargs):
        nonlocal fk_calls
        fk_calls += 1
        return original_fk(*args, **kwargs)

    monkeypatch.setattr(ik_module, "forward_kinematics", counted_fk)
    monkeypatch.setattr(ik_module, "LeastSquaresProblem", _UncachedProblem)
    uncached = _solve_fixed_iterations(model, target)
    uncached_fk_calls = fk_calls

    fk_calls = 0
    monkeypatch.setattr(ik_module, "LeastSquaresProblem", LeastSquaresProblem)
    cached = _solve_fixed_iterations(model, target)

    assert uncached.iters == cached.iters == 5
    assert uncached_fk_calls == 1 + 2 * uncached.iters
    assert fk_calls == 1 + cached.iters
    assert cached.converged is uncached.converged
    torch.testing.assert_close(cached.q, uncached.q, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(cached.residual, uncached.residual, atol=1e-7, rtol=0.0)


def test_state_cache_invalidates_after_in_place_iterate_change() -> None:
    stack = CostStack()
    stack.add("derived", _DerivedResidual())
    factory_calls = 0

    def state_factory(x: torch.Tensor) -> ResidualState:
        nonlocal factory_calls
        factory_calls += 1
        data = SimpleNamespace(derived=x.square())
        model = SimpleNamespace(nv=1)
        return ResidualState(model=model, data=data, variables=x)

    x = torch.tensor([2.0])
    problem = LeastSquaresProblem(cost_stack=stack, state_factory=state_factory, x0=x)

    first = problem.residual(x)
    x.add_(1.0)
    second = problem.residual(x)

    assert factory_calls == 2
    torch.testing.assert_close(first, torch.tensor([4.0]))
    torch.testing.assert_close(second, torch.tensor([9.0]))


def test_state_cache_does_not_retain_autograd_graphs() -> None:
    stack = CostStack()
    stack.add("derived", _DerivedResidual())
    factory_calls = 0

    def state_factory(x: torch.Tensor) -> ResidualState:
        nonlocal factory_calls
        factory_calls += 1
        data = SimpleNamespace(derived=x.square())
        model = SimpleNamespace(nv=1)
        return ResidualState(model=model, data=data, variables=x)

    x = torch.tensor([2.0], requires_grad=True)
    problem = LeastSquaresProblem(cost_stack=stack, state_factory=state_factory, x0=x)

    first_gradient = torch.autograd.grad(problem.residual(x).sum(), x)[0]
    second_gradient = torch.autograd.grad(problem.residual(x).sum(), x)[0]

    assert factory_calls == 2
    torch.testing.assert_close(first_gradient, torch.tensor([4.0]))
    torch.testing.assert_close(second_gradient, first_gradient)


def test_state_cache_does_not_retain_captured_parameter_graphs() -> None:
    stack = CostStack()
    stack.add("derived", _DerivedResidual())
    factory_calls = 0
    scale = torch.tensor(3.0, requires_grad=True)

    def state_factory(x: torch.Tensor) -> ResidualState:
        nonlocal factory_calls
        factory_calls += 1
        data = SimpleNamespace(derived=scale * x.square())
        model = SimpleNamespace(nv=1)
        return ResidualState(model=model, data=data, variables=x)

    x = torch.tensor([2.0])
    problem = LeastSquaresProblem(cost_stack=stack, state_factory=state_factory, x0=x)

    first_gradient = torch.autograd.grad(problem.residual(x).sum(), scale)[0]
    second_gradient = torch.autograd.grad(problem.residual(x).sum(), scale)[0]

    assert factory_calls == 2
    torch.testing.assert_close(first_gradient, torch.tensor(4.0))
    torch.testing.assert_close(second_gradient, first_gradient)
