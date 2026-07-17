"""Regression tests for evaluation-local IK state caching."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import better_robot.optim.blocks.providers as provider_module
from better_robot.costs.stack import CostStack
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim.problem import LeastSquaresProblem
from better_robot.optim import Problem, ResidualItem, RobotConfig, RobotStateProvider, VarSpec
from better_robot.residuals.pose import PoseResidual
from better_robot.residuals.base import ResidualState


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
    builder.add_revolute_z("j1", parent="base", child="link1", origin=identity, lower=-2.5, upper=2.5)
    builder.add_revolute_y("j2", parent="link1", child="link2", origin=offset, lower=-2.0, upper=2.0)
    builder.add_revolute_y("j3", parent="link2", child="link3", origin=offset, lower=-2.0, upper=2.0)
    return build_model(builder.finalize())


def test_block_kinematic_residuals_share_one_fk_per_context(monkeypatch) -> None:
    model = _make_arm()
    q = torch.tensor([1.0, -0.8, 0.6])
    data = forward_kinematics(model, q, compute_frames=True)

    fk_calls = 0
    original_fk = provider_module.forward_kinematics

    def counted_fk(*args, **kwargs):
        nonlocal fk_calls
        fk_calls += 1
        return original_fk(*args, **kwargs)

    monkeypatch.setattr(provider_module, "forward_kinematics", counted_fk)
    residuals = []
    for link in ("body_link2", "body_link3"):
        name = f"pose_{link}"
        residuals.append(
            ResidualItem(
                name,
                PoseResidual(
                    frame_id=model.frame_id(link),
                    target=data.frame_pose_world[model.frame_id(link)],
                    model=model,
                    name=name,
                ),
            )
        )
    problem = Problem(
        vars=(VarSpec("q", (model.nq,), manifold=RobotConfig(model)),),
        residuals=tuple(residuals),
        providers=(RobotStateProvider(model),),
    )

    problem.residual({"q": q})
    assert fk_calls == 1
    problem.jacobian_blocks({"q": q})
    assert fk_calls == 2


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
