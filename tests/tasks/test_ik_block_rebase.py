"""Named-block IK facade contracts and built-in residual parity."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

import better_robot.tasks.ik as ik_module
from better_robot.io import ModelBuilder, build_model, load
from better_robot.kinematics import forward_kinematics
from better_robot.optim import (
    LevenbergMarquardt,
    Problem,
    RobotVariable,
    Variable,
)
from better_robot.residuals.limits import JointPositionLimit
from better_robot.residuals.nodes import RobotState
from better_robot.residuals.pose import OrientationResidual, PoseResidual, PositionResidual
from better_robot.residuals.regularization import RestResidual
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik


@pytest.fixture(scope="module")
def panda():
    panda_description = pytest.importorskip("robot_descriptions.panda_description")
    return load(panda_description.URDF_PATH)


def _panda_hand(model) -> int:
    for name in ("body_panda_hand", "body_panda_link8", "body_panda_link7"):
        if name in model.frame_name_to_id:
            return model.frame_id(name)
    raise AssertionError("Panda hand frame is missing")


def test_built_in_kinematic_residuals_share_object_protocol(panda) -> None:
    """Built-ins compose through object references and analytic tangent J."""
    q = panda.q_neutral.clone()
    q_rest = q.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    q_rest[0] = 0.2
    target_data = forward_kinematics(panda, q_rest, compute_frames=True)
    frame_id = _panda_hand(panda)
    target = target_data.frame_pose_world[frame_id].clone()
    q_variable = RobotVariable(panda, q, name="q")
    state = RobotState(q_variable)
    target_variable = Variable(target, name="target", trainable=False)
    rest_variable = Variable(q_rest, name="q_rest", trainable=False)
    residuals = (
        PoseResidual(
            state,
            frame_id=frame_id,
            target=target_variable,
            pos_weight=0.7,
            ori_weight=1.3,
            name="pose",
        ),
        PositionResidual(
            state,
            frame_id=frame_id,
            target=target_variable,
            row_weight=0.6,
            name="position",
        ),
        OrientationResidual(
            state,
            frame_id=frame_id,
            target=target_variable,
            row_weight=0.8,
            name="orientation",
        ),
        JointPositionLimit(q_variable, row_weight=0.4, name="limits"),
        RestResidual(q_variable, rest_variable, row_weight=0.2, name="rest"),
    )
    problem = Problem(residuals)

    expected_residual = torch.cat([residual.weighted_error() for residual in residuals], dim=-1)

    torch.testing.assert_close(
        problem.error(),
        expected_residual,
        atol=1e-6,
        rtol=1e-6,
    )
    jacobian = problem.dense_jacobian(strategy="analytic")
    assert jacobian.shape == (problem.dim_total, panda.nv)
    assert torch.isfinite(jacobian).all()


def test_direct_builder_ik_auto_wires_robot_state() -> None:
    builder = ModelBuilder("direct_builder_ik")
    base = builder.add_body("base")
    tip = builder.add_body("tip", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent=base,
        child=tip,
        origin=torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-1.0,
        upper=1.0,
    )
    model = build_model(builder.finalize())
    goal = torch.tensor([0.25])
    target = forward_kinematics(model, goal, compute_frames=True).frame_pose_world[model.frame_id("body_tip")]

    q = RobotVariable(model, torch.tensor([-0.2]), bounds=True)
    problem = Problem([PoseResidual(q, frame="body_tip", target=target)])
    LevenbergMarquardt(problem, max_iterations=30).optimize()

    torch.testing.assert_close(q.tensor, goal, atol=2e-4, rtol=2e-4)


def test_solve_ik_path_has_no_legacy_problem_import() -> None:
    path = Path(ik_module.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}

    assert not any(module.endswith("optim.problem") for module in imported_modules)
    forbidden = {
        "EvaluationContext",
        "LeastSquaresProblem",
        "ResidualItem",
        "RobotStateProvider",
        "VarSpec",
        "run_first_order",
    }
    assert forbidden.isdisjoint(imported_names)
    assert "LeastSquaresProblem" not in source


def test_facade_declares_and_reads_static_target_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = ModelBuilder("target_parameter_facade")
    base = builder.add_body("base")
    tip = builder.add_body("tip", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent=base,
        child=tip,
        origin=torch.tensor([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-1.0,
        upper=1.0,
    )
    model = build_model(builder.finalize())
    target = (
        forward_kinematics(model, torch.tensor([0.25]), compute_frames=True)
        .frame_pose_world[model.frame_id("body_tip")]
        .detach()
        .clone()
        .requires_grad_()
    )
    captured: dict[str, Problem] = {}
    original_problem = ik_module.Problem

    def record_problem(*args, **kwargs) -> Problem:
        problem = original_problem(*args, **kwargs)
        captured["problem"] = problem
        return problem

    monkeypatch.setattr(ik_module, "Problem", record_problem)
    initial = torch.tensor([-0.15])
    solve_ik(
        model,
        {"body_tip": target},
        initial_q=initial,
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=OptimizerConfig(max_iter=0),
    )

    problem = captured["problem"]
    target_variable = problem.variables["target_pose_0"]
    assert target_variable.tensor is target
    assert target_variable.trainable is False
    assert target_variable in problem.residuals[0].variables
    target_gradient = torch.autograd.grad(
        problem.objective(),
        target,
    )[0]
    assert torch.isfinite(target_gradient).all()
    assert target_gradient.norm() > 0.0


def test_facade_declares_rest_target_only_when_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = ModelBuilder("rest_parameter_facade")
    base = builder.add_body("base")
    tip = builder.add_body("tip", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent=base,
        child=tip,
        lower=-1.0,
        upper=1.0,
    )
    model = build_model(builder.finalize())
    q_rest = torch.tensor([0.3], requires_grad=True)
    captured: list[Problem] = []
    original_problem = ik_module.Problem

    def record_problem(*args, **kwargs) -> Problem:
        problem = original_problem(*args, **kwargs)
        captured.append(problem)
        return problem

    monkeypatch.setattr(ik_module, "Problem", record_problem)
    solve_ik(
        model,
        {},
        initial_q=torch.tensor([0.0]),
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=1.0, q_rest=q_rest),
        optimizer_cfg=OptimizerConfig(max_iter=0),
    )

    active_problem = captured[-1]
    rest_variable = active_problem.variables["target_rest"]
    assert rest_variable.tensor is q_rest
    assert rest_variable.trainable is False
    assert rest_variable in active_problem.residuals[0].variables
    active_problem.update({"q": torch.tensor([0.1])})
    rest_gradient = torch.autograd.grad(
        active_problem.objective(),
        q_rest,
    )[0]
    assert torch.isfinite(rest_gradient).all()
    assert rest_gradient.norm() > 0.0

    # Disabled terms are outside the problem contract and must not affect
    # batch broadcasting or validation.
    solve_ik(
        model,
        {},
        initial_q=torch.zeros(2, 1),
        cost_cfg=IKCostConfig(
            limit_weight=0.1,
            rest_weight=0.0,
            q_rest=torch.zeros(3, 1),
        ),
        optimizer_cfg=OptimizerConfig(max_iter=0),
    )
    disabled_problem = captured[-1]
    assert "target_rest" not in disabled_problem.variables


def test_bounds_active_target_converges_through_facade() -> None:
    builder = ModelBuilder("bounded_facade")
    base = builder.add_body("base")
    tip = builder.add_body("tip", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent=base,
        child=tip,
        origin=torch.tensor([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-0.2,
        upper=0.2,
    )
    model = build_model(builder.finalize())
    q_target = torch.tensor([0.2])
    target = forward_kinematics(
        model,
        q_target,
        compute_frames=True,
    ).frame_pose_world[model.frame_id("body_tip")]

    result = solve_ik(
        model,
        {"body_tip": target},
        initial_q=torch.tensor([-0.15]),
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=OptimizerConfig(max_iter=60, tol=1e-6),
    )

    assert result.converged is True
    assert result.q <= model.upper_pos_limit + 1e-7
    torch.testing.assert_close(result.q, q_target, atol=2e-5, rtol=0.0)


def test_finite_difference_configuration_remains_supported() -> None:
    builder = ModelBuilder("fd_facade")
    base = builder.add_body("base")
    tip = builder.add_body("tip", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent=base,
        child=tip,
        origin=torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    model = build_model(builder.finalize())
    target = forward_kinematics(
        model,
        torch.tensor([0.1]),
        compute_frames=True,
    ).frame_pose_world[model.frame_id("body_tip")]

    result = solve_ik(
        model,
        {"body_tip": target},
        initial_q=torch.zeros(1),
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=OptimizerConfig(
            max_iter=20,
            jacobian_strategy="finite_difference",
        ),
    )

    assert result.converged is True
    torch.testing.assert_close(result.q, torch.tensor([0.1]), atol=5e-4, rtol=0.0)
