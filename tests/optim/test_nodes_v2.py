"""Evaluation-scoped node memoization and dependency contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.optim.problem import Problem
from better_robot.optim.variables import RobotVariable, Variable
from better_robot.residuals.base import Residual
from better_robot.residuals.nodes import Node, RobotState


class _ScaleNode(Node):
    def __init__(self, variable: Variable, *, multiplier: float = 2.0) -> None:
        self.variable = variable
        self.multiplier = multiplier
        self.calls = 0
        super().__init__(variable)

    def compute(self) -> torch.Tensor:
        self.calls += 1
        return self.variable.tensor * self.multiplier


class _NodeResidual(Residual):
    def __init__(self, node: Node, *, name: str, weight=1.0) -> None:
        self.node = node
        self.nodes = (node,)
        super().__init__(dim=1, name=name, weight=weight)

    def error(self) -> torch.Tensor:
        return self.node.value()[..., :1]


def _one_joint_model():
    builder = ModelBuilder("node_arm")
    builder.add_body("base", mass=0.5)
    builder.add_body("link", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent="base",
        child="link",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    return build_model(builder.finalize())


def test_standalone_node_value_is_fresh_and_does_not_retain_a_memo() -> None:
    x = Variable(torch.tensor([1.0]), name="x")
    node = _ScaleNode(x)

    torch.testing.assert_close(node.value(), torch.tensor([2.0]))
    assert node.calls == 1
    assert not node._has_memo
    assert node._memo is None

    x.tensor = torch.tensor([3.0])
    torch.testing.assert_close(node.value(), torch.tensor([6.0]))
    assert node.calls == 2
    assert not node._has_memo
    assert node._memo is None


def test_explicit_node_memo_is_once_per_evaluation_and_released_afterward() -> None:
    x = Variable(torch.tensor([1.0]), name="x")
    node = _ScaleNode(x)
    problem = Problem([_NodeResidual(node, name=f"consumer_{index}") for index in range(3)])

    torch.testing.assert_close(problem.error(), torch.tensor([2.0, 2.0, 2.0]))
    assert node.calls == 1
    assert not node._has_memo
    assert node._memo is None

    problem.error()
    assert node.calls == 2


def test_nested_problem_evaluation_restores_the_outer_memo_scope() -> None:
    x = Variable(torch.tensor([1.0]), name="x")
    node = _ScaleNode(x)
    problem = Problem([_NodeResidual(node, name="consumer")])
    problem._freeze()

    with problem._evaluation():
        torch.testing.assert_close(node.value(), torch.tensor([2.0]))
        torch.testing.assert_close(node.value(), torch.tensor([2.0]))
        assert node.calls == 1

        with problem._evaluation({"x": torch.tensor([3.0])}):
            torch.testing.assert_close(node.value(), torch.tensor([6.0]))
            torch.testing.assert_close(node.value(), torch.tensor([6.0]))
            assert node.calls == 2

        torch.testing.assert_close(node.value(), torch.tensor([2.0]))
        torch.testing.assert_close(node.value(), torch.tensor([2.0]))
        assert node.calls == 3

    assert not node._evaluation_depth
    assert not node._has_memo
    assert node._memo is None


def test_inactive_residual_does_not_evaluate_its_node() -> None:
    x = Variable(torch.tensor([1.0]), name="x")
    node = _ScaleNode(x)
    problem = Problem([_NodeResidual(node, name="inactive", weight=0.0)])

    torch.testing.assert_close(problem.error(), torch.zeros(1))
    assert node.calls == 0


def test_node_variables_drive_transitive_jacobian_structure() -> None:
    x = Variable(torch.tensor([1.0]), name="x")
    node = _ScaleNode(x, multiplier=3.0)
    problem = Problem([_NodeResidual(node, name="consumer")])

    blocks = problem.jacobian_blocks(strategy="jacrev")

    assert set(blocks) == {("consumer", "x")}
    torch.testing.assert_close(blocks[("consumer", "x")], torch.tensor([[3.0]]))
    assert not node._has_memo


def test_non_robot_nodes_share_only_by_explicit_identity() -> None:
    x = Variable(torch.tensor([1.0]), name="x")
    first = _ScaleNode(x)
    second = _ScaleNode(x)
    problem = Problem(
        [
            _NodeResidual(first, name="first"),
            _NodeResidual(second, name="second"),
        ]
    )

    problem.error()

    assert first.calls == second.calls == 1


def test_robot_state_nodes_merge_by_variable_and_model_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _one_joint_model()
    q = RobotVariable(model, name="q")
    calls: list[torch.Tensor] = []

    def fake_fk(model_arg, value: torch.Tensor, *, compute_frames: bool):
        assert model_arg is model
        assert compute_frames
        calls.append(value)
        return SimpleNamespace(q=value)

    monkeypatch.setattr("better_robot.residuals.nodes.forward_kinematics", fake_fk)

    class _RobotResidual(Residual):
        def __init__(self, state: RobotState, name: str) -> None:
            self.state = state
            self.nodes = (state,)
            super().__init__(dim=1, name=name)

        def error(self) -> torch.Tensor:
            return self.state.value().q[..., :1]

    problem = Problem(
        [
            _RobotResidual(RobotState(q), "first"),
            _RobotResidual(RobotState(q), "second"),
            _RobotResidual(RobotState(q), "third"),
        ]
    )

    problem.error()
    assert len(calls) == 1
    problem.error()
    assert len(calls) == 2
