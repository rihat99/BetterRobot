"""Regression tests for evaluation-local object-referenced IK state caching."""

from __future__ import annotations

import torch

import better_robot.residuals.nodes as node_module
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim import Problem, RobotVariable
from better_robot.residuals.pose import PoseResidual


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


def test_kinematic_residuals_share_one_fk_per_evaluation(monkeypatch) -> None:
    model = _make_arm()
    q = torch.tensor([1.0, -0.8, 0.6])
    data = forward_kinematics(model, q, compute_frames=True)

    fk_calls = 0
    original_fk = node_module.forward_kinematics

    def counted_fk(*args, **kwargs):
        nonlocal fk_calls
        fk_calls += 1
        return original_fk(*args, **kwargs)

    monkeypatch.setattr(node_module, "forward_kinematics", counted_fk)
    q_variable = RobotVariable(model, q, name="q")
    residuals = []
    for link in ("body_link2", "body_link3"):
        name = f"pose_{link}"
        residuals.append(
            PoseResidual(
                q_variable,
                frame_id=model.frame_id(link),
                target=data.frame_pose_world[model.frame_id(link)],
                name=name,
            )
        )
    problem = Problem(residuals)

    problem.error()
    assert fk_calls == 1
    problem.jacobian_blocks()
    assert fk_calls == 2
