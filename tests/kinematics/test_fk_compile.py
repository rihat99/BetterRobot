"""A plain ``torch.compile`` of the public forward kinematics traces as one graph from a cold plan; the opt-in
quaternion check reads the full configuration."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import QuaternionNormError
from better_robot.io import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.kinematics import forward_kinematics


def _free_flyer():
    builder = ModelBuilder("free_flyer_compile")
    builder.add_body("base", mass=1.0)
    builder.add_free_flyer_root(
        "floating",
        child="base",
        origin=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    return build_model(builder.finalize())


def test_free_flyer_fk_compiles_fullgraph_on_a_cold_plan():
    """A fresh model's first call is the compiled one, so the FK plan is built inside the trace: no plan step may read
    a tensor value."""
    model = _free_flyer()
    torch._dynamo.reset()
    q = model.q_neutral.unsqueeze(0).expand(2, -1).clone()
    actual = torch.compile(lambda q: forward_kinematics(model, q).joint_transforms, fullgraph=True)(q)
    torch.testing.assert_close(actual, forward_kinematics(model, q).joint_transforms)


def test_quaternion_norm_check_reads_the_full_configuration():
    """A mimic joint before a spherical joint shifts the quaternion's public columns by one."""
    builder = ModelBuilder("mimic_then_ball")
    for name in ("base", "first", "second", "ball_link", "tip"):
        builder.add_body(name, mass=1.0)
    z, y = (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)
    builder.add_revolute("source", parent="base", child="first", axis=z, lower=-2.0, upper=2.0)
    builder.add_revolute("target", parent="first", child="second", axis=z, lower=-2.0, upper=2.0, mimic_source="source")
    builder.add_spherical("ball", parent="second", child="ball_link")
    builder.add_revolute("wrist", parent="ball_link", child="tip", axis=y, lower=-2.0, upper=2.0)
    model = build_model(builder.finalize(), dtype=torch.float64)
    assert (model.nq, model.nq_full) == (6, 7)  # q = [source, qx, qy, qz, qw, wrist]

    valid = torch.tensor([0.3, 0.0, 0.0, 0.0, 1.0, 0.7], dtype=torch.float64)
    forward_kinematics(model, valid, check_quaternion_norm=True)
    off_norm = torch.tensor([0.3, 0.7, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)  # quaternion norm 1.22
    with pytest.raises(QuaternionNormError, match="quaternion norm"):
        forward_kinematics(model, off_norm, check_quaternion_norm=True)
