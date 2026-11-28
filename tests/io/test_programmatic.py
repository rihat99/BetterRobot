"""``ModelBuilder``: the IR it emits and the errors it raises."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import ModelError, UnknownNameError
from better_robot.io import build_model
from better_robot.io.ir import IRModel, IRShape
from better_robot.io.builder import ModelBuilder
from better_robot.lie import se3
from better_robot.model.geometry import Sphere
from better_robot.model.joints import JointRevolute


def _simple_arm() -> IRModel:
    """Two-body, one-revolute-joint arm."""
    b = ModelBuilder("test_arm")
    b.add_body("base", mass=0.0)
    b.add_body("link1", mass=1.0)
    b.add_revolute(
        "j1",
        axis=(1.0, 0.0, 0.0),
        parent="base",
        child="link1",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    return b.finalize()


def test_finalize_emits_the_bodies_and_joints():
    ir = _simple_arm()
    assert len(ir.bodies) == 2
    assert len(ir.joints) == 1
    j = ir.joints[0]
    assert j.parent_body == "base"
    assert j.child_body == "link1"
    assert j.joint == JointRevolute(axis=(1.0, 0.0, 0.0))


def _add_base_link1_and_j1(b):
    b.add_body("base")
    b.add_body("link1")
    b.add_revolute("j1", axis=(0.0, 0.0, 1.0), parent="base", child="link1", origin=se3.identity())


@pytest.mark.parametrize(
    ("setup", "action", "match"),
    [
        pytest.param(lambda b: b.add_body("base"), lambda b: b.add_body("base"), "already exists", id="duplicate_body"),
        pytest.param(
            _add_base_link1_and_j1,
            lambda b: b.add_revolute("j1", axis=(0.0, 0.0, 1.0), parent="base", child="link1", origin=se3.identity()),
            "already exists",
            id="duplicate_joint",
        ),
        # No joints → 2 root candidates
        pytest.param(
            lambda b: (b.add_body("a"), b.add_body("b")),
            lambda b: build_model(b.finalize()),
            "root body",
            id="multiple_root_bodies",
        ),
    ],
)
def test_builder_invalid_structure_raises(setup, action, match):
    b = ModelBuilder("x")
    setup(b)
    with pytest.raises(ModelError, match=match):
        action(b)


def test_add_shape():
    b = ModelBuilder("x")
    b.add_body("base")
    b.add_body("link1")
    b.add_fixed("j1", parent="base", child="link1", origin=se3.identity())
    sphere = Sphere(radius=0.05, placement=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    b.add_shape("link1", sphere)
    b.add_shape("base", sphere, collision=False)
    assert b.finalize().shapes == [
        IRShape(body="link1", shape=sphere, collision=True),
        IRShape(body="base", shape=sphere, collision=False),
    ]
    with pytest.raises(UnknownNameError, match="'link2'"):
        b.add_shape("link2", sphere)
