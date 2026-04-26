"""Tests for ``ModelBuilder`` — programmatic IR builder.

See ``docs/concepts/parsers_and_ir.md §6``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.io.ir import IRModel, IRJoint, IRBody


def _identity_se3() -> torch.Tensor:
    return torch.tensor([0., 0., 0., 0., 0., 0., 1.])


def _simple_arm() -> IRModel:
    """Two-body, one-revolute-joint arm."""
    b = ModelBuilder("test_arm")
    b.add_body("base", mass=0.0)
    b.add_body("link1", mass=1.0)
    b.add_revolute_x(
        "j1", parent="base", child="link1",
        origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]),
        lower=-math.pi, upper=math.pi,
    )
    return b.finalize()


def test_finalize_returns_irmodel():
    ir = _simple_arm()
    assert isinstance(ir, IRModel)
    assert ir.name == "test_arm"


def test_root_body_detected():
    ir = _simple_arm()
    assert ir.root_body == "base"


def test_bodies_and_joints_count():
    ir = _simple_arm()
    assert len(ir.bodies) == 2
    assert len(ir.joints) == 1


def test_joint_parent_child():
    ir = _simple_arm()
    j = ir.joints[0]
    assert j.parent_body == "base"
    assert j.child_body == "link1"
    assert j.kind == "revolute"
    # Named helper sets axis to X.
    assert j.axis is not None
    assert torch.allclose(j.axis, torch.tensor([1.0, 0.0, 0.0]))


def test_duplicate_body_raises():
    b = ModelBuilder("x")
    b.add_body("base")
    with pytest.raises(ValueError, match="already exists"):
        b.add_body("base")


def test_duplicate_joint_raises():
    b = ModelBuilder("x")
    b.add_body("base")
    b.add_body("link1")
    b.add_revolute_z("j1", parent="base", child="link1", origin=_identity_se3())
    with pytest.raises(ValueError, match="already exists"):
        b.add_revolute_z("j1", parent="base", child="link1", origin=_identity_se3())


def test_multiple_root_bodies_raises():
    b = ModelBuilder("x")
    b.add_body("a")
    b.add_body("b")
    # No joints → 2 root candidates
    with pytest.raises(ValueError, match="root body"):
        b.finalize()


def test_add_frame():
    b = ModelBuilder("x")
    b.add_body("base")
    b.add_body("link1")
    b.add_fixed("j1", parent="base", child="link1", origin=_identity_se3())
    b.add_frame("tip", parent_body="link1",
                placement=torch.tensor([0., 0., 0.05, 0., 0., 0., 1.]))
    ir = b.finalize()
    assert len(ir.frames) == 1
    assert ir.frames[0].name == "tip"
    assert ir.frames[0].parent_body == "link1"


def test_add_collision_geom():
    b = ModelBuilder("x")
    b.add_body("base")
    b.add_body("link1")
    b.add_fixed("j1", parent="base", child="link1", origin=_identity_se3())
    b.add_collision_geom("link1", "sphere", {"radius": 0.05}, _identity_se3())
    ir = b.finalize()
    assert len(ir.bodies[1].collision_geoms) == 1
    assert ir.bodies[1].collision_geoms[0].kind == "sphere"


def test_three_link_chain():
    b = ModelBuilder("chain")
    b.add_body("root")
    b.add_body("l1")
    b.add_body("l2")
    b.add_revolute_z("j1", parent="root", child="l1",
                     origin=_identity_se3(), lower=-1., upper=1.)
    b.add_prismatic_z("j2", parent="l1", child="l2",
                      origin=_identity_se3(), lower=0., upper=0.5)
    ir = b.finalize()
    assert ir.root_body == "root"
    assert len(ir.joints) == 2
