"""A body's frame carries the body's name; the IR frames follow with their own names, and a clash is an error."""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.exceptions import ModelError
from better_robot.io import ModelBuilder, build_model
from better_robot.io.ir import IRFrame
from better_robot.lie import se3

from tests.support.robots import g1_urdf, panda_urdf


@pytest.mark.parametrize("path", [panda_urdf(), g1_urdf()], ids=["panda", "g1"])
def test_every_body_has_a_frame_of_its_name_at_its_joint(path):
    model = br.load(path)
    nbodies = model.nbodies
    assert model.frame_names[:nbodies] == model.body_names
    assert model.frame_parent_joint_ids[:nbodies] == tuple(range(nbodies))
    assert not any(name.startswith("body_") for name in model.frame_names)
    torch.testing.assert_close(model.frame_placements[:nbodies], se3.identity(batch_shape=(nbodies,)))


def _two_links() -> ModelBuilder:
    builder = ModelBuilder("two_links")
    builder.add_body("base")
    builder.add_body("link")
    builder.add_revolute("hinge", axis=(0.0, 0.0, 1.0), parent="base", child="link")
    return builder


def test_ir_frames_keep_their_names_after_the_body_frames():
    builder = _two_links()
    builder.add_frame("tool", parent="link", placement=se3.from_translation(torch.tensor([0.0, 0.0, 0.1])))
    model = build_model(builder.finalize())
    assert model.frame_names == ("universe", "base", "link", "tool")
    assert model.frame_id("tool") == 3
    assert model.frame_parent_joint_ids[3] == model.joint_id("hinge")


def test_a_frame_named_like_a_body_is_a_model_error():
    builder = _two_links()
    builder.add_frame("link", parent="base", placement=se3.identity())
    with pytest.raises(ModelError, match="'link' is already the name of a body"):
        build_model(builder.finalize())


def test_two_frames_of_one_name_are_a_model_error():
    builder = _two_links()
    builder.add_frame("tool", parent="link", placement=se3.identity())
    builder.add_frame("tool", parent="base", placement=se3.identity())
    with pytest.raises(ModelError, match="'tool' is already the name"):
        build_model(builder.finalize())


def test_a_frame_on_an_unknown_body_is_a_model_error():
    ir = _two_links().finalize()
    ir.frames.append(IRFrame(name="tool", parent_body="no_such_body", placement=se3.identity()))
    with pytest.raises(ModelError, match="unknown body 'no_such_body'"):
        build_model(ir)
