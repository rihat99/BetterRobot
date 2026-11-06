"""Joint ids accept ``int | str`` everywhere a frame id does; unknown names raise ``UnknownNameError``."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import UnknownNameError
from better_robot.io.build_model import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.kinematics import (
    forward_kinematics,
    subtree,
    support,
    joint_jacobian,
    joint_jacobian_time_variation,
)


def _two_link():
    builder = ModelBuilder("two_link")
    base = builder.add_body("base")
    link1 = builder.add_body("link1")
    link2 = builder.add_body("link2")
    builder.add_revolute("j1", axis=(0.0, 0.0, 1.0), parent=base, child=link1)
    builder.add_revolute("j2", axis=(0.0, 0.0, 1.0), parent=link1, child=link2)
    return build_model(builder.finalize(), dtype=torch.float64)


def test_joint_names_match_ids() -> None:
    model = _two_link()
    q = torch.tensor([[0.3, -0.7]], dtype=torch.float64)
    v = torch.tensor([[0.2, 0.5]], dtype=torch.float64)
    state = forward_kinematics(model, q)
    joint = model.joint_id("j2")
    torch.testing.assert_close(joint_jacobian(model, state, joint="j2"), joint_jacobian(model, state, joint=joint))
    torch.testing.assert_close(
        joint_jacobian_time_variation(model, state, v=v, joint="j2"),
        joint_jacobian_time_variation(model, state, v=v, joint=joint),
    )
    assert subtree(model, joint="j1") == subtree(model, joint=model.joint_id("j1"))
    assert support(model, joint="j2") == support(model, joint=joint)


@pytest.mark.parametrize("resolver", ["joint_id", "frame_id", "body_id"])
def test_unknown_name_raises_typed_error(resolver: str) -> None:
    model = _two_link()
    with pytest.raises(UnknownNameError):
        getattr(model, resolver)("no_such_name")
