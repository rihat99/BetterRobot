"""Every fixture robot's neutral configuration lies inside its limits."""

from __future__ import annotations

import pytest

import better_robot as br
from better_robot.model.joints import JointFreeFlyer

from tests.support.branching_tree import make_branching_tree_model
from tests.support.robots import g1_urdf, panda_urdf

ROBOTS = {
    "panda": lambda: br.load(panda_urdf()),
    "g1": lambda: br.load(g1_urdf(), root_joint=JointFreeFlyer()),
    "branching_tree": make_branching_tree_model,
}


@pytest.mark.parametrize("name", list(ROBOTS))
def test_q_neutral_lies_inside_the_limits(name):
    model = ROBOTS[name]()
    assert model.q_neutral.shape == (model.nq,)
    assert bool((model.q_neutral >= model.lower_position_limit).all()), name
    assert bool((model.q_neutral <= model.upper_position_limit).all()), name


def test_panda_joint4_neutral_sits_at_the_upper_limit():
    """Panda's joint 4 excludes zero; the clamp puts the neutral entry at its upper limit."""
    model = br.load(panda_urdf())
    index = model.idx_qs[model.joint_id("panda_joint4")]
    assert model.upper_position_limit[index] < 0.0
    assert model.q_neutral[index] == model.upper_position_limit[index]
