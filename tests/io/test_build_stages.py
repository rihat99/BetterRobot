"""``build_model`` runs named stages, and every fixture robot's neutral configuration lies inside its limits."""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.io import build_model
from better_robot.io.build_model import _frames, _joint_order, _layout, _limits, _mimic, _neutral, _root, _topology
from better_robot.model.joints import JointFreeFlyer

from tests.support.branching_tree import JOINT_NAMES, make_branching_tree_body, make_branching_tree_model
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


def test_each_stage_returns_its_named_result_and_the_model_is_their_sum():
    ir = make_branching_tree_body()
    dtype = torch.float32
    root = _root(ir, None)
    assert (root.body, root.name) == ("j0", "root")
    assert isinstance(root.joint, JointFreeFlyer) and root.ir_joint is not None
    assert len(root.regular) == len(ir.joints) - 1

    order = _joint_order(ir, root.body, root.regular)
    assert order == root.regular  # the fixture declares its joints parents first

    topology = _topology(ir, root, order, dtype)
    assert topology.joint_names == ("universe", "root", *JOINT_NAMES[1:])
    assert topology.body_names == ("universe", *JOINT_NAMES)
    assert topology.joint_placements.shape == (len(topology.joints), 7)

    full = _layout(tuple(joint.nq for joint in topology.joints), tuple(joint.nv for joint in topology.joints))
    assert (full.nq, full.nv) == (7 + 4 * 23, 6 + 3 * 23)
    assert full.idx_qs[-1] + full.nqs[-1] == full.nq

    limits = _limits(topology, dtype)
    assert limits.lower.shape == (full.nq,) and limits.velocity.shape == (full.nv,)

    reduction = _mimic(topology, full, limits)
    assert reduction.layout == full  # no mimic joints: the public layout is the full one
    assert reduction.source_ids == tuple(range(len(topology.joints)))

    frames = _frames(ir, topology, dtype)
    assert frames.names == topology.body_names

    neutral = _neutral(topology, reduction.layout, reduction.limits, dtype)
    model = build_model(ir)
    assert model.joint_names == topology.joint_names
    assert model.frame_names == frames.names
    assert model.parent_joint_ids == topology.parent_joint_ids
    torch.testing.assert_close(model.q_neutral, neutral)
    torch.testing.assert_close(model.lower_position_limit, reduction.limits.lower)
