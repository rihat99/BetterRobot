"""Equality and hashing (decision record 0010): a ``Model`` compares by identity; ``structure_key`` compares the
static fields, so two builds of one description share a key and a model with scaled links keeps it. Three
topologies: the Panda (fixed base), the G1 (free flyer) and the branching tree (free flyer plus spherical joints)."""

from __future__ import annotations

import pytest
import torch

from better_robot import load
from better_robot.model import JointFreeFlyer
from tests.support.branching_tree import make_branching_tree_model
from tests.support.robots import g1_urdf, panda_urdf

TOPOLOGIES = [
    pytest.param(lambda: load(panda_urdf()), id="panda"),
    pytest.param(lambda: load(g1_urdf(), root_joint=JointFreeFlyer()), id="g1"),
    pytest.param(make_branching_tree_model, id="branching_tree"),
]


@pytest.mark.parametrize("build", TOPOLOGIES)
def test_two_builds_are_distinct_models_with_one_structure_key(build) -> None:
    first, second = build(), build()
    assert first != second and hash(first) != hash(second)
    assert first.structure_key == second.structure_key
    assert hash(first.structure_key) == hash(second.structure_key)


def test_different_topologies_have_different_structure_keys() -> None:
    keys = [build.values[0]().structure_key for build in TOPOLOGIES]
    assert len(set(keys)) == len(keys)


def test_with_values_keeps_the_structure_key() -> None:
    model = load(panda_urdf())
    scaled = model.joint_placements.clone()
    scaled[..., :3] *= 1.1
    longer = model.with_values(joint_placements=scaled)
    assert longer != model and longer.structure_key == model.structure_key
    assert not torch.equal(longer.joint_placements, model.joint_placements)
