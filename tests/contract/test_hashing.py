"""Equality and hashing (decision record 0010): a ``Model`` compares by identity; ``structure_key`` compares the
static fields, so two loads of one description share a key and a model with scaled links keeps it."""

from __future__ import annotations

import torch

from better_robot import load
from tests.support.robots import g1_urdf, panda_urdf


def test_two_loads_are_distinct_models_with_one_structure_key() -> None:
    first, second = load(panda_urdf()), load(panda_urdf())
    assert first != second and hash(first) != hash(second)
    assert first.structure_key == second.structure_key
    assert hash(first.structure_key) == hash(second.structure_key)
    assert load(g1_urdf(), free_flyer=True).structure_key != first.structure_key


def test_with_values_keeps_the_structure_key() -> None:
    model = load(panda_urdf())
    scaled = model.joint_placements.clone()
    scaled[..., :3] *= 1.1
    longer = model.with_values(joint_placements=scaled)
    assert longer != model and longer.structure_key == model.structure_key
    assert not torch.equal(longer.joint_placements, model.joint_placements)
