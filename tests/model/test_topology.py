"""Tests for src/better_robot/model/topology.py."""

import pytest

from better_robot.model.topology import build_subtrees, build_supports

# Simple chain: 0 → 1 → 2 → 3
CHAIN = (-1, 0, 1, 2)

# Simple tree: 0 → {1, 2}, 1 → {3, 4}
TREE = (-1, 0, 0, 1, 1)


@pytest.mark.parametrize(
    "parent_joint_ids, expected",
    [
        pytest.param(CHAIN, {0: (0, 1, 2, 3), 1: (1, 2, 3), 2: (2, 3), 3: (3,)}, id="chain"),
        pytest.param(TREE, {0: (0, 1, 2, 3, 4), 1: (1, 3, 4), 2: (2,)}, id="tree"),
    ],
)
def test_build_subtrees(parent_joint_ids, expected):
    st = build_subtrees(parent_joint_ids)
    for node, subtree in expected.items():
        assert isinstance(st[node], tuple)
        assert sorted(st[node]) == sorted(subtree)


@pytest.mark.parametrize(
    "parent_joint_ids, expected",
    [
        pytest.param(CHAIN, {0: (0,), 1: (0, 1), 3: (0, 1, 2, 3)}, id="chain"),
        pytest.param(TREE, {3: (0, 1, 3), 4: (0, 1, 4), 2: (0, 2)}, id="tree"),
    ],
)
def test_build_supports(parent_joint_ids, expected):
    sp = build_supports(parent_joint_ids)
    for node, support in expected.items():
        assert sp[node] == support
