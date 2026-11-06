"""Tests for model/topology.py."""

import pytest

from better_robot.model.topology import (
    topo_sort,
    build_children,
    build_subtrees,
    build_supports,
)

# Simple chain: 0 → 1 → 2 → 3
CHAIN = (-1, 0, 1, 2)

# Simple tree: 0 → {1, 2}, 1 → {3, 4}
TREE = (-1, 0, 0, 1, 1)

_SHAPES = [pytest.param(CHAIN, id="chain"), pytest.param(TREE, id="tree")]


@pytest.mark.parametrize("parent_joint_ids", _SHAPES)
def test_topo_sort(parent_joint_ids):
    order = topo_sort(parent_joint_ids)
    # A permutation with every parent before its child: for CHAIN this forces (0, 1, 2, 3);
    # for TREE it gives 0 first, 0 before 2, 1 before 3 and 4.
    assert sorted(order) == list(range(len(parent_joint_ids)))
    for child, parent in enumerate(parent_joint_ids):
        if parent >= 0:
            assert order.index(parent) < order.index(child)


@pytest.mark.parametrize(
    "parent_joint_ids, expected",
    [
        pytest.param(CHAIN, {0: (1,), 1: (2,), 2: (3,), 3: ()}, id="chain"),
        pytest.param(TREE, {0: (1, 2), 1: (3, 4), 2: ()}, id="tree"),
    ],
)
def test_build_children(parent_joint_ids, expected):
    ch = build_children(parent_joint_ids)
    for node, children in expected.items():
        assert isinstance(ch[node], tuple)
        assert sorted(ch[node]) == sorted(children)


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
