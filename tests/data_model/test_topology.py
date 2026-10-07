"""Tests for data_model/topology.py."""
import pytest

from better_robot.data_model.topology import (
    topo_sort, build_children, build_subtrees, build_supports,
)

# Simple chain: 0 → 1 → 2 → 3
CHAIN = (-1, 0, 1, 2)

# Simple tree: 0 → {1, 2}, 1 → {3, 4}
TREE = (-1, 0, 0, 1, 1)

_SHAPES = [pytest.param(CHAIN, id="chain"), pytest.param(TREE, id="tree")]


@pytest.mark.parametrize("parents", _SHAPES)
def test_topo_sort(parents):
    order = topo_sort(parents)
    # A permutation with every parent before its child: for CHAIN this forces (0, 1, 2, 3);
    # for TREE it gives 0 first, 0 before 2, 1 before 3 and 4.
    assert sorted(order) == list(range(len(parents)))
    for child, parent in enumerate(parents):
        if parent >= 0:
            assert order.index(parent) < order.index(child)


@pytest.mark.parametrize(
    "parents, expected",
    [
        pytest.param(CHAIN, {0: (1,), 1: (2,), 2: (3,), 3: ()}, id="chain"),
        pytest.param(TREE, {0: (1, 2), 1: (3, 4), 2: ()}, id="tree"),
    ],
)
def test_build_children(parents, expected):
    ch = build_children(parents)
    for node, children in expected.items():
        assert isinstance(ch[node], tuple)
        assert sorted(ch[node]) == sorted(children)


@pytest.mark.parametrize(
    "parents, expected",
    [
        pytest.param(CHAIN, {0: (0, 1, 2, 3), 1: (1, 2, 3), 2: (2, 3), 3: (3,)}, id="chain"),
        pytest.param(TREE, {0: (0, 1, 2, 3, 4), 1: (1, 3, 4), 2: (2,)}, id="tree"),
    ],
)
def test_build_subtrees(parents, expected):
    st = build_subtrees(parents)
    for node, subtree in expected.items():
        assert isinstance(st[node], tuple)
        assert sorted(st[node]) == sorted(subtree)


@pytest.mark.parametrize(
    "parents, expected",
    [
        pytest.param(CHAIN, {0: (0,), 1: (0, 1), 3: (0, 1, 2, 3)}, id="chain"),
        pytest.param(TREE, {3: (0, 1, 3), 4: (0, 1, 4), 2: (0, 2)}, id="tree"),
    ],
)
def test_build_supports(parents, expected):
    sp = build_supports(parents)
    for node, support in expected.items():
        assert sp[node] == support
