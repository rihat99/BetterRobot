"""Tests for data_model/topology.py."""
import pytest
from better_robot.data_model.topology import (
    topo_sort, build_children, build_subtrees, build_supports,
    get_subtree, get_support,
)

# Simple chain: 0 → 1 → 2 → 3
CHAIN = (-1, 0, 1, 2)

# Simple tree: 0 → {1, 2}, 1 → {3, 4}
TREE = (-1, 0, 0, 1, 1)


def test_topo_sort_chain():
    order = topo_sort(CHAIN)
    assert order == (0, 1, 2, 3)


def test_topo_sort_tree():
    order = topo_sort(TREE)
    # 0 must come before 1, 2; 1 must come before 3, 4
    assert order[0] == 0
    assert order.index(1) < order.index(3)
    assert order.index(1) < order.index(4)
    assert order.index(0) < order.index(2)


def test_build_children_chain():
    ch = build_children(CHAIN)
    assert ch[0] == (1,)
    assert ch[1] == (2,)
    assert ch[2] == (3,)
    assert ch[3] == ()


def test_build_children_tree():
    ch = build_children(TREE)
    assert set(ch[0]) == {1, 2}
    assert set(ch[1]) == {3, 4}
    assert ch[2] == ()


def test_build_subtrees_chain():
    st = build_subtrees(CHAIN)
    assert set(st[0]) == {0, 1, 2, 3}
    assert set(st[1]) == {1, 2, 3}
    assert set(st[2]) == {2, 3}
    assert st[3] == (3,)


def test_build_subtrees_tree():
    st = build_subtrees(TREE)
    assert set(st[0]) == {0, 1, 2, 3, 4}
    assert set(st[1]) == {1, 3, 4}
    assert set(st[2]) == {2}


def test_build_supports_chain():
    sp = build_supports(CHAIN)
    assert sp[0] == (0,)
    assert sp[1] == (0, 1)
    assert sp[3] == (0, 1, 2, 3)


def test_build_supports_tree():
    sp = build_supports(TREE)
    assert sp[3] == (0, 1, 3)
    assert sp[4] == (0, 1, 4)
    assert sp[2] == (0, 2)


def test_get_subtree():
    assert set(get_subtree(CHAIN, 2)) == {2, 3}


def test_get_support():
    assert get_support(CHAIN, 3) == (0, 1, 2, 3)
