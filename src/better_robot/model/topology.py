"""Topology helpers: topological sort, parent/child/subtree, support chains.

Pure tree utilities operating on the ``parent_joint_ids`` tuple of a ``Model``.
Joint 0 is always the universe root (``parent_joint_ids[0] == -1``).
"""

from __future__ import annotations


def topo_sort(parent_joint_ids: tuple[int, ...]) -> tuple[int, ...]:
    """Return a topological order where every parent precedes its children.

    Joint 0 (universe, ``parent_joint_ids[0] == -1``) comes first.
    Uses depth-first traversal for deterministic ordering.
    """
    children = build_children(parent_joint_ids)
    order: list[int] = []
    stack: list[int] = [0]
    while stack:
        j = stack.pop()
        order.append(j)
        # Push children reversed so they pop in ascending (sorted) order
        for child in reversed(children[j]):
            stack.append(child)
    return tuple(order)


def build_children(parent_joint_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return ``children[i]`` = sorted tuple of joint ids whose parent is ``i``."""
    n = len(parent_joint_ids)
    ch: list[list[int]] = [[] for _ in range(n)]
    for j, p in enumerate(parent_joint_ids):
        if p >= 0:
            ch[p].append(j)
    return tuple(tuple(sorted(c)) for c in ch)


def build_subtrees(parent_joint_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return ``subtrees[i]`` = tuple of joint ids in the subtree rooted at ``i``
    (including ``i`` itself), in topological order.
    """
    children = build_children(parent_joint_ids)
    n = len(parent_joint_ids)
    result: list[tuple[int, ...]] = [() for _ in range(n)]
    # Process in reverse topological order so children are ready before parent_joint_ids
    for j in reversed(topo_sort(parent_joint_ids)):
        subtree = [j]
        for child in children[j]:
            subtree.extend(result[child])
        result[j] = tuple(subtree)
    return tuple(result)


def build_supports(parent_joint_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return ``supports[i]`` = chain from joint 0 to joint ``i`` (inclusive)."""
    n = len(parent_joint_ids)
    result: list[tuple[int, ...]] = [() for _ in range(n)]
    for j in topo_sort(parent_joint_ids):
        p = parent_joint_ids[j]
        if p < 0:
            result[j] = (j,)
        else:
            result[j] = result[p] + (j,)
    return tuple(result)
