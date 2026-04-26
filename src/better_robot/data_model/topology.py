"""Topology helpers: topological sort, parent/child/subtree, support chains.

Pure tree utilities operating on the ``parents`` tuple of a ``Model``.
Joint 0 is always the universe root (``parents[0] == -1``).

See ``docs/concepts/model_and_data.md §2``.
"""

from __future__ import annotations


def topo_sort(parents: tuple[int, ...]) -> tuple[int, ...]:
    """Return a topological order where every parent precedes its children.

    Joint 0 (universe, ``parents[0] == -1``) comes first.
    Uses depth-first traversal for deterministic ordering.
    """
    children = build_children(parents)
    order: list[int] = []
    stack: list[int] = [0]
    while stack:
        j = stack.pop()
        order.append(j)
        # Push children reversed so they pop in ascending (sorted) order
        for child in reversed(children[j]):
            stack.append(child)
    return tuple(order)


def build_children(parents: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return ``children[i]`` = sorted tuple of joint ids whose parent is ``i``."""
    n = len(parents)
    ch: list[list[int]] = [[] for _ in range(n)]
    for j, p in enumerate(parents):
        if p >= 0:
            ch[p].append(j)
    return tuple(tuple(sorted(c)) for c in ch)


def build_subtrees(parents: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return ``subtrees[i]`` = tuple of joint ids in the subtree rooted at ``i``
    (including ``i`` itself), in topological order.
    """
    children = build_children(parents)
    n = len(parents)
    result: list[tuple[int, ...]] = [() for _ in range(n)]
    # Process in reverse topological order so children are ready before parents
    for j in reversed(topo_sort(parents)):
        subtree = [j]
        for child in children[j]:
            subtree.extend(result[child])
        result[j] = tuple(subtree)
    return tuple(result)


def build_supports(parents: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return ``supports[i]`` = chain from joint 0 to joint ``i`` (inclusive)."""
    n = len(parents)
    result: list[tuple[int, ...]] = [() for _ in range(n)]
    for j in topo_sort(parents):
        p = parents[j]
        if p < 0:
            result[j] = (j,)
        else:
            result[j] = result[p] + (j,)
    return tuple(result)


def get_subtree(parents: tuple[int, ...], joint_id: int) -> tuple[int, ...]:
    """Return the subtree rooted at ``joint_id`` (including itself)."""
    return build_subtrees(parents)[joint_id]


def get_support(parents: tuple[int, ...], joint_id: int) -> tuple[int, ...]:
    """Return the joint chain from joint 0 to ``joint_id`` (inclusive)."""
    return build_supports(parents)[joint_id]
