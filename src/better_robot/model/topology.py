"""Topology helpers: the subtree and the support chain of every joint.

Pure tree utilities operating on the ``parent_joint_ids`` tuple of a ``Model``. Joint 0 is always the universe
root (``parent_joint_ids[0] == -1``) and every parent precedes its child (``parent_joint_ids[i] < i``), so the
joint order itself is topological.
"""

from __future__ import annotations


def build_subtrees(parent_joint_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return ``subtrees[i]`` = tuple of joint ids in the subtree rooted at ``i``
    (including ``i`` itself), in topological order.
    """
    n = len(parent_joint_ids)
    children: list[list[int]] = [[] for _ in range(n)]
    for j, p in enumerate(parent_joint_ids):
        if p >= 0:
            children[p].append(j)
    result: list[tuple[int, ...]] = [() for _ in range(n)]
    # Children have larger ids, so in reverse joint order every child's subtree is ready before its parent's.
    for j in reversed(range(n)):
        subtree = [j]
        for child in children[j]:
            subtree.extend(result[child])
        result[j] = tuple(subtree)
    return tuple(result)


def build_supports(parent_joint_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return ``supports[i]`` = chain from joint 0 to joint ``i`` (inclusive)."""
    result: list[tuple[int, ...]] = []
    for j, p in enumerate(parent_joint_ids):
        result.append((j,) if p < 0 else result[p] + (j,))
    return tuple(result)
