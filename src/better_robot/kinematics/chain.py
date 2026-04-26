"""Subtree / chain helpers — thin wrappers over ``data_model.topology``.

See ``docs/concepts/kinematics.md``.
"""

from __future__ import annotations

from ..data_model.model import Model


def get_chain(model: Model, start: int, end: int) -> tuple[int, ...]:
    """Return the joint chain from ``start`` to ``end`` through their common ancestor.

    See docs/concepts/kinematics.md.
    """
    raise NotImplementedError("see docs/concepts/kinematics.md")
