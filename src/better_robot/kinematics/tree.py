"""Tree queries on a model: the subtree below a joint and the support chain above it."""

from __future__ import annotations

from ..model.model import Model
from ..model.topology import build_subtrees, build_supports


def subtree(model: Model, *, joint: int | str) -> tuple[int, ...]:
    """Ids of ``joint`` and every joint below it, parents before children."""
    return build_subtrees(model.parent_joint_ids)[model.joint_id(joint)]


def support(model: Model, *, joint: int | str) -> tuple[int, ...]:
    """Ids of the joints from the universe down to ``joint``, inclusive."""
    return build_supports(model.parent_joint_ids)[model.joint_id(joint)]


__all__ = ["subtree", "support"]
