"""LinkInfo dataclass: link metadata for the kinematic tree."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LinkInfo:
    """Stores link metadata."""

    names: tuple[str, ...]
    """Ordered link names (BFS order from base)."""

    num_links: int
    """Total number of links."""

    parent_joint_indices: tuple[int, ...]
    """Index of the parent joint for each link (-1 for base link)."""
