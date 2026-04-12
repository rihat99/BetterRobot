"""Low-level closest-point kernels — segment-segment, point-segment, etc.

These are the building blocks that ``pairs.py`` composes into signed-distance
dispatches.

See ``docs/09_COLLISION_GEOMETRY.md §4``.
"""

from __future__ import annotations

import torch


def point_to_segment(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Closest point on segment ``[a, b]`` to point ``p``. Returns the distance.

    Vectorised over leading dims. See docs/09_COLLISION_GEOMETRY.md §4.
    """
    raise NotImplementedError("see docs/09_COLLISION_GEOMETRY.md §4")


def segment_to_segment(
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Minimum distance between segments ``[a1,b1]`` and ``[a2,b2]``.

    Vectorised over leading dims. See docs/09_COLLISION_GEOMETRY.md §4.
    """
    raise NotImplementedError("see docs/09_COLLISION_GEOMETRY.md §4")
