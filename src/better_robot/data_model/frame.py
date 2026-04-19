"""``Frame`` — named point rigidly attached to a parent joint.

Pinocchio-style frame indirection: operational frames (tool tips, IMUs, IK
targets) live on a joint with a fixed local placement. Frame *placements*
(the world pose of each frame) live on ``Data.frame_pose_world``; this
struct is metadata only.

See ``docs/02_DATA_MODEL.md §6``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

FrameType = Literal["body", "joint", "fixed", "op", "sensor"]


@dataclass(frozen=True)
class Frame:
    """Named point on a parent joint with a fixed local SE3 placement."""

    name: str
    parent_joint: int
    joint_placement: torch.Tensor  # (7,) SE3 in parent joint's frame
    frame_type: FrameType = "op"
