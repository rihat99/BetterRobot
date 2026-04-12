"""Viser helper utilities — quaternion conversions, coordinate remapping, etc.

See ``docs/01_ARCHITECTURE.md``.
"""

from __future__ import annotations

import torch


def quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert scalar-last ``[qx, qy, qz, qw]`` to scalar-first ``[qw, qx, qy, qz]``.

    Viser uses ``wxyz``; BetterRobot uses ``xyzw``.
    """
    return torch.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], dim=-1)


def quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Inverse of ``quat_xyzw_to_wxyz``."""
    return torch.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], dim=-1)
