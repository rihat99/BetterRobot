"""Conversion utilities for quaternion representations and pose formats."""
from __future__ import annotations
import torch


def qxyzw_to_wxyz(q: torch.Tensor) -> tuple:
    """Convert [qx, qy, qz, qw] tensor to viser (w, x, y, z) tuple."""
    return (q[3].item(), q[0].item(), q[1].item(), q[2].item())


def wxyz_to_qxyzw(wxyz: tuple) -> torch.Tensor:
    """Convert viser (w, x, y, z) tuple to [qx, qy, qz, qw] tensor."""
    w, x, y, z = wxyz
    return torch.tensor([x, y, z, w], dtype=torch.float32)
