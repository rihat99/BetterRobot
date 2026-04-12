"""Shared spatial-algebra operators — ``ad``, ``Ad``, ``cross``, ``act``.

These bind the value types in ``spatial/`` to the functional facade in
``lie/`` so callers above the spatial layer never have to know which lives
where.

See ``docs/03_LIE_AND_SPATIAL.md §8``.
"""

from __future__ import annotations

import torch

from .force import Force
from .inertia import Inertia
from .motion import Motion


def ad(v: Motion) -> torch.Tensor:
    """``ad(v)`` — 6×6 motion cross operator. ``(..., 6, 6)``."""
    from ..lie.tangents import hat_so3
    vv = v.linear    # (..., 3)
    wv = v.angular   # (..., 3)
    Vhat = hat_so3(vv)    # (..., 3, 3)
    What = hat_so3(wv)    # (..., 3, 3)
    *batch, _, _ = What.shape
    zeros33 = torch.zeros(*batch, 3, 3, dtype=v.data.dtype, device=v.data.device)
    top    = torch.cat([What, Vhat   ], dim=-1)
    bottom = torch.cat([zeros33, What], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def ad_star(f: Force) -> torch.Tensor:
    """``ad*(f)`` — 6×6 force cross operator (dual). ``(..., 6, 6)``."""
    return -ad(Motion(f.data)).transpose(-1, -2)


def cross_mm(a: Motion, b: Motion) -> Motion:
    """Same as ``a.cross_motion(b)``."""
    return a.cross_motion(b)


def cross_mf(a: Motion, b: Force) -> Force:
    """Same as ``a.cross_force(b)``."""
    return a.cross_force(b)


def act_motion(T: torch.Tensor, m: Motion) -> Motion:
    """Apply an SE3 pose to a motion via the adjoint: Ad(T) * m."""
    return m.se3_action(T)


def act_force(T: torch.Tensor, f: Force) -> Force:
    """Apply an SE3 pose to a force via the dual adjoint."""
    return f.se3_action(T)


def act_inertia(T: torch.Tensor, I: Inertia) -> Inertia:  # noqa: E741
    """Apply an SE3 pose to a spatial inertia: I_new = Ad^{-T}(T) I Ad^{-1}(T)."""
    return I.se3_action(T)
