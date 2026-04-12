"""``Data`` — mutable per-query workspace (Pinocchio-style).

Carries a leading batch (and optional time) axis. Kinematics/dynamics
functions populate the optional fields lazily — this avoids the mjwarp
280-field god-dataclass trap.

See ``docs/02_DATA_MODEL.md §3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Data:
    """Mutable per-query workspace carrying leading batch dims ``B...``.

    Unused cache slots stay ``None``. ``_kinematics_level`` tracks how far
    the kinematic recursion has been run (0=nothing, 1=placements,
    2=velocities, 3=accelerations) so functions that need level-2 can
    assert it has been computed.
    """

    _model_id: int

    # ──────────── configuration & derivatives ────────────
    q: torch.Tensor  # (B..., nq)
    v: Optional[torch.Tensor] = None  # (B..., nv)
    a: Optional[torch.Tensor] = None  # (B..., nv)
    tau: Optional[torch.Tensor] = None  # (B..., nv)

    # ──────────── kinematics ────────────
    liMi: Optional[torch.Tensor] = None  # (B..., njoints, 7)
    oMi: Optional[torch.Tensor] = None  # (B..., njoints, 7)
    oMf: Optional[torch.Tensor] = None  # (B..., nframes, 7)

    ov: Optional[torch.Tensor] = None  # (B..., njoints, 6)
    oa: Optional[torch.Tensor] = None  # (B..., njoints, 6)
    v_joint: Optional[torch.Tensor] = None  # (B..., njoints, 6)
    a_joint: Optional[torch.Tensor] = None

    # ──────────── jacobians ────────────
    J: Optional[torch.Tensor] = None  # (B..., njoints, 6, nv)
    dJ: Optional[torch.Tensor] = None  # (B..., njoints, 6, nv)

    # ──────────── dynamics ────────────
    M: Optional[torch.Tensor] = None  # (B..., nv, nv)
    C: Optional[torch.Tensor] = None  # (B..., nv, nv)
    g: Optional[torch.Tensor] = None  # (B..., nv)
    nle: Optional[torch.Tensor] = None  # (B..., nv)
    ddq: Optional[torch.Tensor] = None  # (B..., nv)

    # ──────────── centroidal ────────────
    Ag: Optional[torch.Tensor] = None  # (B..., 6, nv)
    hg: Optional[torch.Tensor] = None  # (B..., 6)
    com: Optional[torch.Tensor] = None  # (B..., 3)
    vcom: Optional[torch.Tensor] = None  # (B..., 3)
    acom: Optional[torch.Tensor] = None  # (B..., 3)

    # ──────────── cache bookkeeping ────────────
    _kinematics_level: int = 0

    # ────────────────────────── methods ──────────────────────────

    def reset(self) -> None:
        """Set every optional tensor field to ``None`` and reset kinematics level."""
        for f in (
            "v", "a", "tau",
            "liMi", "oMi", "oMf",
            "ov", "oa", "v_joint", "a_joint",
            "J", "dJ",
            "M", "C", "g", "nle", "ddq",
            "Ag", "hg", "com", "vcom", "acom",
        ):
            object.__setattr__(self, f, None)
        object.__setattr__(self, "_kinematics_level", 0)

    def clone(self) -> "Data":
        """Return a deep copy, sharing ``_model_id``."""
        import copy
        return copy.deepcopy(self)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch shape, i.e. ``q.shape[:-1]``."""
        return tuple(self.q.shape[:-1])
