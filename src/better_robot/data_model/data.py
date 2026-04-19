"""``Data`` — mutable per-query workspace (Pinocchio-style).

Carries a leading batch (and optional time) axis. Kinematics/dynamics
functions populate the optional fields lazily — this avoids the mjwarp
280-field god-dataclass trap.

Field naming follows the ``<entity>_<quantity>_<frame>`` convention
documented in ``docs/13_NAMING.md``. Pinocchio-style short aliases
(``oMi``, ``oMf``, ``liMi``, ``nle``, ``Ag``, …) remain available as
deprecated ``@property`` shims for one release (see §11 of
``docs/02_DATA_MODEL.md``); they forward read / write to the renamed
storage field and emit :class:`DeprecationWarning`.

See ``docs/02_DATA_MODEL.md §3`` for the field inventory.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch

# ──────────────────────────────────────────────────────────────────────
# Old-name → new-name mapping. The module installs ``@property`` shims
# for every entry below, post-``@dataclass`` decoration, so the old
# names keep working one more release.
#
# Removal ticket: docs/13_NAMING.md §6 (target: v1.1).
# ──────────────────────────────────────────────────────────────────────

_DEPRECATED_ALIASES: tuple[tuple[str, str], ...] = (
    ("oMi", "joint_pose_world"),
    ("oMf", "frame_pose_world"),
    ("liMi", "joint_pose_local"),
    ("ov", "joint_velocity_world"),
    ("oa", "joint_acceleration_world"),
    ("v_joint", "joint_velocity_local"),
    ("a_joint", "joint_acceleration_local"),
    ("M", "mass_matrix"),
    ("C", "coriolis_matrix"),
    ("g", "gravity_torque"),
    ("nle", "bias_forces"),
    ("J", "joint_jacobians"),
    ("dJ", "joint_jacobians_dot"),
    ("Ag", "centroidal_momentum_matrix"),
    ("hg", "centroidal_momentum"),
    ("com", "com_position"),
    ("vcom", "com_velocity"),
    ("acom", "com_acceleration"),
)


# The set of optional fields that :meth:`Data.reset` clears back to
# ``None``. Kept separate so the rename diff is grep-friendly.
_CLEARABLE_FIELDS: tuple[str, ...] = (
    "v", "a", "tau",
    "joint_pose_local", "joint_pose_world", "frame_pose_world",
    "joint_velocity_world", "joint_velocity_local",
    "joint_acceleration_world", "joint_acceleration_local",
    "joint_jacobians", "joint_jacobians_dot",
    "mass_matrix", "coriolis_matrix", "gravity_torque", "bias_forces", "ddq",
    "centroidal_momentum_matrix", "centroidal_momentum",
    "com_position", "com_velocity", "com_acceleration",
)


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
    q: torch.Tensor                                 # (B..., nq)
    v: Optional[torch.Tensor] = None                # (B..., nv)
    a: Optional[torch.Tensor] = None                # (B..., nv)
    tau: Optional[torch.Tensor] = None              # (B..., nv)

    # ──────────── kinematics placements ────────────
    # Parent-frame joint placement — ``T_{i-1, i}`` in Featherstone notation.
    joint_pose_local:   Optional[torch.Tensor] = None    # (B..., njoints, 7)
    # World-frame joint placement — ``T_{0, i}`` (was ``oMi``).
    joint_pose_world:   Optional[torch.Tensor] = None    # (B..., njoints, 7)
    # World-frame operational frame placement — ``T_{0, f}`` (was ``oMf``).
    frame_pose_world:   Optional[torch.Tensor] = None    # (B..., nframes, 7)

    # ──────────── kinematics velocities / accelerations ────────────
    joint_velocity_world:     Optional[torch.Tensor] = None   # (B..., njoints, 6)
    joint_velocity_local:     Optional[torch.Tensor] = None   # (B..., njoints, 6)
    joint_acceleration_world: Optional[torch.Tensor] = None   # (B..., njoints, 6)
    joint_acceleration_local: Optional[torch.Tensor] = None   # (B..., njoints, 6)

    # ──────────── jacobians ────────────
    joint_jacobians:     Optional[torch.Tensor] = None    # (B..., njoints, 6, nv)
    joint_jacobians_dot: Optional[torch.Tensor] = None    # (B..., njoints, 6, nv)

    # ──────────── dynamics ────────────
    mass_matrix:     Optional[torch.Tensor] = None    # (B..., nv, nv) — M(q)
    coriolis_matrix: Optional[torch.Tensor] = None    # (B..., nv, nv) — C(q, v)
    gravity_torque:  Optional[torch.Tensor] = None    # (B..., nv)     — g(q)
    bias_forces:     Optional[torch.Tensor] = None    # (B..., nv)     — C(q,v)v + g(q)
    ddq:             Optional[torch.Tensor] = None    # (B..., nv)     — generalized accelerations

    # ──────────── centroidal ────────────
    centroidal_momentum_matrix: Optional[torch.Tensor] = None   # (B..., 6, nv) — A_g(q)
    centroidal_momentum:        Optional[torch.Tensor] = None   # (B..., 6)     — h_g = A_g v
    com_position:               Optional[torch.Tensor] = None   # (B..., 3)
    com_velocity:               Optional[torch.Tensor] = None   # (B..., 3)
    com_acceleration:           Optional[torch.Tensor] = None   # (B..., 3)

    # ──────────── cache bookkeeping ────────────
    _kinematics_level: int = 0

    # ────────────────────────── methods ──────────────────────────

    def reset(self) -> None:
        """Set every optional tensor field to ``None`` and reset kinematics level."""
        for f in _CLEARABLE_FIELDS:
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


# ══════════════════════════════════════════════════════════════════════
# Deprecated aliases (removed in v1.1).
# Installed post-``@dataclass`` so they don't participate in ``__init__``.
# See ``docs/02_DATA_MODEL.md §11`` and ``docs/13_NAMING.md §6``.
# ══════════════════════════════════════════════════════════════════════

def _make_alias(old: str, new: str) -> property:
    msg = (
        f"Data.{old} is deprecated; use Data.{new}. "
        f"Will be removed in v1.1. See docs/13_NAMING.md §6."
    )

    def _get(self: "Data") -> Optional[torch.Tensor]:
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return getattr(self, new)

    def _set(self: "Data", value: Optional[torch.Tensor]) -> None:
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        object.__setattr__(self, new, value)

    return property(_get, _set)


for _old_name, _new_name in _DEPRECATED_ALIASES:
    setattr(Data, _old_name, _make_alias(_old_name, _new_name))

del _old_name, _new_name, _make_alias
