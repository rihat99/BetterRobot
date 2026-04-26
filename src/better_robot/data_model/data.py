"""``Data`` — mutable per-query workspace (Pinocchio-style).

Carries a leading batch (and optional time) axis. Kinematics/dynamics
functions populate the optional fields lazily — this avoids the mjwarp
280-field god-dataclass trap.

Field naming follows the ``<entity>_<quantity>_<frame>`` convention
documented in ``docs/conventions/naming.md``. Pinocchio-style short aliases
(``oMi``, ``oMf``, ``liMi``, ``nle``, ``Ag``, …) remain available as
deprecated ``@property`` shims for one release (see §11 of
``docs/concepts/model_and_data.md``); they forward read / write to the renamed
storage field and emit :class:`DeprecationWarning`.

The ``_kinematics_level`` field tracks how far the recursion has been
advanced: ``NONE`` < ``PLACEMENTS`` < ``VELOCITIES`` < ``ACCELERATIONS``.
Reassigning :attr:`q` / :attr:`v` / :attr:`a` invalidates strictly-higher
caches (see ``docs/concepts/model_and_data.md §3.1``). In-place mutation
(``data.q[..., 0] += 1.0``) is *not* detected — that is a documented
limitation.

See ``docs/concepts/model_and_data.md §3``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch

from ._kinematics_level import KinematicsLevel

# ──────────────────────────────────────────────────────────────────────
# Old-name → new-name mapping. The module installs ``@property`` shims
# for every entry below, post-``@dataclass`` decoration, so the old
# names keep working one more release.
#
# Removal ticket: docs/conventions/naming.md §6 (target: v1.1).
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


# Cache buckets per kinematic level. A field is at level ``L`` if its
# value depends on inputs through level ``L`` (see §3.1).
_PLACEMENT_CACHES: tuple[str, ...] = (
    "joint_pose_local", "joint_pose_world", "frame_pose_world",
    "joint_jacobians",
    "mass_matrix", "gravity_torque",
    "centroidal_momentum_matrix",
    "com_position",
)
_VELOCITY_CACHES: tuple[str, ...] = (
    "joint_velocity_world", "joint_velocity_local",
    "joint_jacobians_dot",
    "coriolis_matrix", "bias_forces",
    "centroidal_momentum",
    "com_velocity",
)
_ACCELERATION_CACHES: tuple[str, ...] = (
    "joint_acceleration_world", "joint_acceleration_local",
    "joint_forces", "ddq",
    "com_acceleration",
)

# Aggregate, used by :meth:`Data.reset`.
_CLEARABLE_FIELDS: tuple[str, ...] = (
    "v", "a", "tau",
    *_PLACEMENT_CACHES,
    *_VELOCITY_CACHES,
    *_ACCELERATION_CACHES,
)


@dataclass
class Data:
    """Mutable per-query workspace carrying leading batch dims ``B...``.

    Unused cache slots stay ``None``. ``_kinematics_level`` tracks how far
    the kinematic recursion has been run (``KinematicsLevel`` enum) so
    functions that need a level can call :meth:`require` and raise
    :class:`~better_robot.exceptions.StaleCacheError` on a mismatch.
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
    # RNEA / ABA internal spatial wrench per joint, body-frame (Pinocchio's data.f).
    joint_forces:             Optional[torch.Tensor] = None   # (B..., njoints, 6)

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
    _kinematics_level: KinematicsLevel = KinematicsLevel.NONE

    # ────────────────────────── methods ──────────────────────────

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        # Skip during dataclass ``__init__`` — the cache-bookkeeping field
        # is the *last* declared field, so its absence flags "not yet
        # constructed". Once it appears, q/v/a reassignment triggers
        # cache invalidation.
        if not hasattr(self, "_kinematics_level"):
            return
        if name == "q":
            self.invalidate(KinematicsLevel.NONE)
        elif name == "v":
            self.invalidate(KinematicsLevel.PLACEMENTS)
        elif name == "a":
            self.invalidate(KinematicsLevel.VELOCITIES)

    def reset(self) -> None:
        """Set every optional tensor field to ``None`` and reset kinematics level."""
        for f in _CLEARABLE_FIELDS:
            object.__setattr__(self, f, None)
        object.__setattr__(self, "_kinematics_level", KinematicsLevel.NONE)

    def invalidate(self, level: KinematicsLevel = KinematicsLevel.NONE) -> None:
        """Demote the kinematics level to ``level``, clearing strictly-higher caches.

        If ``level == NONE`` (the default) every cache field is reset.
        """
        target = int(level)
        if target < int(KinematicsLevel.ACCELERATIONS):
            for f in _ACCELERATION_CACHES:
                object.__setattr__(self, f, None)
        if target < int(KinematicsLevel.VELOCITIES):
            for f in _VELOCITY_CACHES:
                object.__setattr__(self, f, None)
        if target < int(KinematicsLevel.PLACEMENTS):
            for f in _PLACEMENT_CACHES:
                object.__setattr__(self, f, None)
        # Demote to ``level`` if currently above it; otherwise leave alone.
        current = int(self._kinematics_level)
        if current > target:
            object.__setattr__(self, "_kinematics_level", level)

    def require(self, level: KinematicsLevel) -> None:
        """Raise :class:`StaleCacheError` if the held kinematics level is below ``level``.

        Call this at the entry point of any function that depends on a
        specific level (e.g. ``compute_joint_jacobians`` requires
        ``PLACEMENTS``).
        """
        if int(self._kinematics_level) < int(level):
            from ..exceptions import StaleCacheError
            held = (
                self._kinematics_level.name
                if isinstance(self._kinematics_level, KinematicsLevel)
                else str(self._kinematics_level)
            )
            raise StaleCacheError(
                f"Data is at kinematics level {held}; need {level.name}. "
                f"Call forward_kinematics first."
            )

    def clone(self) -> "Data":
        """Return a deep copy, sharing ``_model_id``."""
        import copy
        return copy.deepcopy(self)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch shape, i.e. ``q.shape[:-1]``."""
        return tuple(self.q.shape[:-1])

    def joint_pose(self, joint_id: int):
        """Typed accessor for a single joint's world-frame pose.

        Returns an :class:`~better_robot.lie.types.SE3` value. Raises
        :class:`~better_robot.exceptions.StaleCacheError` if forward
        kinematics has not been computed yet.
        """
        if self.joint_pose_world is None:
            from ..exceptions import StaleCacheError
            raise StaleCacheError(
                "Data.joint_pose_world is None; call forward_kinematics first."
            )
        from ..lie.types import SE3
        return SE3(self.joint_pose_world[..., joint_id, :])

    def frame_pose(self, frame_id: int):
        """Typed accessor for a single frame's world-frame pose.

        Returns an :class:`~better_robot.lie.types.SE3` value. Raises
        :class:`~better_robot.exceptions.StaleCacheError` if frame
        placements have not been computed yet (call ``forward_kinematics``
        with ``compute_frames=True`` or ``update_frame_placements``).
        """
        if self.frame_pose_world is None:
            from ..exceptions import StaleCacheError
            raise StaleCacheError(
                "Data.frame_pose_world is None; call forward_kinematics "
                "with compute_frames=True (or update_frame_placements)."
            )
        from ..lie.types import SE3
        return SE3(self.frame_pose_world[..., frame_id, :])


# ══════════════════════════════════════════════════════════════════════
# Deprecated aliases (removed in v1.1).
# Installed post-``@dataclass`` so they don't participate in ``__init__``.
# See ``docs/concepts/model_and_data.md §11`` and ``docs/conventions/naming.md §6``.
# ══════════════════════════════════════════════════════════════════════

def _make_alias(old: str, new: str) -> property:
    msg = (
        f"Data.{old} is deprecated; use Data.{new}. "
        f"Will be removed in v1.1. See docs/conventions/naming.md §6."
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
