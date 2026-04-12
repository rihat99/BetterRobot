"""``Joint`` — string enum of joint kinds + ``(nq, nv)`` lookup table.

Pinocchio-style ``Joint`` enumeration. ``Joint`` is a string-valued
``enum.Enum`` subclass so downstream code can write ``Joint.FREE_FLYER``
and still do cheap equality checks against the raw ``kind`` string on
``JointModel``.

See ``docs/02_DATA_MODEL.md §5``.
"""

from __future__ import annotations

from enum import Enum


class Joint(str, Enum):
    """String enum of supported joint kinds (mirrors ``JointKind`` Literal)."""

    UNIVERSE = "universe"
    FIXED = "fixed"
    REVOLUTE_RX = "revolute_rx"
    REVOLUTE_RY = "revolute_ry"
    REVOLUTE_RZ = "revolute_rz"
    REVOLUTE_UNALIGNED = "revolute_unaligned"
    REVOLUTE_UNBOUNDED = "revolute_unbounded"
    PRISMATIC_PX = "prismatic_px"
    PRISMATIC_PY = "prismatic_py"
    PRISMATIC_PZ = "prismatic_pz"
    PRISMATIC_UNALIGNED = "prismatic_unaligned"
    SPHERICAL = "spherical"
    FREE_FLYER = "free_flyer"
    PLANAR = "planar"
    TRANSLATION = "translation"
    HELICAL = "helical"
    COMPOSITE = "composite"
    MIMIC = "mimic"


# (nq, nv) by kind for the fixed-shape joint families.
JOINT_DIMENSIONS: dict[str, tuple[int, int]] = {
    "universe": (0, 0),
    "fixed": (0, 0),
    "revolute_rx": (1, 1),
    "revolute_ry": (1, 1),
    "revolute_rz": (1, 1),
    "revolute_unaligned": (1, 1),
    "revolute_unbounded": (2, 1),
    "prismatic_px": (1, 1),
    "prismatic_py": (1, 1),
    "prismatic_pz": (1, 1),
    "prismatic_unaligned": (1, 1),
    "spherical": (4, 3),
    "free_flyer": (7, 6),
    "planar": (4, 3),
    "translation": (3, 3),
    "helical": (1, 1),
    "mimic": (0, 0),
    # composite is data-dependent; no default
}

__all__ = ["Joint", "JOINT_DIMENSIONS"]
