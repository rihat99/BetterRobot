"""``make_smpl_like_body`` — SMPL-topology ``IRModel`` with fixed shape params.

This is **not** an SMPL loader. It constructs a kinematic tree with the
SMPL 24-joint topology (pelvis as free-flyer root + 23 ball joints) and
body dimensions derived from the shape parameters, proving the data model
is expressive enough to host an SMPL-like body without any SMPL-specific
code in the core.

See ``docs/04_PARSERS.md §6`` (``SMPL-like body as motivation``).
"""

from __future__ import annotations

import torch

from ...data_model.model import Model
from ..build_model import build_model
from ..ir import IRModel
from ..parsers.programmatic import ModelBuilder

# SMPL 24-joint topology:
# 0 = pelvis (root, free-flyer)
# 1 = left_hip,  2 = right_hip,  3 = spine1
# 4 = left_knee, 5 = right_knee, 6 = spine2
# 7 = left_ankle,8 = right_ankle,9 = spine3
# 10= left_foot, 11= right_foot, 12= neck
# 13= left_collar,14=right_collar,15=head
# 16= left_shoulder,17=right_shoulder
# 18= left_elbow, 19=right_elbow
# 20= left_wrist, 21=right_wrist
# 22= left_hand,  23=right_hand

_JOINT_NAMES = [
    "pelvis",
    "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# Parent indices (SMPL skeleton, 0-indexed)
_PARENTS = [
    -1,   # pelvis → world (root)
    0, 0, 0,       # l_hip, r_hip, spine1
    1, 2, 3,       # l_knee, r_knee, spine2
    4, 5, 6,       # l_ankle, r_ankle, spine3
    7, 8, 9,       # l_foot, r_foot, neck
    12, 12, 12,    # l_collar, r_collar, head
    13, 14,        # l_shoulder, r_shoulder
    16, 17,        # l_elbow, r_elbow
    18, 19,        # l_wrist, r_wrist
    20, 21,        # l_hand, r_hand
]

# Default joint offsets (roughly human proportions, in metres).
# These are the SE3 offsets from the parent joint frame to the child joint frame.
# (tx, ty, tz) only; the rotation part is identity.
def _default_offsets(height: float, mass: float) -> dict[str, tuple[float, float, float]]:
    """Roughly-proportioned joint offsets for a human of given height."""
    s = height / 1.75  # scale factor
    return {
        "pelvis":         (0., 0., 0.),          # root; placed by free-flyer
        "left_hip":       (0., +0.09 * s, 0.),
        "right_hip":      (0., -0.09 * s, 0.),
        "spine1":         (0., 0.,  0.08 * s),
        "left_knee":      (0., 0., -0.42 * s),
        "right_knee":     (0., 0., -0.42 * s),
        "spine2":         (0., 0.,  0.13 * s),
        "left_ankle":     (0., 0., -0.38 * s),
        "right_ankle":    (0., 0., -0.38 * s),
        "spine3":         (0., 0.,  0.13 * s),
        "left_foot":      (0.12 * s, 0.,  -0.06 * s),
        "right_foot":     (0.12 * s, 0.,  -0.06 * s),
        "neck":           (0., 0.,  0.25 * s),
        "left_collar":    (0., +0.05 * s,  0.18 * s),
        "right_collar":   (0., -0.05 * s,  0.18 * s),
        "head":           (0., 0.,  0.08 * s),
        "left_shoulder":  (0., +0.15 * s,  0.),
        "right_shoulder": (0., -0.15 * s,  0.),
        "left_elbow":     (0., 0., -0.27 * s),
        "right_elbow":    (0., 0., -0.27 * s),
        "left_wrist":     (0., 0., -0.24 * s),
        "right_wrist":    (0., 0., -0.24 * s),
        "left_hand":      (0., 0., -0.08 * s),
        "right_hand":     (0., 0., -0.08 * s),
    }


def make_smpl_like_body(
    height: float = 1.75,
    mass: float = 70.0,
    *,
    shape_params: torch.Tensor | None = None,
) -> IRModel:
    """Build an SMPL-topology ``IRModel`` (free-flyer root + 23 ball joints).

    This is NOT an SMPL loader — it constructs a kinematic tree with the
    same 24-joint topology as SMPL (pelvis as free-flyer root, plus 23 ball
    joints for the body), with body dimensions derived from height and mass.

    See docs/04_PARSERS.md §6.
    """
    offsets = _default_offsets(height, mass)
    body_mass = mass / 24.0  # uniform distribution

    b = ModelBuilder("smpl_body")

    # Add all 24 segment bodies
    for jname in _JOINT_NAMES:
        b.add_body(jname, mass=body_mass)

    # Identity SE3 helper
    _id7 = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

    def _origin(name: str) -> torch.Tensor:
        tx, ty, tz = offsets[name]
        o = _id7.clone()
        o[0], o[1], o[2] = tx, ty, tz
        return o

    # Root: free_flyer joint connecting world → pelvis
    b.add_joint(
        "root",
        kind="free_flyer",
        parent="world",
        child="pelvis",
        origin=_origin("pelvis"),
    )

    # 23 spherical joints for all non-root segments
    for idx in range(1, 24):
        child_name = _JOINT_NAMES[idx]
        parent_idx = _PARENTS[idx]
        parent_name = _JOINT_NAMES[parent_idx] if parent_idx >= 0 else "world"
        b.add_joint(
            child_name,
            kind="spherical",
            parent=parent_name,
            child=child_name,
            origin=_origin(child_name),
        )

    return b.finalize()


def make_smpl_like_model(
    height: float = 1.75,
    mass: float = 70.0,
    *,
    shape_params: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Build an SMPL-topology frozen ``Model`` (free-flyer root + 23 ball joints)."""
    ir = make_smpl_like_body(height=height, mass=mass, shape_params=shape_params)
    return build_model(ir, device=device, dtype=dtype)
