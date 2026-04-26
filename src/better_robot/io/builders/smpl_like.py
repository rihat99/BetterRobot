"""``make_smpl_like_body`` — SMPL-topology ``IRModel`` with fixed shape params.

This is **not** an SMPL loader. It constructs a kinematic tree with the SMPL
24-joint topology (pelvis as free-flyer root + 23 ball joints) and body
dimensions derived from height (or supplied ``joint_offsets``), proving the
data model is expressive enough to host an SMPL-like body without any
SMPL-specific code in the core.

Internally delegates to ``build_kinematic_tree_body`` — the generic
array-driven builder in this package. Callers who want realistic per-body
mass / COM / rotational inertia can supply ``mass_per_body`` / ``com_per_body``
/ ``inertia_per_body`` explicitly (computed however they like — e.g., via an
SMPL mesh integrator outside BetterRobot). Leaving those kwargs as their
defaults reproduces the legacy uniform-``mass / 24`` behavior.

See ``docs/concepts/parsers_and_ir.md §6`` (``SMPL-like body as motivation``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...data_model.model import Model
from ..ir import IRModel
from .kinematic_tree import build_kinematic_tree_body, build_kinematic_tree_model

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

JOINT_NAMES: tuple[str, ...] = (
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
)

PARENTS: tuple[int, ...] = (
    -1,               # pelvis → world (root)
    0, 0, 0,          # l_hip, r_hip, spine1
    1, 2, 3,          # l_knee, r_knee, spine2
    4, 5, 6,          # l_ankle, r_ankle, spine3
    7, 8, 9,          # l_foot, r_foot, neck
    12, 12, 12,       # l_collar, r_collar, head
    13, 14,           # l_shoulder, r_shoulder
    16, 17,           # l_elbow, r_elbow
    18, 19,           # l_wrist, r_wrist
    20, 21,           # l_hand, r_hand
)


def _default_offsets_tensor(height: float) -> torch.Tensor:
    """Roughly-proportioned ``(24, 3)`` joint offsets scaled by height (metres)."""
    s = height / 1.75
    return torch.tensor(
        [
            [0., 0., 0.],                 # pelvis (root; placed by free-flyer)
            [0., +0.09 * s, 0.],          # left_hip
            [0., -0.09 * s, 0.],          # right_hip
            [0., 0., 0.08 * s],           # spine1
            [0., 0., -0.42 * s],          # left_knee
            [0., 0., -0.42 * s],          # right_knee
            [0., 0., 0.13 * s],           # spine2
            [0., 0., -0.38 * s],          # left_ankle
            [0., 0., -0.38 * s],          # right_ankle
            [0., 0., 0.13 * s],           # spine3
            [0.12 * s, 0., -0.06 * s],    # left_foot
            [0.12 * s, 0., -0.06 * s],    # right_foot
            [0., 0., 0.25 * s],           # neck
            [0., +0.05 * s, 0.18 * s],    # left_collar
            [0., -0.05 * s, 0.18 * s],    # right_collar
            [0., 0., 0.08 * s],           # head
            [0., +0.15 * s, 0.],          # left_shoulder
            [0., -0.15 * s, 0.],          # right_shoulder
            [0., 0., -0.27 * s],          # left_elbow
            [0., 0., -0.27 * s],          # right_elbow
            [0., 0., -0.24 * s],          # left_wrist
            [0., 0., -0.24 * s],          # right_wrist
            [0., 0., -0.08 * s],          # left_hand
            [0., 0., -0.08 * s],          # right_hand
        ],
        dtype=torch.float32,
    )


def make_smpl_like_body(
    height: float = 1.75,
    mass: float = 70.0,
    *,
    name: str = "smpl_body",
    shape_params: torch.Tensor | None = None,
    joint_offsets: torch.Tensor | None = None,
    mass_per_body: float | Sequence[float] | None = None,
    com_per_body: torch.Tensor | Sequence[torch.Tensor] | None = None,
    inertia_per_body: torch.Tensor | Sequence[torch.Tensor] | None = None,
) -> IRModel:
    """Build an SMPL-topology ``IRModel`` (free-flyer root + 23 ball joints).

    If ``joint_offsets`` (``(24, 3)`` tensor) is supplied, it overrides the
    height-scaled default proportions — useful for shape-aware callers that
    derive offsets from SMPL ``betas``.

    ``mass_per_body`` / ``com_per_body`` / ``inertia_per_body`` are passed
    straight through to :func:`build_kinematic_tree_body`. Supply them when
    you have per-body inertial parameters (e.g. computed from an SMPL mesh
    by an external tool); leave them ``None`` to fall back to the legacy
    uniform-``mass / 24`` default.
    """
    offsets = (
        joint_offsets if joint_offsets is not None else _default_offsets_tensor(height)
    )
    mass_kw = mass_per_body if mass_per_body is not None else mass / 24.0
    return build_kinematic_tree_body(
        name=name,
        joint_names=JOINT_NAMES,
        parents=PARENTS,
        translations=offsets,
        root_kind="free_flyer",
        child_kind="spherical",
        mass_per_body=mass_kw,
        com_per_body=com_per_body,
        inertia_per_body=inertia_per_body,
    )


def make_smpl_like_model(
    height: float = 1.75,
    mass: float = 70.0,
    *,
    name: str = "smpl_body",
    shape_params: torch.Tensor | None = None,
    joint_offsets: torch.Tensor | None = None,
    mass_per_body: float | Sequence[float] | None = None,
    com_per_body: torch.Tensor | Sequence[torch.Tensor] | None = None,
    inertia_per_body: torch.Tensor | Sequence[torch.Tensor] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Build an SMPL-topology frozen ``Model``. See :func:`make_smpl_like_body`."""
    offsets = (
        joint_offsets if joint_offsets is not None else _default_offsets_tensor(height)
    )
    mass_kw = mass_per_body if mass_per_body is not None else mass / 24.0
    return build_kinematic_tree_model(
        name=name,
        joint_names=JOINT_NAMES,
        parents=PARENTS,
        translations=offsets,
        root_kind="free_flyer",
        child_kind="spherical",
        mass_per_body=mass_kw,
        com_per_body=com_per_body,
        inertia_per_body=inertia_per_body,
        device=device,
        dtype=dtype,
    )
