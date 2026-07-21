"""Test-support: a representative 24-body free-flyer + 23-spherical tree.

Not shipped API. BetterRobot's public programmatic entry points are
:func:`build_kinematic_tree_body` / :func:`build_kinematic_tree_model`; this
module only supplies a fixed branching tree with neutral joint names
(``j0``..``j23``) and arbitrary offsets so BR's own suite has a representative
free-flyer + spherical model to exercise.
"""

from __future__ import annotations

import torch

from better_robot.data_model.model import Model
from better_robot.io.builders.kinematic_tree import (
    build_kinematic_tree_body,
    build_kinematic_tree_model,
)
from better_robot.io.ir import IRModel

JOINT_NAMES: tuple[str, ...] = tuple(f"j{index}" for index in range(24))

PARENTS: tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    12,
    12,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)


def default_offsets_tensor(height: float) -> torch.Tensor:
    """Arbitrary ``(24, 3)`` per-joint offsets scaled by ``height``."""
    scale = height / 1.75
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, +0.09 * scale, 0.0],
            [0.0, -0.09 * scale, 0.0],
            [0.0, 0.0, 0.08 * scale],
            [0.0, 0.0, -0.42 * scale],
            [0.0, 0.0, -0.42 * scale],
            [0.0, 0.0, 0.13 * scale],
            [0.0, 0.0, -0.38 * scale],
            [0.0, 0.0, -0.38 * scale],
            [0.0, 0.0, 0.13 * scale],
            [0.12 * scale, 0.0, -0.06 * scale],
            [0.12 * scale, 0.0, -0.06 * scale],
            [0.0, 0.0, 0.25 * scale],
            [0.0, +0.05 * scale, 0.18 * scale],
            [0.0, -0.05 * scale, 0.18 * scale],
            [0.0, 0.0, 0.08 * scale],
            [0.0, +0.15 * scale, 0.0],
            [0.0, -0.15 * scale, 0.0],
            [0.0, 0.0, -0.27 * scale],
            [0.0, 0.0, -0.27 * scale],
            [0.0, 0.0, -0.24 * scale],
            [0.0, 0.0, -0.24 * scale],
            [0.0, 0.0, -0.08 * scale],
            [0.0, 0.0, -0.08 * scale],
        ],
        dtype=torch.float32,
    )


def make_branching_tree_body(
    height: float = 1.75,
    mass: float = 70.0,
    *,
    name: str = "branching_tree",
    joint_offsets: torch.Tensor | None = None,
) -> IRModel:
    """Build the branching-tree ``IRModel`` (free-flyer root + 23 spherical joints)."""
    offsets = joint_offsets if joint_offsets is not None else default_offsets_tensor(height)
    return build_kinematic_tree_body(
        name=name,
        joint_names=JOINT_NAMES,
        parents=PARENTS,
        translations=offsets,
        root_kind="free_flyer",
        child_kind="spherical",
        mass_per_body=mass / 24.0,
    )


def make_branching_tree_model(
    height: float = 1.75,
    mass: float = 70.0,
    *,
    name: str = "branching_tree",
    joint_offsets: torch.Tensor | None = None,
    preserve_joint_order: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Build the branching-tree frozen ``Model``. See :func:`make_branching_tree_body`."""
    offsets = joint_offsets if joint_offsets is not None else default_offsets_tensor(height)
    return build_kinematic_tree_model(
        name=name,
        joint_names=JOINT_NAMES,
        parents=PARENTS,
        translations=offsets,
        root_kind="free_flyer",
        child_kind="spherical",
        mass_per_body=mass / 24.0,
        preserve_joint_order=preserve_joint_order,
        device=device,
        dtype=dtype,
    )
