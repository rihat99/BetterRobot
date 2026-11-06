"""Test-support: a representative 24-body free-flyer + 23-spherical tree, built with ``ModelBuilder``.

Not shipped API. The bodies and joints share the names ``j0``..``j23``; ``PARENTS`` gives the tree and
``default_offsets_tensor`` arbitrary offsets, so BR's own suite has a representative free-flyer + spherical model.
"""

from __future__ import annotations

import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.io.ir import IRModel
from better_robot.lie import se3
from better_robot.model.model import Model

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
    """Build the branching-tree ``IRModel``: the root joint ``root`` (free flyer) and 23 spherical joints.

    ``joint_offsets (24, 3)`` are the joints' translations in their parents' frames; ``mass`` is spread evenly.
    """
    offsets = joint_offsets if joint_offsets is not None else default_offsets_tensor(height)
    builder = ModelBuilder(name)
    for joint_name in JOINT_NAMES:
        builder.add_body(joint_name, mass=mass / 24.0)
    builder.add_free_flyer_root("root", child=JOINT_NAMES[0], origin=se3.from_translation(offsets[0]))
    for index in range(1, len(JOINT_NAMES)):
        builder.add_spherical(
            JOINT_NAMES[index],
            parent=JOINT_NAMES[PARENTS[index]],
            child=JOINT_NAMES[index],
            origin=se3.from_translation(offsets[index]),
        )
    return builder.finalize()


def make_branching_tree_model(
    height: float = 1.75,
    mass: float = 70.0,
    *,
    name: str = "branching_tree",
    joint_offsets: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Build the branching-tree frozen ``Model``. See :func:`make_branching_tree_body`."""
    return build_model(
        make_branching_tree_body(height, mass, name=name, joint_offsets=joint_offsets),
        device=device,
        dtype=dtype,
    )
