"""Test-support: a representative 24-body free-flyer + 23-spherical tree, built with ``ModelBuilder``.

Not shipped API. The bodies and joints share the names ``j0``..``j23``; ``PARENTS`` gives the tree and
``default_offsets_tensor`` arbitrary offsets, so BetterRobot's own suite has a representative free-flyer + spherical
model.
"""

from __future__ import annotations

import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.io.ir import IRModel
from better_robot.lie import se3
from better_robot.model import JointFreeFlyer, JointSpherical
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


def default_offsets_tensor() -> torch.Tensor:
    """Arbitrary ``(24, 3)`` per-joint offsets."""
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.09, 0.0],
            [0.0, -0.09, 0.0],
            [0.0, 0.0, 0.08],
            [0.0, 0.0, -0.42],
            [0.0, 0.0, -0.42],
            [0.0, 0.0, 0.13],
            [0.0, 0.0, -0.38],
            [0.0, 0.0, -0.38],
            [0.0, 0.0, 0.13],
            [0.12, 0.0, -0.06],
            [0.12, 0.0, -0.06],
            [0.0, 0.0, 0.25],
            [0.0, 0.05, 0.18],
            [0.0, -0.05, 0.18],
            [0.0, 0.0, 0.08],
            [0.0, 0.15, 0.0],
            [0.0, -0.15, 0.0],
            [0.0, 0.0, -0.27],
            [0.0, 0.0, -0.27],
            [0.0, 0.0, -0.24],
            [0.0, 0.0, -0.24],
            [0.0, 0.0, -0.08],
            [0.0, 0.0, -0.08],
        ],
        dtype=torch.float32,
    )


def make_branching_tree_body() -> IRModel:
    """Build the branching-tree ``IRModel``: the root joint ``root`` (free flyer) and 23 spherical joints.

    The joints' translations in their parents' frames are ``default_offsets_tensor()``; 70 kg are spread evenly.
    """
    offsets = default_offsets_tensor()
    builder = ModelBuilder("branching_tree")
    for joint_name in JOINT_NAMES:
        builder.add_body(joint_name, mass=70.0 / 24.0)
    builder.add_joint(
        "root", kind=JointFreeFlyer(), parent="world", child=JOINT_NAMES[0], origin=se3.from_translation(offsets[0])
    )
    for index in range(1, len(JOINT_NAMES)):
        builder.add_joint(
            JOINT_NAMES[index],
            kind=JointSpherical(),
            parent=JOINT_NAMES[PARENTS[index]],
            child=JOINT_NAMES[index],
            origin=se3.from_translation(offsets[index]),
        )
    return builder.finalize()


def make_branching_tree_model() -> Model:
    """Build the branching-tree frozen ``Model`` in float32. See :func:`make_branching_tree_body`."""
    return build_model(make_branching_tree_body())
