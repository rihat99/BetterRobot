"""The test robots: the two tracked URDFs, loaded by path so that the tests using them run offline, and the small arms
that several test folders build.

Source, licence and changes of each tracked robot: ``tests/fixtures/robots/panda/README.md``,
``tests/fixtures/robots/g1/README.md``.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import torch

from better_robot.io import ModelBuilder, build_model, parse_urdf
from better_robot.io.ir import IRModel
from better_robot.model import JointRevolute, Model

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "robots"


def panda_urdf() -> Path:
    """Path of the Franka Emika Panda URDF: the arm and its two-finger gripper."""
    return FIXTURES / "panda" / "panda.urdf"


def g1_urdf() -> Path:
    """Path of the Unitree G1 URDF with 29 actuated joints; load it with ``root_joint=JointFreeFlyer()`` for the floating base."""
    return FIXTURES / "g1" / "g1_29dof.urdf"


def panda_without_mimic() -> Model:
    """The Panda in float64 with its mimic metadata dropped, so the two fingers are independent coordinates (nq 9).

    Pinocchio's default URDF loader keeps mimic coordinates independent, and ``tests/kinematics/fk_reference.npz``
    predates reduced mimic coordinates: both compare the full-space recursion, which ``load(panda_urdf())`` (one
    finger coordinate, nq 8) does not run.
    """
    ir = parse_urdf(panda_urdf())
    ir.joints = [
        dataclasses.replace(joint, mimic_source=None, mimic_multiplier=1.0, mimic_offset=0.0) for joint in ir.joints
    ]
    return build_model(ir, dtype=torch.float64)


def one_revolute_arm_ir() -> IRModel:
    """The one-joint arm: ``base`` (0.5 kg) and ``link1`` (1 kg) joined by the revolute ``j1`` about z, 0.1 m above
    ``base``, with position limits ``[-pi, pi]``, velocity limit 2 and effort limit 100."""
    builder = ModelBuilder("arm")
    builder.add_body("base", mass=0.5)
    builder.add_body("link1", mass=1.0)
    builder.add_joint(
        "j1",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
        parent="base",
        child="link1",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
        velocity_limit=2.0,
        effort_limit=100.0,
    )
    return builder.finalize()


def two_link_arm(
    dtype: torch.dtype,
    device: torch.device | None = None,
    *,
    axes: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
) -> Model:
    """The arm ``base -j1-> link1 -j2-> link2`` of two revolute joints about ``axes``, limits ``[-pi, pi]``, with
    offset masses so that ``M(q)`` is positive definite.

    Built in float32 and then moved to ``device`` and ``dtype``, so a float64 arm holds float32-rounded values. Other
    ``axes`` give an arm of the same sizes with another structure.
    """
    builder = ModelBuilder("two_link_arm")
    inertia = 0.01 * torch.eye(3)
    builder.add_body("base", mass=0.5, inertia=inertia)
    builder.add_body("link1", mass=1.0, com=torch.tensor([0.1, 0.0, 0.0]), inertia=inertia)
    builder.add_body("link2", mass=0.7, com=torch.tensor([0.1, 0.0, 0.0]), inertia=inertia)
    origin = torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    builder.add_joint(
        "j1",
        kind=JointRevolute(axis=axes[0]),
        parent="base",
        child="link1",
        origin=origin,
        lower=-math.pi,
        upper=math.pi,
    )
    builder.add_joint(
        "j2",
        kind=JointRevolute(axis=axes[1]),
        parent="link1",
        child="link2",
        origin=origin,
        lower=-math.pi,
        upper=math.pi,
    )
    return build_model(builder.finalize()).to(device=device, dtype=dtype)
