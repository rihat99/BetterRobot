"""Every joint kind honours the ``JointModel`` contract (record 0013), checked by ``check_joint_model``."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import ArgumentError
from better_robot.io import ModelBuilder
from better_robot.model import (
    JointFixed,
    JointFreeFlyer,
    JointHelical,
    JointPlanar,
    JointPrismatic,
    JointRevolute,
    JointRevoluteUnbounded,
    JointSpherical,
    JointTranslation,
)
from better_robot.model.manifolds import CIRCLE, PLANE
from better_robot.testing.joints import check_joint_model

_AXIS = (0.6, 0.0, 0.8)
KINDS = [
    pytest.param(JointFixed(), id="fixed"),
    pytest.param(JointRevolute(axis=_AXIS), id="revolute"),
    pytest.param(JointRevoluteUnbounded(axis=_AXIS), id="continuous"),
    pytest.param(JointPrismatic(axis=_AXIS), id="prismatic"),
    pytest.param(JointHelical(axis=_AXIS, pitch=0.25), id="helical"),
    pytest.param(JointSpherical(), id="spherical"),
    pytest.param(JointPlanar(), id="planar"),
    pytest.param(JointTranslation(), id="translation"),
    pytest.param(JointFreeFlyer(), id="free_flyer"),
]


@pytest.mark.parametrize("joint", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kind_honours_the_joint_model_contract(joint, dtype: torch.dtype, seed: int) -> None:
    check_joint_model(joint, dtype=dtype, generator=torch.Generator().manual_seed(seed))


def test_kinds_are_values() -> None:
    """Equal fields mean equal joints; the axis is data, not a class."""
    assert JointRevolute(axis=(0.0, 0.0, 1.0)) == JointRevolute(axis=(0.0, 0.0, 1.0))
    assert JointRevolute(axis=(0.0, 0.0, 1.0)) != JointRevolute(axis=(0.0, 1.0, 0.0))
    assert JointRevolute(axis=(0.0, 0.0, 1.0)) != JointPrismatic(axis=(0.0, 0.0, 1.0))
    assert hash(JointHelical(axis=_AXIS, pitch=0.25)) == hash(JointHelical(axis=_AXIS, pitch=0.25))
    assert JointFixed() == JointFixed()


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda axis: JointRevolute(axis=axis), id="revolute"),
        pytest.param(lambda axis: JointRevoluteUnbounded(axis=axis), id="continuous"),
        pytest.param(lambda axis: JointPrismatic(axis=axis), id="prismatic"),
        pytest.param(lambda axis: JointHelical(axis=axis, pitch=0.25), id="helical"),
    ],
)
@pytest.mark.parametrize(
    ("axis", "match"),
    [
        pytest.param((1.2, 1.6, 0.0), "has norm 2", id="non_unit"),
        pytest.param((0.0, 0.0, 0.0), "has norm 0", id="zero"),
        pytest.param((0.0, 1.0), "three finite floats", id="two_entries"),
        pytest.param([0.0, 0.0, 1.0], "three finite floats", id="list"),
        pytest.param(torch.tensor([0.0, 0.0, 1.0]), "three finite floats", id="tensor"),
        pytest.param((0.0, float("nan"), 1.0), "three finite floats", id="nan"),
    ],
)
def test_axis_must_be_a_unit_tuple(make, axis, match: str) -> None:
    """A kind with an axis accepts a tuple of three finite floats of unit norm and nothing else: the kinds are
    hashable values, and a non-unit axis would scale the placement and the Jacobian column (R1+R2 review)."""
    with pytest.raises(ArgumentError, match=match):
        make(axis)


def test_builder_builds_the_kind_at_the_call_so_a_non_unit_axis_fails_there() -> None:
    builder = ModelBuilder("axis")
    base, link = builder.add_body("base", mass=1.0), builder.add_body("link", mass=1.0)
    with pytest.raises(ArgumentError, match="has norm 2"):
        builder.add_revolute("joint", parent=base, child=link, axis=torch.tensor([1.2, 1.6, 0.0]))


def test_circle_difference_is_the_wrapped_angle() -> None:
    q0 = torch.tensor([[torch.cos(torch.tensor(3.0)), torch.sin(torch.tensor(3.0))]], dtype=torch.float64)
    q1 = CIRCLE.integrate(q0, torch.tensor([[0.5]], dtype=torch.float64))  # the angle 3.5 wraps past pi
    torch.testing.assert_close(CIRCLE.difference(q0, q1), torch.tensor([[0.5]], dtype=torch.float64))


def test_plane_is_the_se2_exponential() -> None:
    """A forward speed with a yaw rate traces an arc, as the free flyer does in three dimensions."""
    q = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    arc = PLANE.integrate(q, torch.tensor([1.0, 0.0, torch.pi / 2], dtype=torch.float64))
    radius = 2.0 / torch.pi  # arc length 1 over a quarter turn
    torch.testing.assert_close(arc, torch.tensor([radius, radius, 0.0, 1.0], dtype=torch.float64))
