"""Every joint kind honours the ``JointModel`` contract, checked by ``check_joint_model``."""

from __future__ import annotations

import inspect

import pytest
import torch

from better_robot.exceptions import ArgumentError
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
    joints,
)
from better_robot.model.manifolds import CIRCLE, PLANE
from better_robot.testing.joints import check_joint_model
from tests.support.cases import failures

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


def test_kinds_cover_every_exported_joint_class() -> None:
    """A new concrete kind in ``model.joints.__all__`` joins ``KINDS``, the contract check's list, or fails here."""
    exported = {cls for name in joints.__all__ if not inspect.isabstract(cls := getattr(joints, name))}
    assert {type(param.values[0]) for param in KINDS} == exported


@pytest.mark.parametrize("joint", KINDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kind_honours_the_joint_model_contract(joint, seed: int) -> None:
    """In float32 and in float64."""

    def check(dtype: torch.dtype) -> None:
        check_joint_model(joint, dtype=dtype, generator=torch.Generator().manual_seed(seed))

    failed = failures(check, (torch.float32, torch.float64))
    assert not failed, "\n".join(failed)


def test_kinds_are_values() -> None:
    """Equal fields mean equal joints; the axis is data, not a class."""
    assert JointRevolute(axis=(0.0, 0.0, 1.0)) == JointRevolute(axis=(0.0, 0.0, 1.0))
    assert JointRevolute(axis=(0.0, 0.0, 1.0)) != JointRevolute(axis=(0.0, 1.0, 0.0))
    assert JointRevolute(axis=(0.0, 0.0, 1.0)) != JointPrismatic(axis=(0.0, 0.0, 1.0))
    assert hash(JointHelical(axis=_AXIS, pitch=0.25)) == hash(JointHelical(axis=_AXIS, pitch=0.25))
    assert JointFixed() == JointFixed()


AXIS_KINDS = {
    "revolute": lambda axis: JointRevolute(axis=axis),
    "continuous": lambda axis: JointRevoluteUnbounded(axis=axis),
    "prismatic": lambda axis: JointPrismatic(axis=axis),
    "helical": lambda axis: JointHelical(axis=axis, pitch=0.25),
}
BAD_AXES = {
    "non_unit": ((1.2, 1.6, 0.0), "has norm 2"),
    "zero": ((0.0, 0.0, 0.0), "has norm 0"),
    "two_entries": ((0.0, 1.0), "three finite floats"),
    "list": ([0.0, 0.0, 1.0], "three finite floats"),
    "tensor": (torch.tensor([0.0, 0.0, 1.0]), "three finite floats"),
    "nan": ((0.0, float("nan"), 1.0), "three finite floats"),
}


def test_axis_must_be_a_unit_tuple() -> None:
    """A kind with an axis accepts a tuple of three finite floats of unit norm and nothing else: the kinds are
    hashable values, and a non-unit axis would scale the placement and the Jacobian column."""

    def check(case: str) -> None:
        kind, bad = case.split("-")
        axis, match = BAD_AXES[bad]
        with pytest.raises(ArgumentError, match=match):
            AXIS_KINDS[kind](axis)

    failed = failures(check, [f"{kind}-{bad}" for kind in AXIS_KINDS for bad in BAD_AXES])
    assert not failed, "\n".join(failed)


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
