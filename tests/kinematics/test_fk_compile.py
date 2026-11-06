"""The public forward kinematics traces as one graph; its quaternion check is opt-in."""

from __future__ import annotations

import pytest
import torch

from better_robot import load
from better_robot.dynamics.state import inverse_dynamics, mass_matrix
from better_robot.exceptions import QuaternionNormError
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics import forward_kinematics
from tests.support.robots import panda_urdf


@pytest.fixture(scope="module")
def free_flyer_model():
    builder = ModelBuilder("free_flyer_compile")
    builder.add_body("base", mass=1.0)
    builder.add_free_flyer_root(
        "floating",
        child="base",
        origin=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    return build_model(builder.finalize())


def test_free_flyer_fk_compiles_fullgraph(free_flyer_model):
    """The public wrapper (validation, memoised placements, the kernel) traces as one graph for a quaternion joint."""
    torch._dynamo.reset()
    q = free_flyer_model.q_neutral.unsqueeze(0).expand(2, -1).clone()
    expected = forward_kinematics(free_flyer_model, q)
    actual = torch.compile(lambda q: forward_kinematics(free_flyer_model, q).joint_transforms, fullgraph=True)(q)
    torch.testing.assert_close(actual, expected.joint_transforms)


def test_quaternion_norm_check_is_opt_in(free_flyer_model):
    q = free_flyer_model.q_neutral.clone()
    q[3:7] = 0.0

    # The default public path trusts the pre-normalized-input contract.
    forward_kinematics(free_flyer_model, q)

    with pytest.raises(QuaternionNormError, match="quaternion norm"):
        forward_kinematics(free_flyer_model, q, check_quaternion_norm=True)


@pytest.fixture(scope="module")
def panda64():
    return load(panda_urdf(), dtype=torch.float64)


def _fk(model, q):
    return forward_kinematics(model, q).joint_transforms


def _rnea(model, q, v, a):
    return inverse_dynamics(model, forward_kinematics(model, q), v=v, a=a)


def _crba(model, q):
    return mass_matrix(model, forward_kinematics(model, q))


@pytest.mark.parametrize("fn", [_fk, _rnea, _crba], ids=["fk", "rnea", "crba"])
def test_public_fk_rnea_crba_compile_fullgraph(panda64, fn):
    """Public FK, RNEA and CRBA trace as one graph (the memos step aside) and match eager."""
    torch._dynamo.reset()
    generator = torch.Generator().manual_seed(0)
    nv = panda64.nv
    q = panda64.q_neutral.expand(2, -1) + 0.1 * torch.randn(2, panda64.nq, generator=generator, dtype=torch.float64)
    extra = [torch.randn(2, nv, generator=generator, dtype=torch.float64) for _ in range(2)]
    args = (q, *extra) if fn is _rnea else (q,)
    expected = fn(panda64, *args)
    actual = torch.compile(fn, fullgraph=True)(panda64, *args)
    torch.testing.assert_close(actual, expected)
