"""Compilation and opt-in validation contracts for forward kinematics."""

from __future__ import annotations

import pytest
import torch

from better_robot import load
from better_robot.dynamics.state import inverse_dynamics, mass_matrix
from better_robot.exceptions import QuaternionNormError
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics import forward_kinematics_state
from better_robot.kinematics._fk_lean import lean_core, placement_matrices


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


def test_free_flyer_fk_kernel_compiles_fullgraph(free_flyer_model):
    """The FK kernel under ``forward_kinematics_state`` traces as one graph (the public wrapper validates and caches)."""
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        pytest.skip("torch.compile is unavailable")

    q = free_flyer_model.q_neutral.unsqueeze(0).expand(2, -1).clone()
    q_full = free_flyer_model.expand_configuration(q)
    placements, _ = placement_matrices(free_flyer_model.values)
    expected_world, expected_local = lean_core(free_flyer_model.structure, q_full, placements)
    compiled_core = compile_fn(lean_core, fullgraph=True)
    actual_world, actual_local = compiled_core(free_flyer_model.structure, q_full, placements)

    torch.testing.assert_close(actual_world, expected_world)
    torch.testing.assert_close(actual_local, expected_local)
    torch.testing.assert_close(forward_kinematics_state(free_flyer_model, q).joint_transform, expected_world)


def test_free_flyer_quaternion_norm_check_is_opt_in(free_flyer_model):
    q = free_flyer_model.q_neutral.clone()
    q[3:7] = 0.0

    # The default public path trusts the pre-normalized-input contract.
    forward_kinematics_state(free_flyer_model, q)

    with pytest.raises(QuaternionNormError, match="quaternion norm"):
        forward_kinematics_state(
            free_flyer_model,
            q,
            check_quaternion_norm=True,
        )



@pytest.fixture(scope="module")
def panda64():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415 - optional test dependency

    return load(panda_description.URDF_PATH, dtype=torch.float64)


def _fk(model, q):
    return forward_kinematics_state(model, q).joint_transform


def _rnea(model, q, v, a):
    return inverse_dynamics(model, forward_kinematics_state(model, q), v=v, a=a)


def _crba(model, q):
    return mass_matrix(model, forward_kinematics_state(model, q))


@pytest.mark.parametrize("fn", [_fk, _rnea, _crba], ids=["fk", "rnea", "crba"])
def test_public_fk_rnea_crba_compile_fullgraph(panda64, fn):
    """Public FK, RNEA and CRBA trace as one graph (placement/inertia caches step aside) and match eager."""
    torch._dynamo.reset()
    generator = torch.Generator().manual_seed(0)
    nv = panda64.nv
    q = panda64.q_neutral.expand(2, -1) + 0.1 * torch.randn(2, panda64.nq, generator=generator, dtype=torch.float64)
    extra = [torch.randn(2, nv, generator=generator, dtype=torch.float64) for _ in range(2)]
    args = (q, *extra) if fn is _rnea else (q,)
    expected = fn(panda64, *args)
    actual = torch.compile(fn, fullgraph=True)(panda64, *args)
    torch.testing.assert_close(actual, expected)
