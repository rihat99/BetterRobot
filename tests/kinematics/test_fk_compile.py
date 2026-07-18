"""Compilation and opt-in validation contracts for forward kinematics."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import QuaternionNormError
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import (
    forward_kinematics,
    forward_kinematics_raw,
)


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


def test_free_flyer_fk_raw_compiles_fullgraph(free_flyer_model):
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        pytest.skip("torch.compile is unavailable")

    q = free_flyer_model.q_neutral.unsqueeze(0).expand(2, -1).clone()
    expected = forward_kinematics_raw(free_flyer_model.structure, free_flyer_model.values, q)
    compiled_fk = compile_fn(forward_kinematics_raw, fullgraph=True)
    actual = compiled_fk(free_flyer_model.structure, free_flyer_model.values, q)

    torch.testing.assert_close(actual.joint_pose_world, expected.joint_pose_world)
    torch.testing.assert_close(actual.joint_pose_local, expected.joint_pose_local)


def test_free_flyer_quaternion_norm_check_is_opt_in(free_flyer_model):
    q = free_flyer_model.q_neutral.clone()
    q[3:7] = 0.0

    # Raw FK and the default public path trust the pre-normalized-input contract.
    forward_kinematics_raw(free_flyer_model.structure, free_flyer_model.values, q)
    forward_kinematics(free_flyer_model, q)

    with pytest.raises(QuaternionNormError, match="quaternion norm"):
        forward_kinematics(
            free_flyer_model,
            q,
            check_quaternion_norm=True,
        )
