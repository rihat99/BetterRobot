"""Parity safety net for joint families absent from the Panda/G1 fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot.data_model.joint_models import (
    JointHelical,
    JointPlanar,
    JointRevoluteUnaligned,
    JointRevoluteUnbounded,
    JointTranslation,
)
from better_robot.io import ModelBuilder, build_model

from .conftest import rot_matrix_from_pose

pin = pytest.importorskip("pinocchio")

_AXIS = np.array([1.0, 2.0, 3.0], dtype=np.float64)
_AXIS /= np.linalg.norm(_AXIS)


def _planar_q() -> torch.Tensor:
    theta = torch.tensor(0.3, dtype=torch.float64)
    return torch.stack(
        [
            torch.tensor(0.2, dtype=torch.float64),
            torch.tensor(-0.1, dtype=torch.float64),
            torch.cos(theta),
            torch.sin(theta),
        ]
    )


@pytest.fixture(
    params=[
        pytest.param(
            (
                JointPlanar,
                pin.JointModelPlanar,
                _planar_q(),
                torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
            ),
            id="planar",
        ),
        pytest.param(
            (
                JointTranslation,
                pin.JointModelTranslation,
                torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64),
                torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
            ),
            id="translation",
        ),
        pytest.param(
            (
                lambda: JointRevoluteUnbounded(torch.tensor([0.0, 0.0, 1.0])),
                pin.JointModelRUBZ,
                torch.tensor([np.cos(0.3), np.sin(0.3)], dtype=torch.float64),
                torch.tensor([0.2], dtype=torch.float64),
            ),
            id="revolute_unbounded",
        ),
        pytest.param(
            (
                lambda: JointRevoluteUnaligned(torch.from_numpy(_AXIS.copy())),
                lambda: pin.JointModelRevoluteUnaligned(_AXIS),
                torch.tensor([0.3], dtype=torch.float64),
                torch.tensor([0.2], dtype=torch.float64),
            ),
            id="revolute_unaligned",
        ),
        pytest.param(
            (
                lambda: JointHelical(torch.from_numpy(_AXIS.copy()), pitch=0.25),
                lambda: pin.JointModelHelicalUnaligned(_AXIS, 0.25),
                torch.tensor([0.3], dtype=torch.float64),
                torch.tensor([0.2], dtype=torch.float64),
            ),
            id="helical",
        ),
    ]
)
def missing_joint_pair(request):
    """Build the same one-body fixed-base model in BR and Pinocchio."""
    br_joint_factory, pin_joint_factory, q, v = request.param
    mass = 1.2
    com = torch.tensor([0.1, -0.05, 0.2], dtype=torch.float64)
    inertia = torch.diag(torch.tensor([0.04, 0.05, 0.06], dtype=torch.float64))

    builder = ModelBuilder("missing_joint_parity")
    base = builder.add_body("base", mass=0.0)
    tip = builder.add_body("tip", mass=mass, com=com, inertia=inertia)
    builder.add_joint(
        "joint",
        kind=br_joint_factory(),
        parent=base,
        child=tip,
    )
    br_model = build_model(builder.finalize()).to(dtype=torch.float64)

    pin_model = pin.Model()
    pin_joint_id = pin_model.addJoint(
        0,
        pin_joint_factory(),
        pin.SE3.Identity(),
        "joint",
    )
    pin_model.appendBodyToJoint(
        pin_joint_id,
        pin.Inertia(mass, com.numpy(), inertia.numpy()),
        pin.SE3.Identity(),
    )
    return br_model, pin_model, pin_model.createData(), pin_joint_id, q, v


def test_missing_joint_fk_matches_pinocchio(missing_joint_pair):
    br_model, pin_model, pin_data, pin_joint_id, q, _ = missing_joint_pair
    br_data = br.forward_kinematics(br_model, q)
    pin.forwardKinematics(pin_model, pin_data, q.numpy())

    pose_br = br_data.joint_pose_world[2]
    np.testing.assert_allclose(
        pose_br[:3].numpy(),
        np.asarray(pin_data.oMi[pin_joint_id].translation),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        rot_matrix_from_pose(pose_br),
        np.asarray(pin_data.oMi[pin_joint_id].rotation),
        atol=2e-6,
    )


def test_missing_joint_rnea_matches_pinocchio(missing_joint_pair):
    br_model, pin_model, pin_data, _, q, v = missing_joint_pair
    acceleration = (
        torch.arange(
            1,
            br_model.nv + 1,
            dtype=torch.float64,
        )
        * 0.05
    )

    tau_br = (
        br.rnea(
            br_model,
            br_model.create_data(),
            q,
            v,
            acceleration,
        )
        .detach()
        .numpy()
    )
    tau_pin = np.asarray(
        pin.rnea(
            pin_model,
            pin_data,
            q.numpy(),
            v.numpy(),
            acceleration.numpy(),
        )
    )
    np.testing.assert_allclose(tau_br, tau_pin, atol=2e-6, rtol=1e-6)


def test_missing_joint_integrate_difference_roundtrip(missing_joint_pair):
    """Each local manifold recovers a small tangent after retraction."""
    br_model, _, _, _, q, v = missing_joint_pair
    step = v * 0.1
    q_next = br_model.integrate(q, step)
    recovered = br_model.difference(q, q_next)
    torch.testing.assert_close(recovered, step, rtol=1e-12, atol=1e-12)
