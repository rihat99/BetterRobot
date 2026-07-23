"""Joint Jacobian parity for all three reference frames.

This is the regression test that anchors the ``reference=`` parameter in
``get_joint_jacobian`` against Pinocchio's canonical convention, including
``nq != nv`` joints (free-flyer, spherical) where a wrong frame convention
would hide on Panda.

BetterRobot keeps the fixed joints that Pinocchio drops, so joint indices do
not align 1-to-1; joints are resolved by name in both models.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot.kinematics.jacobian import get_joint_jacobian

from .conftest import _build_spherical_chain, sample_panda_q

pin = pytest.importorskip("pinocchio")


_REFERENCES = [
    ("world", pin.ReferenceFrame.WORLD),
    ("local_world_aligned", pin.ReferenceFrame.LOCAL_WORLD_ALIGNED),
    ("local", pin.ReferenceFrame.LOCAL),
]

# Joints present in both the BetterRobot and Pinocchio Panda models, spanning
# base, mid-chain, and the wrist tip.
PANDA_TARGET_JOINTS = ["panda_joint1", "panda_joint4", "panda_joint7"]


def _fk_both(br_model, pin_model, pin_data, q):
    data = br.forward_kinematics(br_model, q)
    q_pin = q.detach().cpu().double().numpy()
    pin.forwardKinematics(pin_model, pin_data, q_pin)
    pin.computeJointJacobians(pin_model, pin_data, q_pin)
    return data


@pytest.mark.parametrize("joint_name", PANDA_TARGET_JOINTS)
@pytest.mark.parametrize(
    ("reference", "pin_reference"),
    _REFERENCES,
)
def test_joint_jacobian_matches_pinocchio(panda_both, joint_name, reference, pin_reference):
    """Single-joint Panda Jacobians agree with Pinocchio across configs and frames."""
    br_model, pin_model, pin_data, _ = panda_both
    br_jid = br_model.joint_id(joint_name)
    pin_jid = pin_model.getJointId(joint_name)

    for q in sample_panda_q(n=3, seed=13):
        data = _fk_both(br_model, pin_model, pin_data, q)
        J_br = (
            get_joint_jacobian(br_model, data, br_jid, reference=reference)
            .detach()
            .cpu()
            .double()
            .numpy()
        )
        J_pin = pin.getJointJacobian(pin_model, pin_data, pin_jid, pin_reference)
        np.testing.assert_allclose(
            J_br, J_pin, atol=2e-6, rtol=1e-5, err_msg=f"joint {joint_name}, reference {reference}"
        )


@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)], ids=["q1", "q_multi"])
@pytest.mark.parametrize(
    ("reference", "pin_reference"),
    _REFERENCES,
)
def test_joint_jacobian_batched_matches_pinocchio(panda_both, batch_shape, reference, pin_reference):
    """Batched joint Jacobians agree with a Pinocchio loop."""
    br_model, pin_model, pin_data, _ = panda_both
    qs = sample_panda_q(n=int(np.prod(batch_shape)), seed=31).reshape(*batch_shape, br_model.nq)

    data = br.forward_kinematics(br_model, qs)
    br_jid = br_model.joint_id("panda_joint7")
    pin_jid = pin_model.getJointId("panda_joint7")
    J_br = (
        get_joint_jacobian(br_model, data, br_jid, reference=reference).detach().cpu().numpy()
    )

    for batch_index in np.ndindex(batch_shape):
        q_pin = qs[batch_index].detach().cpu().numpy()
        pin.forwardKinematics(pin_model, pin_data, q_pin)
        pin.computeJointJacobians(pin_model, pin_data, q_pin)
        J_pin = pin.getJointJacobian(pin_model, pin_data, pin_jid, pin_reference)
        np.testing.assert_allclose(
            J_br[batch_index],
            J_pin,
            atol=2e-6,
            rtol=1e-5,
            err_msg=f"batch {batch_index}, reference {reference}",
        )


def _sample_g1_q(br_model, seed):
    """Random fp64 q for G1 with a normalized free-flyer quaternion."""
    rng = torch.Generator().manual_seed(seed)
    q = torch.zeros(br_model.nq, dtype=torch.float64)
    q[0:3] = torch.rand(3, generator=rng, dtype=torch.float64) * 0.5 - 0.25
    quat = torch.randn(4, generator=rng, dtype=torch.float64)
    q[3:7] = quat / quat.norm()
    if br_model.nq > 7:
        q[7:] = torch.rand(br_model.nq - 7, generator=rng, dtype=torch.float64) * 0.6 - 0.3
    return q


@pytest.mark.parametrize(
    ("reference", "pin_reference"),
    _REFERENCES,
)
def test_joint_jacobian_free_flyer_matches_pinocchio(g1_both, reference, pin_reference):
    """Free-flyer (nq != nv) joint Jacobians agree with Pinocchio."""
    br_model, pin_model, pin_data = g1_both
    q = _sample_g1_q(br_model, seed=7)

    data = br.forward_kinematics(br_model, q)
    q_pin = q.detach().cpu().numpy()
    pin.forwardKinematics(pin_model, pin_data, q_pin)
    pin.computeJointJacobians(pin_model, pin_data, q_pin)

    for joint_name in ["root_joint", "left_hip_yaw_joint", "right_wrist_yaw_joint"]:
        br_jid = br_model.joint_id(joint_name)
        pin_jid = pin_model.getJointId(joint_name)
        J_br = get_joint_jacobian(br_model, data, br_jid, reference=reference).detach().cpu().numpy()
        J_pin = pin.getJointJacobian(pin_model, pin_data, pin_jid, pin_reference)
        np.testing.assert_allclose(
            J_br, J_pin, atol=2e-6, rtol=1e-5, err_msg=f"joint {joint_name}, reference {reference}"
        )


def _sample_spherical_q(seed):
    """Random fp64 q=[quat, angle] for the spherical + revolute chain."""
    rng = torch.Generator().manual_seed(seed)
    quat = torch.randn(4, generator=rng, dtype=torch.float64)
    quat = quat / quat.norm()
    angle = torch.rand(1, generator=rng, dtype=torch.float64) * 2.0 - 1.0
    return torch.cat([quat, angle])


def test_joint_jacobian_spherical_matches_pinocchio():
    """Spherical joint (nq != nv) Jacobians agree with a hand-built Pinocchio model."""
    br_model, pin_model, pin_data = _build_spherical_chain()
    q = _sample_spherical_q(seed=11)

    data = br.forward_kinematics(br_model, q)
    q_pin = q.detach().cpu().numpy()
    pin.forwardKinematics(pin_model, pin_data, q_pin)
    pin.computeJointJacobians(pin_model, pin_data, q_pin)

    for joint_name in ["j_sph", "j_rz"]:
        br_jid = br_model.joint_id(joint_name)
        pin_jid = pin_model.getJointId(joint_name)
        for reference, pin_reference in _REFERENCES:
            J_br = (
                get_joint_jacobian(br_model, data, br_jid, reference=reference)
                .detach()
                .cpu()
                .numpy()
            )
            J_pin = pin.getJointJacobian(pin_model, pin_data, pin_jid, pin_reference)
            # Programmatic, fp64 end-to-end — no URDF fp32 round-trip.
            np.testing.assert_allclose(
                J_br, J_pin, atol=2e-6, rtol=1e-5, err_msg=f"joint {joint_name}, reference {reference}"
            )
