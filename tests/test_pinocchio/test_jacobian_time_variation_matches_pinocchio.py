"""Jacobian time-variation (J̇) parity for all three reference frames.

Anchors ``get_joint_jacobian_time_variation`` / ``get_frame_jacobian_time_variation``
against Pinocchio's ``computeJointJacobiansTimeVariation`` oracle, including the
``nq != nv`` joints (free-flyer, spherical) where a wrong frame convention or a
dropped transport term would hide on Panda.

BetterRobot keeps the fixed joints Pinocchio merges, so joints/frames are
resolved by name in both models (see ``test_joint_jacobian_matches_pinocchio``).

Tolerances match the static frame-Jacobian file (``atol=2e-6, rtol=1e-5``); the
URDF parser loads placements as fp32 (``io/parsers/urdf.py``), and J̇ carries one
extra ``J × v`` product on top of that ~5e-7 drift — the kept tolerance covers it
without widening.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot.kinematics.jacobian import (
    get_frame_jacobian_time_variation,
    get_joint_jacobian_time_variation,
)

from .conftest import _build_spherical_chain, sample_panda_q, sample_panda_v

pin = pytest.importorskip("pinocchio")


_REFERENCES = [
    ("world", pin.ReferenceFrame.WORLD),
    ("local_world_aligned", pin.ReferenceFrame.LOCAL_WORLD_ALIGNED),
    ("local", pin.ReferenceFrame.LOCAL),
]

PANDA_TARGET_JOINTS = ["panda_joint1", "panda_joint4", "panda_joint7"]
# panda_hand is a fixed-joint operational frame with a nontrivial world pose:
# its origin sits away from the world origin with the whole arm's angular DOF
# above it, so the LWA ``-v_pt × J_ang`` transport term is O(1) here.
PANDA_TARGET_FRAME = ("body_panda_hand", "panda_hand")


def _fk_both(br_model, pin_model, pin_data, q, v):
    """Run BR forward kinematics + set velocity, and the Pinocchio J̇ oracle."""
    data = br.forward_kinematics(br_model, q, compute_frames=True)
    data.v = v
    q_pin = q.detach().cpu().double().numpy()
    v_pin = v.detach().cpu().double().numpy()
    pin.computeJointJacobiansTimeVariation(pin_model, pin_data, q_pin, v_pin)
    pin.updateFramePlacements(pin_model, pin_data)
    return data


@pytest.mark.parametrize("joint_name", PANDA_TARGET_JOINTS)
@pytest.mark.parametrize(("reference", "pin_reference"), _REFERENCES)
def test_joint_time_variation_matches_pinocchio(panda_both, joint_name, reference, pin_reference):
    """Single-joint Panda J̇ agrees with Pinocchio across configs and frames."""
    br_model, pin_model, pin_data, _ = panda_both
    br_jid = br_model.joint_id(joint_name)
    pin_jid = pin_model.getJointId(joint_name)

    for q, v in zip(sample_panda_q(n=3, seed=13), sample_panda_v(n=3, seed=17)):
        data = _fk_both(br_model, pin_model, pin_data, q, v)
        J_br = (
            get_joint_jacobian_time_variation(br_model, data, br_jid, reference=reference)
            .detach()
            .cpu()
            .double()
            .numpy()
        )
        J_pin = pin.getJointJacobianTimeVariation(pin_model, pin_data, pin_jid, pin_reference)
        np.testing.assert_allclose(
            J_br, J_pin, atol=2e-6, rtol=1e-5, err_msg=f"joint {joint_name}, reference {reference}"
        )


@pytest.mark.parametrize(("reference", "pin_reference"), _REFERENCES)
def test_frame_time_variation_matches_pinocchio(panda_both, reference, pin_reference):
    """Operational-frame Panda J̇ agrees with Pinocchio across configs and frames."""
    br_model, pin_model, pin_data, _ = panda_both
    br_name, pin_name = PANDA_TARGET_FRAME
    br_fid = br_model.frame_id(br_name)
    pin_fid = pin_model.getFrameId(pin_name)

    for q, v in zip(sample_panda_q(n=3, seed=19), sample_panda_v(n=3, seed=23)):
        data = _fk_both(br_model, pin_model, pin_data, q, v)
        J_br = (
            get_frame_jacobian_time_variation(br_model, data, br_fid, reference=reference)
            .detach()
            .cpu()
            .double()
            .numpy()
        )
        J_pin = pin.getFrameJacobianTimeVariation(pin_model, pin_data, pin_fid, pin_reference)
        np.testing.assert_allclose(
            J_br, J_pin, atol=2e-6, rtol=1e-5, err_msg=f"frame {br_name}, reference {reference}"
        )


@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)], ids=["q1", "q_multi"])
@pytest.mark.parametrize(("reference", "pin_reference"), _REFERENCES)
def test_joint_time_variation_batched_matches_pinocchio(panda_both, batch_shape, reference, pin_reference):
    """Batched joint J̇ agrees with a Pinocchio loop."""
    br_model, pin_model, pin_data, _ = panda_both
    count = int(np.prod(batch_shape))
    qs = sample_panda_q(n=count, seed=31).reshape(*batch_shape, br_model.nq)
    vs = sample_panda_v(n=count, seed=37).reshape(*batch_shape, br_model.nv)

    data = br.forward_kinematics(br_model, qs, compute_frames=True)
    data.v = vs
    br_jid = br_model.joint_id("panda_joint7")
    pin_jid = pin_model.getJointId("panda_joint7")
    J_br = (
        get_joint_jacobian_time_variation(br_model, data, br_jid, reference=reference)
        .detach()
        .cpu()
        .numpy()
    )

    for batch_index in np.ndindex(batch_shape):
        q_pin = qs[batch_index].detach().cpu().numpy()
        v_pin = vs[batch_index].detach().cpu().numpy()
        pin.computeJointJacobiansTimeVariation(pin_model, pin_data, q_pin, v_pin)
        J_pin = pin.getJointJacobianTimeVariation(pin_model, pin_data, pin_jid, pin_reference)
        np.testing.assert_allclose(
            J_br[batch_index],
            J_pin,
            atol=2e-6,
            rtol=1e-5,
            err_msg=f"batch {batch_index}, reference {reference}",
        )


def _sample_g1(br_model, seed):
    """Random fp64 (q, v) for G1 with a normalized free-flyer quaternion."""
    rng = torch.Generator().manual_seed(seed)
    q = torch.zeros(br_model.nq, dtype=torch.float64)
    q[0:3] = torch.rand(3, generator=rng, dtype=torch.float64) * 0.5 - 0.25
    quat = torch.randn(4, generator=rng, dtype=torch.float64)
    q[3:7] = quat / quat.norm()
    if br_model.nq > 7:
        q[7:] = torch.rand(br_model.nq - 7, generator=rng, dtype=torch.float64) * 0.6 - 0.3
    v = torch.rand(br_model.nv, generator=rng, dtype=torch.float64) * 1.2 - 0.6
    v = torch.sign(v) * (v.abs() + 0.2)
    return q, v


@pytest.mark.parametrize(("reference", "pin_reference"), _REFERENCES)
def test_joint_time_variation_free_flyer_matches_pinocchio(g1_both, reference, pin_reference):
    """Free-flyer (nq != nv) joint J̇ agrees with Pinocchio."""
    br_model, pin_model, pin_data = g1_both
    q, v = _sample_g1(br_model, seed=7)

    data = br.forward_kinematics(br_model, q)
    data.v = v
    pin.computeJointJacobiansTimeVariation(pin_model, pin_data, q.numpy(), v.numpy())

    for joint_name in ["root_joint", "left_hip_yaw_joint", "right_wrist_yaw_joint"]:
        br_jid = br_model.joint_id(joint_name)
        pin_jid = pin_model.getJointId(joint_name)
        J_br = (
            get_joint_jacobian_time_variation(br_model, data, br_jid, reference=reference)
            .detach()
            .cpu()
            .numpy()
        )
        J_pin = pin.getJointJacobianTimeVariation(pin_model, pin_data, pin_jid, pin_reference)
        np.testing.assert_allclose(
            J_br, J_pin, atol=2e-6, rtol=1e-5, err_msg=f"joint {joint_name}, reference {reference}"
        )


def _sample_spherical(seed):
    """Random fp64 (q=[quat, angle], v=[omega(3), dtheta]) for the chain."""
    rng = torch.Generator().manual_seed(seed)
    quat = torch.randn(4, generator=rng, dtype=torch.float64)
    quat = quat / quat.norm()
    angle = torch.rand(1, generator=rng, dtype=torch.float64) * 2.0 - 1.0
    q = torch.cat([quat, angle])
    v = torch.rand(4, generator=rng, dtype=torch.float64) * 1.2 - 0.6
    v = torch.sign(v) * (v.abs() + 0.2)
    return q, v


def test_joint_time_variation_spherical_matches_pinocchio():
    """Spherical joint (nq != nv) J̇ agrees with a hand-built Pinocchio model."""
    br_model, pin_model, pin_data = _build_spherical_chain()
    q, v = _sample_spherical(seed=11)

    data = br.forward_kinematics(br_model, q)
    data.v = v
    pin.computeJointJacobiansTimeVariation(pin_model, pin_data, q.numpy(), v.numpy())

    for joint_name in ["j_sph", "j_rz"]:
        br_jid = br_model.joint_id(joint_name)
        pin_jid = pin_model.getJointId(joint_name)
        for reference, pin_reference in _REFERENCES:
            J_br = (
                get_joint_jacobian_time_variation(br_model, data, br_jid, reference=reference)
                .detach()
                .cpu()
                .numpy()
            )
            J_pin = pin.getJointJacobianTimeVariation(pin_model, pin_data, pin_jid, pin_reference)
            # Programmatic, fp64 end-to-end — no URDF fp32 round-trip.
            np.testing.assert_allclose(
                J_br, J_pin, atol=2e-6, rtol=1e-5, err_msg=f"joint {joint_name}, reference {reference}"
            )
