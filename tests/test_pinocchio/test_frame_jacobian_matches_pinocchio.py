"""Frame Jacobian parity for all three reference frames.

This is the regression test that anchors the ``reference=`` parameter in
``get_frame_jacobian`` against Pinocchio's canonical convention.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot.kinematics.jacobian import get_frame_jacobian

from .conftest import sample_panda_q

pin = pytest.importorskip("pinocchio")


# Test against the end-effector frame (panda_hand — the most natural choice
# for manipulation residuals).
TARGET_FRAME = "body_panda_hand"
TARGET_PIN_FRAME = "panda_hand"


def _fk_both(br_model, pin_model, pin_data, q):
    data = br.forward_kinematics(br_model, q, compute_frames=True)
    q_pin = q.detach().cpu().double().numpy()
    pin.forwardKinematics(pin_model, pin_data, q_pin)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.computeJointJacobians(pin_model, pin_data, q_pin)
    return data


@pytest.mark.parametrize("i", range(4))
def test_frame_jacobian_local_world_aligned(panda_both, i):
    br_model, pin_model, pin_data, frame_map = panda_both
    qs = sample_panda_q(n=4, seed=10)
    q = qs[i]

    data = _fk_both(br_model, pin_model, pin_data, q)
    br_fid = br_model.frame_id(TARGET_FRAME)
    pin_fid = pin_model.getFrameId(TARGET_PIN_FRAME)

    J_br = get_frame_jacobian(
        br_model, data, br_fid, reference="local_world_aligned"
    ).detach().cpu().double().numpy()
    J_pin = pin.getFrameJacobian(
        pin_model, pin_data, pin_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    # URDF parser loads placements as fp32 (io/parsers/urdf.py) → ~5e-7 drift.
    np.testing.assert_allclose(J_br, J_pin, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("i", range(4))
def test_frame_jacobian_world(panda_both, i):
    br_model, pin_model, pin_data, frame_map = panda_both
    qs = sample_panda_q(n=4, seed=11)
    q = qs[i]

    data = _fk_both(br_model, pin_model, pin_data, q)
    br_fid = br_model.frame_id(TARGET_FRAME)
    pin_fid = pin_model.getFrameId(TARGET_PIN_FRAME)

    J_br = get_frame_jacobian(
        br_model, data, br_fid, reference="world"
    ).detach().cpu().double().numpy()
    J_pin = pin.getFrameJacobian(
        pin_model, pin_data, pin_fid, pin.ReferenceFrame.WORLD
    )

    # URDF parser loads placements as fp32 (io/parsers/urdf.py) → ~5e-7 drift.
    np.testing.assert_allclose(J_br, J_pin, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("i", range(4))
def test_frame_jacobian_local(panda_both, i):
    br_model, pin_model, pin_data, frame_map = panda_both
    qs = sample_panda_q(n=4, seed=12)
    q = qs[i]

    data = _fk_both(br_model, pin_model, pin_data, q)
    br_fid = br_model.frame_id(TARGET_FRAME)
    pin_fid = pin_model.getFrameId(TARGET_PIN_FRAME)

    J_br = get_frame_jacobian(
        br_model, data, br_fid, reference="local"
    ).detach().cpu().double().numpy()
    J_pin = pin.getFrameJacobian(
        pin_model, pin_data, pin_fid, pin.ReferenceFrame.LOCAL
    )

    # URDF parser loads placements as fp32 (io/parsers/urdf.py) → ~5e-7 drift.
    np.testing.assert_allclose(J_br, J_pin, atol=2e-6, rtol=1e-5)
