"""Forward kinematics parity — BetterRobot vs Pinocchio."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import better_robot as br

from .conftest import rot_matrix_from_pose, sample_panda_q

pin = pytest.importorskip("pinocchio")


@pytest.mark.parametrize("i", range(8))
def test_fk_panda_translation_matches(panda_both, i):
    br_model, pin_model, pin_data, frame_map = panda_both
    qs = sample_panda_q(n=8)
    q = qs[i]

    # BetterRobot FK
    data = br.forward_kinematics(br_model, q, compute_frames=True)

    # Pinocchio FK
    q_pin = q.detach().cpu().double().numpy()
    pin.forwardKinematics(pin_model, pin_data, q_pin)
    pin.updateFramePlacements(pin_model, pin_data)

    # Compare translations for every mapped frame
    for br_name, pin_fid in frame_map.items():
        br_fid = br_model.frame_id(br_name)
        t_br = data.frame_pose_world[br_fid, :3].detach().cpu().double().numpy()
        t_pin = np.asarray(pin_data.oMf[pin_fid].translation)
        # URDF parser loads placements through np.float32 (see io/parsers/urdf.py);
        # chain accumulation adds another ~1e-7 on top of fp32 machine epsilon.
        np.testing.assert_allclose(t_br, t_pin, atol=2e-6, err_msg=f"frame {br_name}")


@pytest.mark.parametrize("i", range(8))
def test_fk_panda_rotation_matches(panda_both, i):
    br_model, pin_model, pin_data, frame_map = panda_both
    qs = sample_panda_q(n=8)
    q = qs[i]

    data = br.forward_kinematics(br_model, q, compute_frames=True)
    q_pin = q.detach().cpu().double().numpy()
    pin.forwardKinematics(pin_model, pin_data, q_pin)
    pin.updateFramePlacements(pin_model, pin_data)

    for br_name, pin_fid in frame_map.items():
        br_fid = br_model.frame_id(br_name)
        R_br = rot_matrix_from_pose(data.frame_pose_world[br_fid])
        R_pin = np.asarray(pin_data.oMf[pin_fid].rotation)
        np.testing.assert_allclose(R_br, R_pin, atol=2e-6, err_msg=f"frame {br_name}")
