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
    state = br.forward_kinematics_state(br_model, q)

    # Pinocchio FK
    q_pin = q.detach().cpu().double().numpy()
    pin.forwardKinematics(pin_model, pin_data, q_pin)
    pin.updateFramePlacements(pin_model, pin_data)

    # Compare translations for every mapped frame
    for br_name, pin_fid in frame_map.items():
        br_fid = br_model.frame_id(br_name)
        t_br = state.frame_pose[br_fid, :3].detach().cpu().double().numpy()
        t_pin = np.asarray(pin_data.oMf[pin_fid].translation)
        np.testing.assert_allclose(t_br, t_pin, atol=1e-12, err_msg=f"frame {br_name}")


@pytest.mark.parametrize("i", range(8))
def test_fk_panda_rotation_matches(panda_both, i):
    br_model, pin_model, pin_data, frame_map = panda_both
    qs = sample_panda_q(n=8)
    q = qs[i]

    state = br.forward_kinematics_state(br_model, q)
    q_pin = q.detach().cpu().double().numpy()
    pin.forwardKinematics(pin_model, pin_data, q_pin)
    pin.updateFramePlacements(pin_model, pin_data)

    for br_name, pin_fid in frame_map.items():
        br_fid = br_model.frame_id(br_name)
        R_br = rot_matrix_from_pose(state.frame_pose[br_fid])
        R_pin = np.asarray(pin_data.oMf[pin_fid].rotation)
        np.testing.assert_allclose(R_br, R_pin, atol=1e-12, err_msg=f"frame {br_name}")


@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)], ids=["q1", "q_multi"])
def test_fk_batched_matches_pinocchio(panda_both, batch_shape):
    """One BR batched call matches Pinocchio looped over every q slice."""
    br_model, pin_model, pin_data, frame_map = panda_both
    qs = sample_panda_q(n=int(np.prod(batch_shape)), seed=20).reshape(
        *batch_shape,
        br_model.nq,
    )

    state = br.forward_kinematics_state(br_model, qs)

    for batch_index in np.ndindex(batch_shape):
        q_pin = qs[batch_index].detach().cpu().numpy()
        pin.forwardKinematics(pin_model, pin_data, q_pin)
        pin.updateFramePlacements(pin_model, pin_data)

        for br_name, pin_fid in frame_map.items():
            br_fid = br_model.frame_id(br_name)
            pose_br = state.frame_pose[batch_index + (br_fid,)]
            np.testing.assert_allclose(
                pose_br[:3].detach().cpu().numpy(),
                np.asarray(pin_data.oMf[pin_fid].translation),
                atol=2e-6,
                err_msg=f"batch {batch_index}, frame {br_name}",
            )
            np.testing.assert_allclose(
                rot_matrix_from_pose(pose_br),
                np.asarray(pin_data.oMf[pin_fid].rotation),
                atol=2e-6,
                err_msg=f"batch {batch_index}, frame {br_name}",
            )


def test_fk_panda_fp32_matches_pinocchio(panda_both):
    """Panda FK retains its existing 2e-6 URDF parity band in fp32."""
    br_model_fp64, pin_model, pin_data, frame_map = panda_both
    br_model = br_model_fp64.to(dtype=torch.float32)
    qs = sample_panda_q(n=4, seed=40).to(torch.float32)

    for q in qs:
        state = br.forward_kinematics_state(br_model, q)
        pin.forwardKinematics(pin_model, pin_data, q.double().numpy())
        pin.updateFramePlacements(pin_model, pin_data)

        for br_name, pin_fid in frame_map.items():
            br_fid = br_model.frame_id(br_name)
            pose_br = state.frame_pose[br_fid]
            np.testing.assert_allclose(
                pose_br[:3].double().numpy(),
                np.asarray(pin_data.oMf[pin_fid].translation),
                atol=2e-6,
                err_msg=f"frame {br_name}",
            )
            np.testing.assert_allclose(
                rot_matrix_from_pose(pose_br),
                np.asarray(pin_data.oMf[pin_fid].rotation),
                atol=2e-6,
                err_msg=f"frame {br_name}",
            )
