"""``kinematics.forward`` / ``frame_jacobian`` agree with the ``Data``-based API and with autograd."""

import pytest
import torch

import better_robot as br
from better_robot import kinematics


def test_state_matches_data_api_and_autograd():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description

    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    q = torch.stack([model.random_configuration(generator) for _ in range(4)])
    frame = "body_panda_hand"

    state = kinematics.forward(model, q)
    data = br.forward_kinematics(model, q, compute_frames=True)
    torch.testing.assert_close(state.joint_pose, data.joint_pose_world)
    torch.testing.assert_close(state.frame_pose, data.frame_pose_world)
    assert state.frame_rotation.shape == (4, model.nframes, 3, 3)
    with pytest.raises(AttributeError):
        state.q = q

    jacobian = kinematics.frame_jacobian(model, state, frame)
    torch.testing.assert_close(jacobian, br.get_frame_jacobian(model, data, model.frame_id(frame)))
    frame_id = model.frame_id(frame)
    position_jacobian = torch.autograd.functional.jacobian(
        lambda x: kinematics.forward(model, x).frame_translation[:, frame_id].sum(0), q
    )  # (3, 4, nq); Panda has nq == nv
    # The URDF parser rounds placements to float32, so their quaternions are off unit norm by ~1e-7;
    # the analytic Jacobian and autograd through FK agree only to that level.
    torch.testing.assert_close(jacobian[:, :3, :], position_jacobian.permute(1, 0, 2), atol=1e-6, rtol=0.0)
