"""``kinematics.forward_kinematics`` / ``frame_jacobian`` agree with autograd and with each other."""

import pytest
import torch

import better_robot as br
from better_robot import dynamics, kinematics
from better_robot.io import ModelBuilder, build_model
from better_robot.lie import so3
from tests.support.robots import panda_urdf


def test_state_matches_data_api_and_autograd():
    model = br.load(panda_urdf(), dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    q = torch.stack([model.random_configuration(generator=generator) for _ in range(4)])
    frame = "body_panda_hand"

    state = kinematics.forward_kinematics(model, q)
    state = br.forward_kinematics(model, q)
    torch.testing.assert_close(state.joint_poses, state.joint_poses)
    torch.testing.assert_close(state.frame_translations, state.frame_poses[..., :3], atol=1e-14, rtol=0.0)
    torch.testing.assert_close(state.frame_rotations, so3.to_matrix(state.frame_poses[..., 3:]), atol=1e-14, rtol=0.0)
    hand = model.frame_id(frame)
    torch.testing.assert_close(state.frame_pose_of(hand), state.frame_poses[:, hand])
    assert state.frame_rotations.shape == (4, model.nframes, 3, 3)
    with pytest.raises(AttributeError):
        state.q = q

    jacobian = kinematics.frame_jacobian(model, state, frame=frame)
    torch.testing.assert_close(jacobian, br.frame_jacobian(model, state, frame=model.frame_id(frame)))
    for reference in ("world", "local"):
        torch.testing.assert_close(
            kinematics.frame_jacobian(model, state, frame=frame, reference=reference),
            br.frame_jacobian(model, state, frame=model.frame_id(frame), reference=reference),
        )
    frame_id = model.frame_id(frame)
    position_jacobian = torch.autograd.functional.jacobian(
        lambda x: kinematics.forward_kinematics(model, x).frame_translations[:, frame_id].sum(0), q
    )  # (3, 4, nq); Panda has nq == nv
    torch.testing.assert_close(jacobian[:, :3, :], position_jacobian.permute(1, 0, 2), atol=1e-12, rtol=0.0)


def test_state_is_differentiable_in_model_values_and_under_torch_func():
    model = br.load(panda_urdf(), dtype=torch.float64)
    q = model.q_neutral.clamp(model.lower_position_limit, model.upper_position_limit).expand(2, -1)
    hand = model.frame_id("body_panda_hand")

    def hand_position(placements: torch.Tensor) -> torch.Tensor:
        moved = model.with_values(joint_placements=placements)
        return kinematics.forward_kinematics(moved, q).frame_translations[:, hand].sum(0)

    placements = model.joint_placements.clone()
    by_func = torch.func.jacrev(hand_position)(placements)
    by_autograd = torch.autograd.functional.jacobian(hand_position, placements)
    torch.testing.assert_close(by_func, by_autograd)
    assert float(by_func[:, :, :3].abs().max()) > 0.5  # moving a joint origin moves the hand

    by_q = torch.func.jacrev(lambda x: kinematics.forward_kinematics(model, x).frame_translations[:, hand].sum(0))(q)
    jacobian = kinematics.frame_jacobian(model, kinematics.forward_kinematics(model, q), frame=hand)
    torch.testing.assert_close(by_q.permute(1, 0, 2), jacobian[:, :3, :], atol=1e-12, rtol=0.0)

    # Batched model values: one set of placements and frames per configuration.
    scale = torch.tensor([1.0, 1.3], dtype=torch.float64)[:, None, None]
    batched = model.with_values(
        joint_placements=torch.cat(
            [model.joint_placements[None, :, :3] * scale, model.joint_placements[None, :, 3:].expand(2, -1, -1)], -1
        ),
        frame_placements=model.frame_placements[None].expand(2, -1, -1),
    )
    state = kinematics.forward_kinematics(batched, q)
    state = br.forward_kinematics(batched, q)
    torch.testing.assert_close(state.joint_poses, state.joint_poses)
    torch.testing.assert_close(state.frame_translations, state.frame_poses[..., :3], atol=1e-14, rtol=0.0)
    assert float((state.frame_translations[1] - state.frame_translations[0]).abs().max()) > 0.05


def test_state_review_regressions():
    """An unaligned joint axis, a moved placement in a new model after a call, first call in inference mode."""
    builder = ModelBuilder("axis")
    base = builder.add_body("base", mass=1.0)
    link = builder.add_body("link", mass=1.0)
    builder.add_revolute(
        "joint",
        parent=base,
        child=link,
        axis=torch.tensor([0.6, 0.8, 0.0]),
        origin=torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
    )
    model = build_model(builder.finalize(), dtype=torch.float64)
    q = torch.tensor([[0.7], [-2.1]], dtype=torch.float64)
    with torch.inference_mode():
        kinematics.forward_kinematics(model, q)
    state = kinematics.forward_kinematics(model, q.clone().requires_grad_(True))
    state.joint_translations.sum().backward()
    state = br.forward_kinematics(model, q)
    rotation = state.joint_rotations
    torch.testing.assert_close(
        rotation @ rotation.mT, torch.eye(3, dtype=torch.float64).expand_as(rotation), atol=1e-14, rtol=0.0
    )
    torch.testing.assert_close(rotation, so3.to_matrix(state.joint_poses[..., 3:]), atol=1e-14, rtol=0.0)

    placements = model.joint_placements.clone()
    placements[-1, 0] += 1.0
    moved = kinematics.forward_kinematics(model.with_values(joint_placements=placements), q)
    torch.testing.assert_close(moved.joint_translations[:, -1, 0], state.joint_translations[:, -1, 0].detach() + 1.0)


def test_empty_batch_flows_through_fk_and_dynamics():
    """A zero-size batch must not hit ``reshape(0, -1)``, which cannot infer the trailing size."""

    model = br.load(panda_urdf(), free_flyer=True, dtype=torch.float64)
    state = br.forward_kinematics(model, model.q_neutral.expand(0, -1))
    v = torch.zeros(0, model.nv, dtype=torch.float64)
    assert state.frame_poses.shape == (0, model.nframes, 7)
    assert dynamics.forward_dynamics(model, state, v=v, tau=v).shape == (0, model.nv)
    assert dynamics.inverse_dynamics(model, state, v=v, a=v).shape == (0, model.nv)
