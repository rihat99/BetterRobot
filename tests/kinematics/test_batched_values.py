"""Value-batched FK, frames, Jacobians, and broadcast-contract tests."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import ShapeError
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics import get_frame_jacobian
from better_robot.kinematics.forward import forward_kinematics


def _pose(x: float = 0.0, y: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, y, 0.0, 0.0, 0.0, 0.0, 1.0])


def _model(dtype: torch.dtype = torch.float64):
    builder = ModelBuilder("batched_values_fk")
    builder.add_body("base", mass=1.0, inertia=torch.eye(3) * 0.1)
    builder.add_body("link", mass=2.0, inertia=torch.eye(3) * 0.2)
    builder.add_revolute_z(
        "joint",
        parent="base",
        child="link",
        origin=_pose(0.5),
        lower=-2.0,
        upper=2.0,
    )
    builder.add_frame("tip", parent_body="link", placement=_pose(0.3))
    return build_model(builder.finalize(), dtype=dtype)


def _value_batch(model, batch_shape: tuple[int, ...]):
    placements = model.values.joint_placements.expand(*batch_shape, model.njoints, 7).clone()
    frames = model.values.frame_placements.expand(*batch_shape, model.nframes, 7).clone()
    offsets = torch.linspace(
        0.01,
        0.03,
        int(torch.tensor(batch_shape).prod()),
        dtype=placements.dtype,
    ).reshape(*batch_shape)
    placements[..., 2, 0] += offsets
    frames[..., -1, 1] += offsets * 0.5
    return placements, frames


@pytest.mark.parametrize("batch_shape", ((3,), (2, 3)))
def test_value_batched_fk_matches_scalar_loop(
    value_batch_loop_oracle,
    batch_shape: tuple[int, ...],
) -> None:
    model = _model()
    placements, frames = _value_batch(model, batch_shape)
    rebound = model.with_values(
        joint_placements=placements,
        frame_placements=frames,
    )

    def batched_call():
        data = forward_kinematics(rebound, model.q_neutral, compute_frames=True)
        assert data.batch_shape == batch_shape
        assert data.frame_pose_world is not None
        return data.joint_pose_world, data.joint_pose_local, data.frame_pose_world

    def loop_call(index: tuple[int, ...]):
        scalar = model.with_values(
            joint_placements=placements[index],
            frame_placements=frames[index],
        )
        data = forward_kinematics(scalar, model.q_neutral, compute_frames=True)
        return data.joint_pose_world, data.joint_pose_local, data.frame_pose_world

    value_batch_loop_oracle(
        batched_call,
        loop_call,
        execution_batch_shape=batch_shape,
        rtol=1e-11,
        atol=1e-12,
    )


def test_person_time_broadcast_requires_explicit_singleton_axis() -> None:
    model = _model()
    q = model.q_neutral.expand(2, 3, -1).clone()
    placements, _ = _value_batch(model, (2,))

    with pytest.raises(
        ShapeError,
        match=r"cannot broadcast q batch \(2, 3\) with joint_placements batch \(2,\)",
    ):
        forward_kinematics(model.with_values(joint_placements=placements), q)

    rebound = model.with_values(joint_placements=placements[:, None])
    data = forward_kinematics(rebound, q)
    assert data.q.shape == (2, 3, model.nq)
    assert data.joint_pose_world.shape == (2, 3, model.njoints, 7)


def test_frame_only_batch_expands_data_and_jacobian_uses_frame_table() -> None:
    model = _model()
    frames = model.values.frame_placements.repeat(2, 1, 1)
    frames[:, -1, 0] += torch.tensor([0.1, 0.4], dtype=frames.dtype)
    rebound = model.with_values(frame_placements=frames)
    data = forward_kinematics(rebound, model.q_neutral, compute_frames=True)

    assert data.batch_shape == (2,)
    assert data.q.shape == (2, model.nq)
    assert data.frame_pose_world is not None
    assert not torch.allclose(
        data.frame_pose_world[0, -1],
        data.frame_pose_world[1, -1],
    )
    jacobian = get_frame_jacobian(rebound, data, rebound.frame_id("tip"))
    assert jacobian.shape == (2, 6, model.nv)
    assert not torch.allclose(jacobian[0], jacobian[1])


def test_batched_joint_placement_autograd_reaches_leaf() -> None:
    model = _model()
    delta = torch.zeros(
        2,
        model.njoints,
        7,
        dtype=model.values.joint_placements.dtype,
        requires_grad=True,
    )
    placements = model.values.joint_placements + delta
    rebound = model.with_values(joint_placements=placements)
    data = forward_kinematics(rebound, model.q_neutral)
    loss = data.joint_pose_world[..., 2, :3].square().sum()
    loss.backward()

    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
    assert delta.grad.abs().sum() > 0
