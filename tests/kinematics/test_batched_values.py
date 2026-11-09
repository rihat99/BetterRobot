"""Value-batched FK, frames, Jacobians, and broadcast-contract tests."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import ShapeError
from better_robot.io.build_model import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.kinematics import forward_kinematics, frame_jacobian


def _pose(x: float = 0.0, y: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, y, 0.0, 0.0, 0.0, 0.0, 1.0])


def _model(dtype: torch.dtype = torch.float64):
    builder = ModelBuilder("batched_values_fk")
    builder.add_body("base", mass=1.0, inertia=torch.eye(3) * 0.1)
    builder.add_body("link", mass=2.0, inertia=torch.eye(3) * 0.2)
    builder.add_revolute(
        "joint",
        axis=(0.0, 0.0, 1.0),
        parent="base",
        child="link",
        origin=_pose(0.5),
        lower=-2.0,
        upper=2.0,
    )
    builder.add_frame("tip", parent="link", placement=_pose(0.3))
    return build_model(builder.finalize(), dtype=dtype)


def _value_batch(model, batch_shape: tuple[int, ...]):
    placements = model.joint_placements.expand(*batch_shape, model.njoints, 7).clone()
    frames = model.frame_placements.expand(*batch_shape, model.nframes, 7).clone()
    offsets = torch.linspace(
        0.01,
        0.03,
        int(torch.tensor(batch_shape).prod()),
        dtype=placements.dtype,
    ).reshape(*batch_shape)
    placements[..., 2, 0] += offsets
    frames[..., -1, 1] += offsets * 0.5
    return placements, frames


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
    state = forward_kinematics(rebound, q)
    assert state.q.shape == (2, 3, model.nq)
    assert state.joint_poses.shape == (2, 3, model.njoints, 7)


def test_frame_only_batch_expands_data_and_jacobian_uses_frame_table() -> None:
    model = _model()
    frames = model.frame_placements.repeat(2, 1, 1)
    frames[:, -1, 0] += torch.tensor([0.1, 0.4], dtype=frames.dtype)
    rebound = model.with_values(frame_placements=frames)
    state = forward_kinematics(rebound, model.q_neutral)

    assert state.q.shape[:-1] == (2,)
    assert state.q.shape == (2, model.nq)
    assert not torch.allclose(
        state.frame_poses[0, -1],
        state.frame_poses[1, -1],
    )
    jacobian = frame_jacobian(rebound, state, frame=rebound.frame_id("tip"))
    assert jacobian.shape == (2, 6, model.nv)
    assert not torch.allclose(jacobian[0], jacobian[1])


def test_batched_joint_placement_autograd_reaches_leaf() -> None:
    model = _model()
    delta = torch.zeros(
        2,
        model.njoints,
        7,
        dtype=model.joint_placements.dtype,
        requires_grad=True,
    )
    placements = model.joint_placements + delta
    rebound = model.with_values(joint_placements=placements)
    state = forward_kinematics(rebound, model.q_neutral)
    loss = state.joint_poses[..., 2, :3].square().sum()
    loss.backward()

    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
    assert delta.grad.abs().sum() > 0
