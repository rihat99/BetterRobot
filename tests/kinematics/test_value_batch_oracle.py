"""Batched model values ``I...`` broadcast against the configuration batch ``B...`` (record 0028, C3).

The oracle is the scalar loop: one unbatched model per identity, one configuration per row, compared with the
batched pass at every index of ``broadcast(B..., I...)``.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import ShapeError
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics import forward_kinematics, frame_jacobian


def _pose(x: float = 0.0, y: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, y, 0.0, 0.0, 0.0, 0.0, 1.0])


def _model():
    builder = ModelBuilder("value_batch")
    builder.add_body("base", mass=1.0, inertia=torch.eye(3) * 0.1)
    builder.add_body("link", mass=2.0, inertia=torch.eye(3) * 0.2)
    builder.add_body("tip_link", mass=0.5, inertia=torch.eye(3) * 0.05)
    builder.add_revolute("shoulder", axis=(0.0, 0.0, 1.0), parent="base", child="link", origin=_pose(0.5), lower=-2.0, upper=2.0)
    builder.add_revolute("elbow", axis=(0.0, 1.0, 0.0), parent="link", child="tip_link", origin=_pose(0.4), lower=-2.0, upper=2.0)
    builder.add_frame("tip", parent="tip_link", placement=_pose(0.3))
    return build_model(builder.finalize(), dtype=torch.float64)


def _identities(model, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint and frame placements ``(count, njoints, 7)`` and ``(count, nframes, 7)``, one offset per identity."""
    offsets = torch.linspace(0.01, 0.05, count, dtype=torch.float64)
    placements = model.joint_placements.expand(count, -1, -1).clone()
    frames = model.frame_placements.expand(count, -1, -1).clone()
    placements[:, 2, 0] += offsets
    frames[:, -1, 1] += offsets * 0.5
    return placements, frames


def test_identity_batch_broadcasts_against_the_configuration_batch(value_batch_loop_oracle) -> None:
    """``I = (2,)`` against ``B = (3, 1)`` gives the state batch ``(3, 2)``; every index matches the scalar loop."""
    model = _model()
    placements, frames = _identities(model, 2)
    batched = model.with_values(joint_placements=placements, frame_placements=frames)
    generator = torch.Generator().manual_seed(0)
    q = torch.rand(3, 1, model.nq, generator=generator, dtype=torch.float64) - 0.5

    def batched_call():
        state = forward_kinematics(batched, q)
        assert state.q.shape == (3, 2, model.nq)
        assert state.frame_poses.shape == (3, 2, model.nframes, 7)
        return state.joint_poses, state.frame_poses, frame_jacobian(batched, state, frame="tip")

    def loop_call(index: tuple[int, ...]):
        row, identity = index
        scalar = model.with_values(joint_placements=placements[identity], frame_placements=frames[identity])
        state = forward_kinematics(scalar, q[row, 0])
        return state.joint_poses, state.frame_poses, frame_jacobian(scalar, state, frame="tip")

    value_batch_loop_oracle(batched_call, loop_call, execution_batch_shape=(3, 2), rtol=1e-12, atol=1e-12)


def test_non_broadcastable_batches_raise() -> None:
    model = _model()
    placements, _ = _identities(model, 2)
    batched = model.with_values(joint_placements=placements)
    q = model.q_neutral.expand(3, -1)
    with pytest.raises(ShapeError, match=r"cannot broadcast q batch \(3,\) with joint_placements batch \(2,\)"):
        forward_kinematics(batched, q)
