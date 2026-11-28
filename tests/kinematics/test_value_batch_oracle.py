"""Batched model values ``I...`` broadcast against the configuration batch ``B...``, and gradients reach them.

The oracle is the scalar loop: one unbatched model per identity, one configuration per row, compared with the
batched pass at every index of ``broadcast(B..., I...)``.
"""

from __future__ import annotations

import re

import pytest
import torch

from better_robot.exceptions import ShapeError
from better_robot.io import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.kinematics import (
    forward_kinematics,
    frame_jacobian,
    frame_jacobian_time_variation,
    joint_jacobian,
    joint_jacobian_time_variation,
    point_jacobian,
)
from better_robot.model import JointRevolute
from tests.support.value_batch import assert_value_batched_matches_loop


def _pose(x: float = 0.0, y: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, y, 0.0, 0.0, 0.0, 0.0, 1.0])


def _model():
    builder = ModelBuilder("value_batch")
    builder.add_body("base", mass=1.0, inertia=torch.eye(3) * 0.1)
    builder.add_body("link", mass=2.0, inertia=torch.eye(3) * 0.2)
    builder.add_body("tip_link", mass=0.5, inertia=torch.eye(3) * 0.05)
    builder.add_joint(
        "shoulder",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
        parent="base",
        child="link",
        origin=_pose(0.5),
        lower=-2.0,
        upper=2.0,
    )
    builder.add_joint(
        "elbow",
        kind=JointRevolute(axis=(0.0, 1.0, 0.0)),
        parent="link",
        child="tip_link",
        origin=_pose(0.4),
        lower=-2.0,
        upper=2.0,
    )
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


def _outputs(model, state, v: torch.Tensor, shared: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Every value a state yields: poses, the Jacobians of a frame, a joint and points (``shared: (K, 3)`` and
    ``points: (B..., K, 3)``), and both time variations."""
    frames = ["tip", "link"]
    return (
        state.joint_poses,
        state.frame_poses,
        frame_jacobian(model, state, frame="tip"),
        joint_jacobian(model, state, joint="elbow", reference="world"),
        joint_jacobian(model, state, joint="elbow", reference="local"),
        point_jacobian(model, state, frames=frames, points=shared),
        point_jacobian(model, state, frames=frames, points=points),
        frame_jacobian_time_variation(model, state, v=v, frame="tip"),
        frame_jacobian_time_variation(model, state, v=v, frame="tip", reference="local"),
        joint_jacobian_time_variation(model, state, v=v, joint="elbow", reference="world"),
    )


@pytest.mark.parametrize("fields", [("joint_placements", "frame_placements"), ("frame_placements",)])
def test_identity_batch_broadcasts_against_the_configuration_batch(fields: tuple[str, ...]) -> None:
    """``I = (2,)`` against ``B = (3, 1)`` gives the state batch ``(3, 2)``; every index equals the scalar loop. A
    batch on the frame placements alone still sets the state batch, and the frame Jacobians read its rows."""
    model = _model()
    placements, frames = _identities(model, 2)
    values = {"joint_placements": placements, "frame_placements": frames}
    values = {name: values[name] for name in fields}
    batched = model.with_values(**values)
    generator = torch.Generator().manual_seed(0)
    q = torch.rand(3, 1, model.nq, generator=generator, dtype=torch.float64) - 0.5
    v = torch.randn(3, 1, model.nv, generator=generator, dtype=torch.float64)
    shared = torch.randn(2, 3, generator=generator, dtype=torch.float64)  # (K, 3)
    points = torch.randn(3, 2, 2, 3, generator=generator, dtype=torch.float64)  # (B..., K, 3) at the state batch

    def batched_call():
        state = forward_kinematics(batched, q)
        assert state.q.shape == (3, 2, model.nq)
        assert state.frame_poses.shape == (3, 2, model.nframes, 7)
        return _outputs(batched, state, v, shared, points)

    def loop_call(index: tuple[int, ...]):
        row, identity = index
        scalar = model.with_values(**{name: value[identity] for name, value in values.items()})
        state = forward_kinematics(scalar, q[row, 0])
        return _outputs(scalar, state, v[row, 0], shared, points[row, identity])

    assert_value_batched_matches_loop(batched_call, loop_call, execution_batch_shape=(3, 2), rtol=0.0, atol=0.0)


def test_time_variations_broadcast_v_against_the_state_batch() -> None:
    """``v`` of batch ``(3, 2)`` against a state of batch ``(3, 1)`` (C-TENS-2, both ways): J̇ is linear in ``v``, so
    several velocities per configuration are well defined; every index equals the scalar loop."""
    model = _model()
    generator = torch.Generator().manual_seed(1)
    q = torch.rand(3, 1, model.nq, generator=generator, dtype=torch.float64) - 0.5
    v = torch.randn(3, 2, model.nv, generator=generator, dtype=torch.float64)

    def variations(state, v: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            frame_jacobian_time_variation(model, state, v=v, frame="tip"),
            frame_jacobian_time_variation(model, state, v=v, frame="tip", reference="local"),
            joint_jacobian_time_variation(model, state, v=v, joint="elbow", reference="world"),
        )

    def loop_call(index: tuple[int, ...]):
        row, velocity = index
        return variations(forward_kinematics(model, q[row, 0]), v[row, velocity])

    assert_value_batched_matches_loop(
        lambda: variations(forward_kinematics(model, q), v), loop_call, execution_batch_shape=(3, 2), rtol=0.0, atol=0.0
    )


def test_batched_joint_placement_autograd_reaches_leaf() -> None:
    """Batched placements are applied by their own product in forward kinematics (unbatched ones fold into the joint
    map), so their gradient is checked apart from C-DIFF-1's unbatched one."""
    model = _model()
    delta = torch.zeros(2, model.njoints, 7, dtype=torch.float64, requires_grad=True)
    state = forward_kinematics(model.with_values(joint_placements=model.joint_placements + delta), model.q_neutral)
    state.joint_poses[..., 2, :3].square().sum().backward()
    assert torch.isfinite(delta.grad).all()
    assert delta.grad.abs().sum() > 0


def test_non_broadcastable_batches_raise() -> None:
    model = _model()
    placements, _ = _identities(model, 2)
    batched = model.with_values(joint_placements=placements)
    q = model.q_neutral.expand(3, -1)
    with pytest.raises(ShapeError, match=r"cannot broadcast q batch \(3,\) with joint_placements batch \(2,\)"):
        forward_kinematics(batched, q)
    state = forward_kinematics(batched, q.unsqueeze(1))  # batch (3, 2)
    for points_batch in ((5,), (4, 2)):
        points = torch.zeros(*points_batch, 2, 3, dtype=torch.float64)
        with pytest.raises(ShapeError, match=rf"cannot broadcast points batch {re.escape(str(points_batch))} with"):
            point_jacobian(batched, state, frames=["tip", "link"], points=points)
        v = torch.zeros(*points_batch, model.nv, dtype=torch.float64)
        with pytest.raises(ShapeError, match=rf"cannot broadcast v batch {re.escape(str(points_batch))} with"):
            frame_jacobian_time_variation(batched, state, v=v, frame="tip")
