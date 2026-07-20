"""Contracts for the object-owned optimization variable hierarchy."""

from __future__ import annotations

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.optim.variables import Bounds, RobotVariable, SE3Variable, SO3Variable, Variable


def _mixed_model():
    builder = ModelBuilder("mixed")
    base = builder.add_body("base", mass=1.0)
    tip = builder.add_body("tip", mass=1.0)
    builder.add_free_flyer_root("floating", child=base)
    builder.add_revolute_z("hinge", parent=base, child=tip, lower=-0.25, upper=0.25)
    return build_model(builder.finalize())


def test_plain_variable_owns_value_and_declares_batch_axes_explicitly() -> None:
    event = Variable(torch.zeros(3, 2), name="event")
    batched = Variable(torch.zeros(3, 2), name="batched", batch_ndim=1)

    assert event.shape == (3, 2)
    assert event.batch_shape == ()
    assert event.tangent_dim() == 6
    assert batched.shape == (2,)
    assert batched.batch_shape == (3,)
    assert batched.tangent_dim() == 2


def test_typed_variables_infer_batch_axes_and_round_trip() -> None:
    rotation = SO3Variable(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, 4).clone())
    pose = SE3Variable(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(2, 7).clone())
    rotation_delta = torch.tensor([[0.01, -0.02, 0.03], [-0.02, 0.01, 0.04]])
    pose_delta = torch.cat((torch.full((2, 3), 0.01), rotation_delta), dim=-1)

    rotation_next = rotation.retract(rotation_delta)
    pose_next = pose.retract(pose_delta)

    assert rotation.batch_shape == pose.batch_shape == (2,)
    torch.testing.assert_close(rotation._difference_from(rotation.tensor, rotation_next), rotation_delta)
    torch.testing.assert_close(pose._difference_from(pose.tensor, pose_next), pose_delta)


def test_robot_variable_preserves_geometry_limits_and_trajectory_layout() -> None:
    model = _mixed_model()
    trajectory = model.q_neutral.expand(4, model.nq).clone()
    variable = RobotVariable(model, trajectory, name="q", bounds=True, time_axis=0)

    assert variable.shape == (4, model.nq)
    assert variable.batch_shape == ()
    assert variable.tangent_dim() == 4 * model.nv
    assert variable.temporal_tangent_width == model.nv
    assert variable.bounds is not None

    delta = torch.zeros(variable.free_dim)
    delta[-1] = 2.0
    projected = variable.retract(delta)
    assert projected[-1, -1].item() == pytest.approx(0.25)
    torch.testing.assert_close(
        variable._difference_from(variable.tensor, projected),
        model.difference(variable.tensor, projected).reshape(-1),
    )


def test_group_variables_reject_box_bounds() -> None:
    bounds4 = Bounds(torch.full((4,), -1.0), torch.full((4,), 1.0))
    bounds7 = Bounds(torch.full((7,), -1.0), torch.full((7,), 1.0))

    with pytest.raises(ValueError, match="SO3 variables have no meaningful global box bound"):
        SO3Variable(torch.tensor([0.0, 0.0, 0.0, 1.0]), bounds=bounds4)
    with pytest.raises(ValueError, match="SE3 variables have no meaningful global box bound"):
        SE3Variable(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), bounds=bounds7)


def test_auto_names_are_unique_within_each_variable_type() -> None:
    first = Variable(torch.zeros(1))
    second = Variable(torch.zeros(1))

    assert first.name.startswith("variable_")
    assert second.name.startswith("variable_")
    assert first.name != second.name
