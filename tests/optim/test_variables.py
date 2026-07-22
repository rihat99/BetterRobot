"""Optimization variable hierarchy contracts."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError
from better_robot.io import ModelBuilder, build_model
from better_robot.optim.variables import Bounds, RobotVariable, SE3Variable, SO3Variable, Variable
from better_robot.residuals.base import Difference


def _mixed_model():
    builder = ModelBuilder("mixed")
    base = builder.add_body("base", mass=1.0)
    tip = builder.add_body("tip", mass=1.0)
    builder.add_free_flyer_root("floating", child=base)
    builder.add_revolute_z("hinge", parent=base, child=tip, lower=-0.25, upper=0.25)
    return build_model(builder.finalize())


def _tangent_group_model():
    builder = ModelBuilder("grouped")
    base = builder.add_body("base", mass=1.0)
    middle = builder.add_body("middle", mass=1.0)
    tip = builder.add_body("tip", mass=1.0)
    builder.add_free_flyer_root("floating", child=base)
    builder.add_spherical("ball", parent=base, child=middle)
    builder.add_revolute_z("hinge", parent=middle, child=tip, lower=-0.5, upper=0.5)
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


def test_bool_and_integer_variables_are_static_only_and_have_no_tangent() -> None:
    mask = Variable(torch.tensor([True, False]), trainable=False)
    labels = Variable(torch.tensor([1, 2]), trainable=False)

    assert mask.tensor.dtype == torch.bool
    assert labels.tensor.dtype == torch.int64
    with pytest.raises(AssertionError, match="have no tangent"):
        mask.tangent_dim()
    with pytest.raises(DtypeMismatchError, match="trainable tensor must use"):
        Variable(torch.tensor([True]))


def test_robot_tangent_groups_and_weights_match_topology() -> None:
    variable = RobotVariable(_tangent_group_model())
    groups = variable.tangent_groups()

    assert tuple(groups) == ("universe", "floating", "ball", "hinge", "root", "root_lin", "root_ang", "joints")
    torch.testing.assert_close(groups["root"], torch.arange(6))
    torch.testing.assert_close(groups["root_lin"], torch.arange(3))
    torch.testing.assert_close(groups["root_ang"], torch.arange(3, 6))
    torch.testing.assert_close(groups["ball"], torch.arange(6, 9))
    torch.testing.assert_close(groups["joints"], torch.arange(6, 10))
    torch.testing.assert_close(
        variable.tangent_weight({"root_lin": 2.0, "root_ang": 3.0, "joints": 4.0}),
        torch.tensor([2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]),
    )


def test_fixed_base_model_rejects_missing_root_group_by_model_name() -> None:
    builder = ModelBuilder("fixed_groups")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_revolute_z("hinge", parent=base, child=tip)
    model = build_model(builder.finalize())
    variable = RobotVariable(model)

    assert "root" not in variable.tangent_groups()
    with pytest.raises(ValueError, match="fixed_groups.*root"):
        variable.tangent_weight({"root": 2.0})
    with pytest.raises(ValueError, match="fixed_groups.*root"):
        RobotVariable(model, frozen_groups=("root",))


@pytest.mark.parametrize("reserved_name", ("root", "root_lin", "root_ang", "joints"))
def test_derived_group_name_collision_raises_instead_of_overwriting_joint(reserved_name: str) -> None:
    builder = ModelBuilder("ambiguous_groups")
    base = builder.add_body("base")
    middle = builder.add_body("middle")
    tip = builder.add_body("tip")
    builder.add_revolute_x(reserved_name, parent=base, child=middle)
    builder.add_revolute_y("other", parent=middle, child=tip)
    variable = RobotVariable(build_model(builder.finalize()))

    with pytest.raises(ValueError, match=rf"ambiguous_groups.*{reserved_name}.*conflict"):
        variable.tangent_groups()
    with pytest.raises(ValueError, match=rf"ambiguous_groups.*{reserved_name}.*conflict"):
        RobotVariable(variable.model, frozen_groups=(reserved_name,))


def test_frozen_robot_groups_gather_expand_and_retract_knot_major() -> None:
    model = _tangent_group_model()
    trajectory = model.q_neutral.expand(2, model.nq).clone()
    variable = RobotVariable(model, trajectory, time_axis=0, frozen_groups=("root",))
    expected_indices = torch.tensor([6, 7, 8, 9, 16, 17, 18, 19])

    assert variable.frozen_groups == ("root",)
    assert variable.free_dim == 8
    assert variable.temporal_tangent_width == 4
    torch.testing.assert_close(variable.free_indices, expected_indices)
    torch.testing.assert_close(variable.temporal_free_indices, torch.arange(6, 10))
    exposed = variable.free_indices
    exposed.zero_()
    torch.testing.assert_close(variable.free_indices, expected_indices)
    with pytest.raises(AttributeError):
        variable.frozen_groups = ()  # type: ignore[misc]

    for full in (torch.arange(20), torch.arange(20) % 2 == 0):
        gathered = variable.gather_tangent(full)
        expanded = variable.expand_tangent(gathered)
        assert gathered.dtype == expanded.dtype == full.dtype
        torch.testing.assert_close(gathered, full[expected_indices])
        torch.testing.assert_close(expanded[expected_indices], gathered)
        torch.testing.assert_close(expanded.reshape(2, 10)[:, :6], torch.zeros(2, 6, dtype=full.dtype))

    delta = torch.full((variable.free_dim,), 0.01)
    expanded_delta = variable.expand_tangent(delta).reshape(2, model.nv)
    retracted = variable.retract(delta)
    torch.testing.assert_close(retracted, model.integrate(trajectory, expanded_delta))
    torch.testing.assert_close(retracted[:, :7], trajectory[:, :7], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        variable._difference_from(trajectory, retracted),
        model.difference(trajectory, retracted).reshape(-1),
    )


def test_difference_residual_keeps_full_tangent_when_groups_are_frozen() -> None:
    model = _tangent_group_model()
    variable = RobotVariable(model, frozen_groups=("joints",))
    item = Difference(variable, model.q_neutral.clone())

    assert variable.free_dim == 6
    assert item.dim == model.nv
    assert item.error().shape == (model.nv,)


def test_frozen_bounded_coordinate_is_not_projected_from_its_current_value() -> None:
    model = _tangent_group_model()
    outside = model.q_neutral.clone()
    outside[-1] = 1.0
    variable = RobotVariable(model, outside, bounds=True, frozen_groups=("hinge",))

    candidate = model.q_neutral.clone()
    candidate[-1] = 1.0
    projected = variable.project(candidate)
    retracted = variable.retract(torch.zeros(variable.free_dim))

    assert projected[-1].item() == pytest.approx(0.5)
    assert retracted[-1].item() == 1.0
