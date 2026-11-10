"""``Model.with_values`` and ``Model.to``: validated tensor rebinds that keep every static field."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from better_robot.exceptions import (
    ArgumentError,
    DeviceMismatchError,
    DtypeMismatchError,
    ShapeError,
)
from better_robot import dynamics, kinematics, load
from better_robot.dynamics._world import dynamics_plan
from better_robot.io import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.kinematics._plans import fk_plan, jacobian_plan
from better_robot.kinematics.configuration import manifold_plan, mimic_plan
from tests.support.robots import panda_urdf


def _pose(x: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _model(dtype: torch.dtype = torch.float32):
    builder = ModelBuilder("with_values")
    builder.add_body("base", mass=1.0, inertia=torch.eye(3) * 0.1)
    builder.add_body("link", mass=2.0, com=torch.tensor([0.1, 0.0, 0.0]), inertia=torch.eye(3) * 0.2)
    builder.add_revolute(
        "joint",
        axis=(0.0, 0.0, 1.0),
        parent="base",
        child="link",
        origin=_pose(0.5),
        lower=-1.0,
        upper=1.0,
    )
    builder.add_frame("tip", parent="link", placement=_pose(0.25))
    return build_model(builder.finalize(), dtype=dtype)


def _tensor_fields(model) -> list[str]:
    return [f.name for f in dataclasses.fields(model) if isinstance(getattr(model, f.name), torch.Tensor)]


def test_with_values_keeps_static_fields_and_normalizes_placements() -> None:
    model = _model()
    placements = model.joint_placements.repeat(3, 1, 1)
    placements[..., 3:7] *= 2.0
    rebound = model.with_values(joint_placements=placements)

    assert rebound is not model
    assert rebound.structure_key == model.structure_key
    assert rebound.joints is model.joints
    assert rebound.body_inertias is model.body_inertias
    assert rebound.joint_placements.shape == (3, model.njoints, 7)
    torch.testing.assert_close(
        rebound.joint_placements[..., 3:7].norm(dim=-1),
        torch.ones(3, model.njoints),
    )


def test_with_values_accepts_every_tensor_field_and_nothing_else() -> None:
    model = _model()
    names = _tensor_fields(model)
    rebound = model.with_values(**{name: getattr(model, name).clone() for name in names})
    for name in names:
        torch.testing.assert_close(getattr(rebound, name), getattr(model, name))
    with pytest.raises(ArgumentError, match="unknown tensor fields"):
        model.with_values(mass=torch.zeros(1))


@pytest.mark.parametrize(
    ("name", "shape"),
    (
        ("joint_placements", (2, 7)),
        ("body_inertias", (2, 9)),
        ("frame_placements", (2, 8)),
        ("armature", (3,)),
    ),
)
def test_with_values_rejects_wrong_event_shapes(name: str, shape: tuple[int, ...]) -> None:
    model = _model()
    with pytest.raises(ShapeError, match=rf"{name} must have shape \(\.\.\., .*\), got"):
        model.with_values(**{name: torch.zeros(shape)})


def test_with_values_armature_is_unbatched() -> None:
    model = _model()
    assert model.with_values(armature=torch.tensor([0.1])).armature.tolist() == pytest.approx([0.1])
    with pytest.raises(ShapeError, match=r"armature must have shape \(1,\)"):
        model.with_values(armature=torch.zeros(2, 1))


def test_with_values_rejects_dtype_device_and_incompatible_value_batches() -> None:
    model = _model()
    with pytest.raises(DtypeMismatchError, match="body_inertias.dtype"):
        model.with_values(body_inertias=model.body_inertias.double())
    with pytest.raises(DeviceMismatchError, match="frame_placements.device"):
        model.with_values(
            frame_placements=torch.empty(
                model.nframes,
                7,
                device="meta",
                dtype=model.frame_placements.dtype,
            )
        )
    with pytest.raises(ShapeError, match="value batches do not broadcast"):
        model.with_values(
            joint_placements=model.joint_placements.expand(2, -1, -1),
            body_inertias=model.body_inertias.expand(3, -1, -1),
        )


def test_to_moves_every_tensor_field_and_keeps_a_rebound_batch() -> None:
    model = _model()
    frames = model.frame_placements.repeat(2, 1, 1)
    frames[..., 0] += torch.tensor([0.1, 0.2])[:, None]
    rebound = model.with_values(frame_placements=frames)
    moved = rebound.to(dtype=torch.float64)

    assert moved.structure_key == model.structure_key
    assert moved.frame_placements.shape == frames.shape
    assert all(getattr(moved, name).dtype == torch.float64 for name in _tensor_fields(moved))
    torch.testing.assert_close(moved.frame_placements, frames.double())


_PLANS = (
    fk_plan,
    jacobian_plan,
    dynamics_plan,
    mimic_plan,
    lambda model, *, dtype, device: manifold_plan(model, device=device),
)


def _queries(model) -> tuple[torch.Tensor, ...]:
    """Every result that reads a plan: FK, a frame Jacobian, RNEA, CRBA, ABA and a configuration step."""
    q = model.q_neutral + 0.1
    v = torch.linspace(-0.5, 0.5, model.nv, dtype=model.dtype)
    state = kinematics.forward_kinematics(model, q)
    return (
        state.frame_poses,
        kinematics.frame_jacobian(model, state, frame=model.frame_names[-1]),
        dynamics.inverse_dynamics(model, state, v=v, a=v),
        dynamics.mass_matrix(model, state),
        dynamics.forward_dynamics(model, state, v=v, tau=v),
        kinematics.integrate(model, q, v),
    )


@pytest.mark.parametrize("name", _tensor_fields(_model()))
def test_a_copy_shares_its_source_plans_and_computes_as_a_fresh_model(name: str) -> None:
    """A ``with_values`` or ``to`` copy takes its source's plans (they depend on the static fields only) and
    computes, bitwise, what a model with its values and plans of its own computes (so no value memo came along)."""
    model = load(panda_urdf(), dtype=torch.float64)
    _queries(model)  # the source builds every plan
    changed = getattr(model, name) * 1.01  # every entry moves; placements are renormalised by with_values
    copy = model.with_values(**{name: changed})
    copied = _queries(copy)
    fresh = torch.utils._pytree.tree_unflatten(*torch.utils._pytree.tree_flatten(copy))  # no plan owner
    expected = _queries(fresh)

    key = {"dtype": model.dtype, "device": model.device}
    assert all(plan(copy, **key) is plan(model, **key) for plan in _PLANS)
    assert all(plan(fresh, **key) is not plan(model, **key) for plan in _PLANS)
    for result, oracle in zip(copied, expected, strict=True):
        assert torch.equal(result, oracle), name
    moved = copy.to(dtype=torch.float32)
    _queries(moved)
    assert fk_plan(moved, dtype=torch.float32, device=model.device) is fk_plan(
        model, dtype=torch.float32, device=model.device
    )
    assert [key for key in model._plans if key[0] == "manifold"] == [("manifold", torch.long, model.device)]


def test_a_replace_that_changes_the_structure_builds_its_own_plans() -> None:
    model = load(panda_urdf())
    copy = model.with_values(gravity=model.gravity * 2.0)
    renamed = dataclasses.replace(copy, name="other")  # carries the plan link; the structure key differs
    key = {"dtype": model.dtype, "device": model.device}
    assert fk_plan(copy, **key) is fk_plan(model, **key)
    assert fk_plan(renamed, **key) is not fk_plan(model, **key)
