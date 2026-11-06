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
from better_robot.io.build_model import build_model
from better_robot.io.builder import ModelBuilder


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
    with pytest.raises(ShapeError, match=rf"{name}\.shape must end in .* got"):
        model.with_values(**{name: torch.zeros(shape)})


def test_with_values_armature_is_unbatched() -> None:
    model = _model()
    assert model.with_values(armature=torch.tensor([0.1])).armature.tolist() == pytest.approx([0.1])
    with pytest.raises(ShapeError, match=r"armature\.shape must be \(1,\)"):
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
