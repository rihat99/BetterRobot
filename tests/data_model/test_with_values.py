"""Public parametric ModelValues rebind and transfer contracts."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import (
    DeviceMismatchError,
    DtypeMismatchError,
    ShapeError,
)
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder


def _pose(x: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _model(dtype: torch.dtype = torch.float32):
    builder = ModelBuilder("with_values")
    builder.add_body("base", mass=1.0, inertia=torch.eye(3) * 0.1)
    builder.add_body("link", mass=2.0, com=torch.tensor([0.1, 0.0, 0.0]), inertia=torch.eye(3) * 0.2)
    builder.add_revolute_z(
        "joint",
        parent="base",
        child="link",
        origin=_pose(0.5),
        lower=-1.0,
        upper=1.0,
    )
    builder.add_frame("tip", parent_body="link", placement=_pose(0.25))
    return build_model(builder.finalize(), dtype=dtype)


def test_with_values_reuses_structure_and_normalizes_placements() -> None:
    model = _model()
    placements = model.values.joint_placements.repeat(3, 1, 1)
    placements[..., 3:7] *= 2.0
    rebound = model.with_values(joint_placements=placements)

    assert rebound is not model
    assert rebound.structure is model.structure
    assert rebound.values.joint_placements.shape == (3, model.njoints, 7)
    torch.testing.assert_close(
        rebound.values.joint_placements[..., 3:7].norm(dim=-1),
        torch.ones(3, model.njoints),
    )
    assert rebound.joint_placements is rebound.values.joint_placements


@pytest.mark.parametrize(
    ("name", "shape"),
    (
        ("joint_placements", (2, 7)),
        ("body_inertias", (2, 9)),
        ("frame_placements", (2, 8)),
    ),
)
def test_with_values_rejects_wrong_event_shapes(name: str, shape: tuple[int, ...]) -> None:
    model = _model()
    with pytest.raises(ShapeError, match=rf"{name}\.shape must end in .* got"):
        model.with_values(**{name: torch.zeros(shape)})


def test_with_values_rejects_dtype_device_and_incompatible_value_batches() -> None:
    model = _model()
    with pytest.raises(DtypeMismatchError, match="body_inertias.dtype"):
        model.with_values(body_inertias=model.values.body_inertias.double())
    with pytest.raises(DeviceMismatchError, match="frame_placements.device"):
        model.with_values(
            frame_placements=torch.empty(
                model.nframes,
                7,
                device="meta",
                dtype=model.values.frame_placements.dtype,
            )
        )
    with pytest.raises(ShapeError, match="cannot broadcast execution batch"):
        model.with_values(
            joint_placements=model.values.joint_placements.expand(2, -1, -1),
            body_inertias=model.values.body_inertias.expand(3, -1, -1),
        )


def test_to_preserves_rebound_frame_table_and_integer_structure_dtypes() -> None:
    model = _model()
    frames = model.values.frame_placements.repeat(2, 1, 1)
    frames[..., 0] += torch.tensor([0.1, 0.2])[:, None]
    rebound = model.with_values(frame_placements=frames)
    moved = rebound.to(dtype=torch.float64)

    assert moved.values.frame_placements.shape == frames.shape
    assert moved.values.frame_placements.dtype == torch.float64
    assert moved.structure.parents_tensor.dtype == model.structure.parents_tensor.dtype
    assert moved.structure.joint_kind_tensor.dtype == model.structure.joint_kind_tensor.dtype
    torch.testing.assert_close(moved.values.frame_placements, frames.double())


def test_rebound_inertias_are_derived_live() -> None:
    model = _model(dtype=torch.float64)
    assert model.values.spatial_inertias() is not model.values.spatial_inertias()

    delta = torch.zeros_like(model.values.body_inertias, requires_grad=True)
    rebound = model.with_values(body_inertias=model.values.body_inertias + delta)
    first = rebound.values.spatial_inertias()
    second = rebound.values.spatial_inertias()
    assert first is not second
    first.sum().backward()
    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
