"""``Model`` is a pytree: the tensor fields are its leaves, the static fields its context, so ``torch.func``
transforms map over a batch of models exactly like a loop of ``with_values``."""

from __future__ import annotations

import dataclasses

import torch
from torch.utils import _pytree

from better_robot.io import build_model
from better_robot.io.builder import ModelBuilder
from better_robot.kinematics import forward_kinematics
from better_robot.model import Model


def _arm() -> Model:
    builder = ModelBuilder("arm")
    builder.add_body("base", mass=1.0)
    builder.add_body("upper", mass=1.0)
    builder.add_body("hand", mass=0.5)
    builder.add_revolute("shoulder", axis=(0.0, 0.0, 1.0), parent="base", child="upper", lower=-1.0, upper=1.0)
    builder.add_revolute(
        "elbow",
        axis=(0.0, 1.0, 0.0),
        parent="upper",
        child="hand",
        origin=torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
        lower=-1.0,
        upper=1.0,
    )
    builder.add_frame("tip", parent="hand", placement=torch.tensor([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]))
    return build_model(builder.finalize(), dtype=torch.float64)


def test_leaves_are_the_tensor_fields_and_unflatten_keeps_the_static_fields() -> None:
    model = _arm()
    leaves, spec = _pytree.tree_flatten(model)
    tensor_fields = [f.name for f in dataclasses.fields(model) if isinstance(getattr(model, f.name), torch.Tensor)]
    assert len(leaves) == len(tensor_fields) and all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    assert [leaf.shape for leaf in leaves] == [getattr(model, name).shape for name in tensor_fields]

    rebuilt = _pytree.tree_unflatten([leaf * 2 for leaf in leaves], spec)
    assert isinstance(rebuilt, Model) and rebuilt is not model
    assert rebuilt.structure_key == model.structure_key
    assert torch.equal(rebuilt.joint_placements, 2 * model.joint_placements)
    assert rebuilt._plans == {}


def test_vmap_over_batched_placements_matches_with_values_loop() -> None:
    model = _arm()
    generator = torch.Generator().manual_seed(0)
    count = 3
    joint_placements = model.joint_placements.expand(count, -1, -1).clone()
    joint_placements[..., :3] += 0.05 * torch.randn(count, model.njoints, 3, generator=generator, dtype=torch.float64)
    frame_placements = model.frame_placements.expand(count, -1, -1).clone()
    frame_placements[..., :3] += 0.05 * torch.randn(count, model.nframes, 3, generator=generator, dtype=torch.float64)
    q = torch.tensor([0.3, -0.4], dtype=torch.float64)

    def tip_position(sample: Model) -> torch.Tensor:
        return forward_kinematics(sample, q).frame_translations[-1]

    batched = model.with_values(joint_placements=joint_placements, frame_placements=frame_placements)
    in_dims = _pytree.tree_unflatten(
        [0 if name in ("joint_placements", "frame_placements") else None for name in _tensor_names(model)],
        _pytree.tree_flatten(model)[1],
    )
    by_vmap = torch.func.vmap(tip_position, in_dims=(in_dims,))(batched)
    by_loop = torch.stack(
        [
            tip_position(model.with_values(joint_placements=joint_placements[i], frame_placements=frame_placements[i]))
            for i in range(count)
        ]
    )
    torch.testing.assert_close(by_vmap, by_loop)
    torch.testing.assert_close(by_vmap, forward_kinematics(batched, q).frame_translations[:, -1])


def _tensor_names(model: Model) -> list[str]:
    return [f.name for f in dataclasses.fields(model) if isinstance(getattr(model, f.name), torch.Tensor)]
