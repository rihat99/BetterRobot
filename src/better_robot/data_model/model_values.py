"""Tensor-only, differentiable values paired with :class:`ModelStructure`."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.utils import _pytree

from ..lie.tangents import hat_so3

if TYPE_CHECKING:
    from .model import Model


def packed_inertias_to_6x6(body_inertias: torch.Tensor) -> torch.Tensor:
    """Expand ``(..., 10)`` packed inertias to linear-first ``(..., 6, 6)``."""

    mass = body_inertias[..., 0]
    com = body_inertias[..., 1:4]
    sym = body_inertias[..., 4:10]
    ixx, iyy, izz, ixy, ixz, iyz = sym.unbind(dim=-1)
    row0 = torch.stack((ixx, ixy, ixz), dim=-1)
    row1 = torch.stack((ixy, iyy, iyz), dim=-1)
    row2 = torch.stack((ixz, iyz, izz), dim=-1)
    inertia_com = torch.stack((row0, row1, row2), dim=-2)
    hat_com = hat_so3(com)
    inertia_origin = inertia_com - mass[..., None, None] * (hat_com @ hat_com)
    eye = torch.eye(3, dtype=body_inertias.dtype, device=body_inertias.device)
    mass_eye = mass[..., None, None] * eye
    mass_hat = mass[..., None, None] * hat_com
    top = torch.cat((mass_eye, -mass_hat), dim=-1)
    bottom = torch.cat((mass_hat, inertia_origin), dim=-1)
    return torch.cat((top, bottom), dim=-2)


@dataclass(frozen=True)
class ModelValues:
    """All floating-point model inputs as a torch tensor pytree."""

    joint_placements: torch.Tensor
    body_inertias: torch.Tensor
    frame_placements: torch.Tensor
    lower_pos_limit: torch.Tensor
    upper_pos_limit: torch.Tensor
    velocity_limit: torch.Tensor
    effort_limit: torch.Tensor
    rotor_inertia: torch.Tensor
    armature: torch.Tensor
    friction: torch.Tensor
    damping: torch.Tensor
    gravity: torch.Tensor
    mimic_multiplier: torch.Tensor
    mimic_offset: torch.Tensor
    q_neutral: torch.Tensor
    body_inertias_6x6: torch.Tensor | None = None

    @classmethod
    def from_model(cls, model: "Model") -> "ModelValues":
        if model.frames:
            frames = torch.stack(
                [
                    frame.joint_placement.to(
                        device=model.joint_placements.device,
                        dtype=model.joint_placements.dtype,
                    )
                    for frame in model.frames
                ],
                dim=0,
            )
        else:
            frames = model.joint_placements.new_empty((0, 7))
        cached_inertias = (
            None
            if model.body_inertias.requires_grad
            else packed_inertias_to_6x6(model.body_inertias)
        )
        return cls(
            joint_placements=model.joint_placements,
            body_inertias=model.body_inertias,
            frame_placements=frames,
            lower_pos_limit=model.lower_pos_limit,
            upper_pos_limit=model.upper_pos_limit,
            velocity_limit=model.velocity_limit,
            effort_limit=model.effort_limit,
            rotor_inertia=model.rotor_inertia,
            armature=model.armature,
            friction=model.friction,
            damping=model.damping,
            gravity=model.gravity,
            mimic_multiplier=model.mimic_multiplier,
            mimic_offset=model.mimic_offset,
            q_neutral=model.q_neutral,
            body_inertias_6x6=cached_inertias,
        )

    def spatial_inertias(self) -> torch.Tensor:
        """Return cached static inertias or derive live parametric inertias once."""

        if self.body_inertias_6x6 is not None:
            return self.body_inertias_6x6
        return packed_inertias_to_6x6(self.body_inertias)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ModelValues":
        replacements = {
            field.name: value.to(device=device, dtype=dtype)
            for field in dataclasses.fields(self)
            if isinstance((value := getattr(self, field.name)), torch.Tensor)
        }
        return dataclasses.replace(self, **replacements)

    def tree_flatten(self) -> tuple[list[torch.Tensor], tuple[str, ...]]:
        names: list[str] = []
        leaves: list[torch.Tensor] = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is not None:
                names.append(field.name)
                leaves.append(value)
        return leaves, tuple(names)

    @classmethod
    def tree_unflatten(
        cls,
        leaves: list[torch.Tensor],
        context: tuple[str, ...],
    ) -> "ModelValues":
        values: dict[str, Any] = dict(zip(context, leaves))
        values.setdefault("body_inertias_6x6", None)
        return cls(**values)


_pytree.register_pytree_node(
    ModelValues,
    lambda values: values.tree_flatten(),
    lambda leaves, context: ModelValues.tree_unflatten(list(leaves), context),
)


__all__ = ["ModelValues", "packed_inertias_to_6x6"]
