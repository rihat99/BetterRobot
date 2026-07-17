"""Frozen topology shared by the torch and Warp compute lanes.

``ModelStructure`` deliberately stores the robot topology twice: Python
tuples keep the torch lane statically unrollable under ``torch.compile``;
flat tensors expose the same data to device kernels.  Construction is the
only place where the two representations are allowed to diverge, and
``validate_consistency`` makes that boundary explicit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import torch

from .joint_models.base import JointModel

if TYPE_CHECKING:
    from .model import Model


# Stable kernel ABI.  New kinds append codes; existing values never change.
JOINT_KIND_CODES: Mapping[str, int] = MappingProxyType(
    {
        "universe": 0,
        "fixed": 1,
        "revolute_rx": 2,
        "revolute_ry": 3,
        "revolute_rz": 4,
        "revolute_unaligned": 5,
        "revolute_unbounded": 6,
        "prismatic_px": 7,
        "prismatic_py": 8,
        "prismatic_pz": 9,
        "prismatic_unaligned": 10,
        "spherical": 11,
        "free_flyer": 12,
        "planar": 13,
        "translation": 14,
        "helical": 15,
        "composite": 16,
        "mimic": 17,
    }
)


def _flatten_rows(rows: tuple[tuple[int, ...], ...]) -> tuple[list[int], list[int]]:
    offsets = [0]
    indices: list[int] = []
    for row in rows:
        indices.extend(row)
        offsets.append(len(indices))
    return offsets, indices


def _joint_axis(joint: JointModel) -> tuple[float, float, float]:
    aligned = {
        "revolute_rx": (1.0, 0.0, 0.0),
        "revolute_ry": (0.0, 1.0, 0.0),
        "revolute_rz": (0.0, 0.0, 1.0),
        "prismatic_px": (1.0, 0.0, 0.0),
        "prismatic_py": (0.0, 1.0, 0.0),
        "prismatic_pz": (0.0, 0.0, 1.0),
    }
    if joint.kind in aligned:
        return aligned[joint.kind]
    axis = getattr(joint, "axis", None)
    if axis is None:
        return (0.0, 0.0, 0.0)
    values = axis.detach().to(device="cpu", dtype=torch.float64).tolist()
    return (float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class ModelStructure:
    """Immutable robot topology in static and kernel-consumable forms."""

    njoints: int
    nbodies: int
    nframes: int
    nq: int
    nv: int

    name: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    frame_names: tuple[str, ...]

    parents: tuple[int, ...]
    children: tuple[tuple[int, ...], ...]
    subtrees: tuple[tuple[int, ...], ...]
    supports: tuple[tuple[int, ...], ...]
    topo_order: tuple[int, ...]
    nqs: tuple[int, ...]
    nvs: tuple[int, ...]
    idx_qs: tuple[int, ...]
    idx_vs: tuple[int, ...]
    joint_models: tuple[JointModel, ...]
    joint_kind_codes: tuple[int, ...]
    mimic_source: tuple[int, ...]

    joint_kind_tensor: torch.Tensor
    parents_tensor: torch.Tensor
    topo_order_tensor: torch.Tensor
    nqs_tensor: torch.Tensor
    nvs_tensor: torch.Tensor
    idx_qs_tensor: torch.Tensor
    idx_vs_tensor: torch.Tensor
    children_offsets: torch.Tensor
    children_indices: torch.Tensor
    subtree_offsets: torch.Tensor
    subtree_indices: torch.Tensor
    support_offsets: torch.Tensor
    support_indices: torch.Tensor
    frame_parent_joints: torch.Tensor
    mimic_source_tensor: torch.Tensor
    joint_axes: torch.Tensor
    joint_pitches: torch.Tensor
    joint_motion_subspaces: torch.Tensor

    @classmethod
    def from_model(cls, model: "Model") -> "ModelStructure":
        device = model.joint_placements.device
        dtype = model.joint_placements.dtype
        # ``-1`` is the stable torch-only/custom-joint sentinel.  Device
        # kernels reject/fallback on it without preventing Model construction.
        codes = tuple(JOINT_KIND_CODES.get(joint.kind, -1) for joint in model.joint_models)
        axes = torch.tensor(
            [_joint_axis(joint) for joint in model.joint_models],
            device=device,
            dtype=dtype,
        )
        pitches = torch.tensor(
            [float(getattr(joint, "pitch", 0.0)) for joint in model.joint_models],
            device=device,
            dtype=dtype,
        )
        max_nv = max(model.nvs, default=0)
        motion_subspaces = torch.zeros(
            (model.njoints, 6, max_nv), device=device, dtype=dtype
        )
        for index, joint in enumerate(model.joint_models):
            if joint.nv == 0:
                continue
            neutral = joint.neutral().to(device=device, dtype=dtype)
            subspace = joint.joint_motion_subspace(neutral).to(
                device=device, dtype=dtype
            )
            motion_subspaces[index, :, : joint.nv] = subspace
        child_offsets, child_indices = _flatten_rows(model.children)
        subtree_offsets, subtree_indices = _flatten_rows(model.subtrees)
        support_offsets, support_indices = _flatten_rows(model.supports)

        def i32(values: list[int] | tuple[int, ...]) -> torch.Tensor:
            return torch.tensor(values, device=device, dtype=torch.int32)

        result = cls(
            njoints=model.njoints,
            nbodies=model.nbodies,
            nframes=model.nframes,
            nq=model.nq,
            nv=model.nv,
            name=model.name,
            joint_names=model.joint_names,
            body_names=model.body_names,
            frame_names=model.frame_names,
            parents=model.parents,
            children=model.children,
            subtrees=model.subtrees,
            supports=model.supports,
            topo_order=model.topo_order,
            nqs=model.nqs,
            nvs=model.nvs,
            idx_qs=model.idx_qs,
            idx_vs=model.idx_vs,
            joint_models=model.joint_models,
            joint_kind_codes=codes,
            mimic_source=model.mimic_source,
            joint_kind_tensor=torch.tensor(codes, device=device, dtype=torch.int8),
            parents_tensor=i32(model.parents),
            topo_order_tensor=i32(model.topo_order),
            nqs_tensor=i32(model.nqs),
            nvs_tensor=i32(model.nvs),
            idx_qs_tensor=i32(model.idx_qs),
            idx_vs_tensor=i32(model.idx_vs),
            children_offsets=i32(child_offsets),
            children_indices=i32(child_indices),
            subtree_offsets=i32(subtree_offsets),
            subtree_indices=i32(subtree_indices),
            support_offsets=i32(support_offsets),
            support_indices=i32(support_indices),
            frame_parent_joints=i32(tuple(frame.parent_joint for frame in model.frames)),
            mimic_source_tensor=i32(model.mimic_source),
            joint_axes=axes,
            joint_pitches=pitches,
            joint_motion_subspaces=motion_subspaces,
        )
        result.validate_consistency()
        return result

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ModelStructure":
        """Move kernel tables; integer tables never undergo dtype casts."""

        replacements: dict[str, torch.Tensor] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, torch.Tensor):
                continue
            target_dtype = dtype if value.is_floating_point() else value.dtype
            replacements[field.name] = value.to(device=device, dtype=target_dtype)
        return dataclasses.replace(self, **replacements)

    def validate_consistency(self) -> None:
        """Raise ``ValueError`` if static mirrors and device tables disagree."""

        checks = {
            "parents": (self.parents_tensor, self.parents),
            "topo_order": (self.topo_order_tensor, self.topo_order),
            "nqs": (self.nqs_tensor, self.nqs),
            "nvs": (self.nvs_tensor, self.nvs),
            "idx_qs": (self.idx_qs_tensor, self.idx_qs),
            "idx_vs": (self.idx_vs_tensor, self.idx_vs),
            "joint kinds": (self.joint_kind_tensor, self.joint_kind_codes),
            "mimic source": (self.mimic_source_tensor, self.mimic_source),
        }
        for name, (tensor, static) in checks.items():
            if tuple(int(v) for v in tensor.detach().cpu().tolist()) != tuple(static):
                raise ValueError(f"ModelStructure {name} tensor disagrees with static mirror")

        for name, rows, offsets, indices in (
            ("children", self.children, self.children_offsets, self.children_indices),
            ("subtrees", self.subtrees, self.subtree_offsets, self.subtree_indices),
            ("supports", self.supports, self.support_offsets, self.support_indices),
        ):
            flat_offsets, flat_indices = _flatten_rows(rows)
            if tuple(offsets.detach().cpu().tolist()) != tuple(flat_offsets):
                raise ValueError(f"ModelStructure {name} CSR offsets disagree with static mirror")
            if tuple(indices.detach().cpu().tolist()) != tuple(flat_indices):
                raise ValueError(f"ModelStructure {name} CSR indices disagree with static mirror")


__all__ = ["JOINT_KIND_CODES", "ModelStructure"]
