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

from .joint_models import (
    JointFreeFlyer,
    JointHelical,
    JointPlanar,
    JointPrismaticUnaligned,
    JointPX,
    JointPY,
    JointPZ,
    JointRevoluteUnaligned,
    JointRevoluteUnbounded,
    JointRX,
    JointRY,
    JointRZ,
    JointSpherical,
    JointTranslation,
)
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


# Exact classes are intentional. A programmatic/custom JointModel may inherit
# from a built-in class while overriding its manifold semantics; such a joint
# must stay on the per-joint fallback path.
_EUCLIDEAN_MANIFOLD_TYPES = frozenset(
    {
        JointRX,
        JointRY,
        JointRZ,
        JointRevoluteUnaligned,
        JointPX,
        JointPY,
        JointPZ,
        JointPrismaticUnaligned,
        JointTranslation,
        JointHelical,
    }
)


def _matrix_slice_indices(
    joint_ids: tuple[int, ...],
    starts: tuple[int, ...],
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build a ``(len(joint_ids), width)`` device-local index table."""

    indices = [starts[joint_id] + offset for joint_id in joint_ids for offset in range(width)]
    return torch.tensor(indices, device=device, dtype=torch.long).reshape(len(joint_ids), width)


def _flat_slice_indices(
    joint_ids: tuple[int, ...],
    starts: tuple[int, ...],
    widths: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Build CSR-style offsets and flattened indices for variable-width joints."""

    offsets = [0]
    indices: list[int] = []
    for joint_id in joint_ids:
        indices.extend(range(starts[joint_id], starts[joint_id] + widths[joint_id]))
        offsets.append(len(indices))
    return tuple(offsets), torch.tensor(indices, device=device, dtype=torch.long)


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
    nq_full: int
    nv_full: int

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
    nqs_full: tuple[int, ...]
    nvs_full: tuple[int, ...]
    idx_qs_full: tuple[int, ...]
    idx_vs_full: tuple[int, ...]
    joint_models: tuple[JointModel, ...]
    joint_kind_codes: tuple[int, ...]
    mimic_source: tuple[int, ...]
    has_mimic: bool

    # Public-coordinate manifold groups. Mimic targets have zero public width
    # and therefore never appear here; their source joint owns the coordinate.
    manifold_euclidean_joint_ids: tuple[int, ...]
    manifold_spherical_joint_ids: tuple[int, ...]
    manifold_free_flyer_joint_ids: tuple[int, ...]
    manifold_unbounded_joint_ids: tuple[int, ...]
    manifold_planar_joint_ids: tuple[int, ...]
    manifold_fallback_joint_ids: tuple[int, ...]
    manifold_fallback_q_offsets: tuple[int, ...]
    manifold_fallback_v_offsets: tuple[int, ...]

    joint_kind_tensor: torch.Tensor
    parents_tensor: torch.Tensor
    topo_order_tensor: torch.Tensor
    nqs_tensor: torch.Tensor
    nvs_tensor: torch.Tensor
    idx_qs_tensor: torch.Tensor
    idx_vs_tensor: torch.Tensor
    nqs_full_tensor: torch.Tensor
    nvs_full_tensor: torch.Tensor
    idx_qs_full_tensor: torch.Tensor
    idx_vs_full_tensor: torch.Tensor
    children_offsets: torch.Tensor
    children_indices: torch.Tensor
    subtree_offsets: torch.Tensor
    subtree_indices: torch.Tensor
    support_offsets: torch.Tensor
    support_indices: torch.Tensor
    frame_parent_joints: torch.Tensor
    mimic_source_tensor: torch.Tensor
    q_expansion: torch.Tensor
    q_offset: torch.Tensor
    v_expansion: torch.Tensor
    manifold_euclidean_q_indices: torch.Tensor
    manifold_euclidean_v_indices: torch.Tensor
    manifold_spherical_q_indices: torch.Tensor
    manifold_spherical_v_indices: torch.Tensor
    manifold_free_flyer_q_indices: torch.Tensor
    manifold_free_flyer_v_indices: torch.Tensor
    manifold_unbounded_q_indices: torch.Tensor
    manifold_unbounded_v_indices: torch.Tensor
    manifold_planar_q_indices: torch.Tensor
    manifold_planar_v_indices: torch.Tensor
    manifold_fallback_q_indices: torch.Tensor
    manifold_fallback_v_indices: torch.Tensor
    joint_axes: torch.Tensor
    joint_pitches: torch.Tensor
    joint_motion_subspaces: torch.Tensor

    @classmethod
    def from_model(  # noqa: PLR0915 - one audited topology packing boundary
        cls,
        model: "Model",
    ) -> "ModelStructure":
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
        max_nv = max(model.nvs_full, default=0)
        motion_subspaces = torch.zeros((model.njoints, 6, max_nv), device=device, dtype=dtype)
        for index, joint in enumerate(model.joint_models):
            if joint.nv == 0:
                continue
            neutral = joint.neutral().to(device=device, dtype=dtype)
            subspace = joint.joint_motion_subspace(neutral).to(device=device, dtype=dtype)
            motion_subspaces[index, :, : joint.nv] = subspace
        child_offsets, child_indices = _flatten_rows(model.children)
        subtree_offsets, subtree_indices = _flatten_rows(model.subtrees)
        support_offsets, support_indices = _flatten_rows(model.supports)

        manifold_groups: dict[str, list[int]] = {
            "euclidean": [],
            "spherical": [],
            "free_flyer": [],
            "unbounded": [],
            "planar": [],
            "fallback": [],
        }
        for joint_id, joint in enumerate(model.joint_models):
            public_nq = model.nqs[joint_id]
            public_nv = model.nvs[joint_id]
            if public_nq == 0 and public_nv == 0:
                continue

            # Reduced mimic targets have zero public widths. Any other public
            # width mismatch is not a built-in manifold group and must retain
            # the joint object's own dispatch semantics.
            if public_nq != joint.nq or public_nv != joint.nv:
                manifold_groups["fallback"].append(joint_id)
                continue

            joint_type = type(joint)
            if joint_type in _EUCLIDEAN_MANIFOLD_TYPES:
                manifold_groups["euclidean"].append(joint_id)
            elif joint_type is JointSpherical:
                manifold_groups["spherical"].append(joint_id)
            elif joint_type is JointFreeFlyer:
                manifold_groups["free_flyer"].append(joint_id)
            elif joint_type is JointRevoluteUnbounded:
                manifold_groups["unbounded"].append(joint_id)
            elif joint_type is JointPlanar:
                manifold_groups["planar"].append(joint_id)
            else:
                manifold_groups["fallback"].append(joint_id)

        euclidean_ids = tuple(manifold_groups["euclidean"])
        spherical_ids = tuple(manifold_groups["spherical"])
        free_flyer_ids = tuple(manifold_groups["free_flyer"])
        unbounded_ids = tuple(manifold_groups["unbounded"])
        planar_ids = tuple(manifold_groups["planar"])
        fallback_ids = tuple(manifold_groups["fallback"])

        # Euclidean joints may be scalar or 3-DoF translation joints, so they
        # use a flat table. The other built-in groups have fixed q/v widths.
        _, euclidean_q_indices = _flat_slice_indices(euclidean_ids, model.idx_qs, model.nqs, device=device)
        _, euclidean_v_indices = _flat_slice_indices(euclidean_ids, model.idx_vs, model.nvs, device=device)
        spherical_q_indices = _matrix_slice_indices(spherical_ids, model.idx_qs, 4, device=device)
        spherical_v_indices = _matrix_slice_indices(spherical_ids, model.idx_vs, 3, device=device)
        free_flyer_q_indices = _matrix_slice_indices(free_flyer_ids, model.idx_qs, 7, device=device)
        free_flyer_v_indices = _matrix_slice_indices(free_flyer_ids, model.idx_vs, 6, device=device)
        unbounded_q_indices = _matrix_slice_indices(unbounded_ids, model.idx_qs, 2, device=device)
        unbounded_v_indices = _matrix_slice_indices(unbounded_ids, model.idx_vs, 1, device=device)
        planar_q_indices = _matrix_slice_indices(planar_ids, model.idx_qs, 4, device=device)
        planar_v_indices = _matrix_slice_indices(planar_ids, model.idx_vs, 3, device=device)
        fallback_q_offsets, fallback_q_indices = _flat_slice_indices(
            fallback_ids, model.idx_qs, model.nqs, device=device
        )
        fallback_v_offsets, fallback_v_indices = _flat_slice_indices(
            fallback_ids, model.idx_vs, model.nvs, device=device
        )

        def i32(values: list[int] | tuple[int, ...]) -> torch.Tensor:
            return torch.tensor(values, device=device, dtype=torch.int32)

        result = cls(
            njoints=model.njoints,
            nbodies=model.nbodies,
            nframes=model.nframes,
            nq=model.nq,
            nv=model.nv,
            nq_full=model.nq_full,
            nv_full=model.nv_full,
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
            nqs_full=model.nqs_full,
            nvs_full=model.nvs_full,
            idx_qs_full=model.idx_qs_full,
            idx_vs_full=model.idx_vs_full,
            joint_models=model.joint_models,
            joint_kind_codes=codes,
            mimic_source=model.mimic_source,
            has_mimic=model.has_mimic,
            manifold_euclidean_joint_ids=euclidean_ids,
            manifold_spherical_joint_ids=spherical_ids,
            manifold_free_flyer_joint_ids=free_flyer_ids,
            manifold_unbounded_joint_ids=unbounded_ids,
            manifold_planar_joint_ids=planar_ids,
            manifold_fallback_joint_ids=fallback_ids,
            manifold_fallback_q_offsets=fallback_q_offsets,
            manifold_fallback_v_offsets=fallback_v_offsets,
            joint_kind_tensor=torch.tensor(codes, device=device, dtype=torch.int8),
            parents_tensor=i32(model.parents),
            topo_order_tensor=i32(model.topo_order),
            nqs_tensor=i32(model.nqs),
            nvs_tensor=i32(model.nvs),
            idx_qs_tensor=i32(model.idx_qs),
            idx_vs_tensor=i32(model.idx_vs),
            nqs_full_tensor=i32(model.nqs_full),
            nvs_full_tensor=i32(model.nvs_full),
            idx_qs_full_tensor=i32(model.idx_qs_full),
            idx_vs_full_tensor=i32(model.idx_vs_full),
            children_offsets=i32(child_offsets),
            children_indices=i32(child_indices),
            subtree_offsets=i32(subtree_offsets),
            subtree_indices=i32(subtree_indices),
            support_offsets=i32(support_offsets),
            support_indices=i32(support_indices),
            frame_parent_joints=i32(tuple(frame.parent_joint for frame in model.frames)),
            mimic_source_tensor=i32(model.mimic_source),
            q_expansion=model.q_expansion,
            q_offset=model.q_offset,
            v_expansion=model.v_expansion,
            manifold_euclidean_q_indices=euclidean_q_indices,
            manifold_euclidean_v_indices=euclidean_v_indices,
            manifold_spherical_q_indices=spherical_q_indices,
            manifold_spherical_v_indices=spherical_v_indices,
            manifold_free_flyer_q_indices=free_flyer_q_indices,
            manifold_free_flyer_v_indices=free_flyer_v_indices,
            manifold_unbounded_q_indices=unbounded_q_indices,
            manifold_unbounded_v_indices=unbounded_v_indices,
            manifold_planar_q_indices=planar_q_indices,
            manifold_planar_v_indices=planar_v_indices,
            manifold_fallback_q_indices=fallback_q_indices,
            manifold_fallback_v_indices=fallback_v_indices,
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
            "nqs_full": (self.nqs_full_tensor, self.nqs_full),
            "nvs_full": (self.nvs_full_tensor, self.nvs_full),
            "idx_qs_full": (self.idx_qs_full_tensor, self.idx_qs_full),
            "idx_vs_full": (self.idx_vs_full_tensor, self.idx_vs_full),
            "joint kinds": (self.joint_kind_tensor, self.joint_kind_codes),
            "mimic source": (self.mimic_source_tensor, self.mimic_source),
        }
        for name, (tensor, static) in checks.items():
            if tuple(int(v) for v in tensor.detach().cpu().tolist()) != tuple(static):
                raise ValueError(f"ModelStructure {name} tensor disagrees with static mirror")

        q_group_tensors = (
            self.manifold_euclidean_q_indices,
            self.manifold_spherical_q_indices,
            self.manifold_free_flyer_q_indices,
            self.manifold_unbounded_q_indices,
            self.manifold_planar_q_indices,
            self.manifold_fallback_q_indices,
        )
        v_group_tensors = (
            self.manifold_euclidean_v_indices,
            self.manifold_spherical_v_indices,
            self.manifold_free_flyer_v_indices,
            self.manifold_unbounded_v_indices,
            self.manifold_planar_v_indices,
            self.manifold_fallback_v_indices,
        )
        q_grouped = sorted(
            int(index) for tensor in q_group_tensors for index in tensor.detach().cpu().reshape(-1).tolist()
        )
        v_grouped = sorted(
            int(index) for tensor in v_group_tensors for index in tensor.detach().cpu().reshape(-1).tolist()
        )
        if q_grouped != list(range(self.nq)):
            raise ValueError("ModelStructure manifold groups do not partition public q")
        if v_grouped != list(range(self.nv)):
            raise ValueError("ModelStructure manifold groups do not partition public v")
        if len(self.manifold_fallback_q_offsets) != len(self.manifold_fallback_joint_ids) + 1:
            raise ValueError("ModelStructure fallback q offsets disagree with joint ids")
        if len(self.manifold_fallback_v_offsets) != len(self.manifold_fallback_joint_ids) + 1:
            raise ValueError("ModelStructure fallback v offsets disagree with joint ids")

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
