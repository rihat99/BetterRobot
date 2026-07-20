"""Frozen topology shared by the torch and Warp compute lanes.

``ModelStructure`` deliberately stores the robot topology twice: Python
tuples keep the torch lane statically unrollable under ``torch.compile``;
flat tensors expose the same data to device kernels.  Construction is the
only place where the two representations are allowed to diverge, and
``validate_consistency`` makes that boundary explicit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from .frame import FrameType
from .joint_models.base import JointModel


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
    joint_name_to_id: dict[str, int]
    body_name_to_id: dict[str, int]
    frame_name_to_id: dict[str, int]
    frame_parent_joint_ids: tuple[int, ...]
    frame_types: tuple[FrameType, ...]

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

    def __post_init__(self) -> None:
        self.validate_consistency()

    def joint_id(self, name: str) -> int:
        """Return the integer id of the named joint."""

        return self.joint_name_to_id[name]

    def frame_id(self, name: str) -> int:
        """Return the integer id of the named frame."""

        return self.frame_name_to_id[name]

    def body_id(self, name: str) -> int:
        """Return the integer id of the named body."""

        return self.body_name_to_id[name]

    def get_subtree(self, joint_id: int) -> tuple[int, ...]:
        """Return the subtree rooted at ``joint_id``."""

        return self.subtrees[joint_id]

    def get_support(self, joint_id: int) -> tuple[int, ...]:
        """Return the joint chain from joint 0 to ``joint_id``."""

        return self.supports[joint_id]

    def q_permutation(
        self,
        other_joint_order: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return q/v gather indices from an external joint-slice order.

        The external vectors must concatenate the same public per-joint
        ``nqs``/``nvs`` slices as this structure, in ``other_joint_order``.
        Names for zero-DOF joints may be omitted. The returned tensors make
        the remap a batched-safe gather over the trailing dimension.
        """

        order = tuple(other_joint_order)
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in order:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(f"other_joint_order contains duplicate joint names: {duplicates}")

        unknown = [name for name in order if name not in self.joint_name_to_id]
        if unknown:
            raise ValueError(f"other_joint_order contains unknown joint names: {unknown}")

        provided = set(order)
        missing = [
            name
            for joint_id, name in enumerate(self.joint_names)
            if (self.nqs[joint_id] > 0 or self.nvs[joint_id] > 0) and name not in provided
        ]
        if missing:
            raise ValueError(f"other_joint_order is missing joints with public q/v slices: {missing}")

        q_starts: dict[str, int] = {}
        v_starts: dict[str, int] = {}
        q_offset = 0
        v_offset = 0
        for name in order:
            joint_id = self.joint_name_to_id[name]
            q_starts[name] = q_offset
            v_starts[name] = v_offset
            q_offset += self.nqs[joint_id]
            v_offset += self.nvs[joint_id]

        perm_q: list[int] = []
        perm_v: list[int] = []
        for joint_id, name in enumerate(self.joint_names):
            nq_joint = self.nqs[joint_id]
            nv_joint = self.nvs[joint_id]
            if nq_joint:
                start = q_starts[name]
                perm_q.extend(range(start, start + nq_joint))
            if nv_joint:
                start = v_starts[name]
                perm_v.extend(range(start, start + nv_joint))

        device = self.idx_qs_tensor.device
        return (
            torch.tensor(perm_q, dtype=torch.long, device=device),
            torch.tensor(perm_v, dtype=torch.long, device=device),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ModelStructure":
        """Move kernel tables; integer tables never undergo dtype casts."""

        # Moving an already-validated table cannot change its contents. A
        # trusted copy also keeps ``to("meta")`` valid and avoids a CUDA-to-CPU
        # consistency scan after every device transfer.
        result = object.__new__(type(self))
        for structure_field in dataclasses.fields(self):
            value = getattr(self, structure_field.name)
            if isinstance(value, torch.Tensor):
                target_dtype = dtype if value.is_floating_point() else value.dtype
                value = value.to(device=device, dtype=target_dtype)
            object.__setattr__(result, structure_field.name, value)
        return result

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
            "frame parents": (self.frame_parent_joints, self.frame_parent_joint_ids),
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
