"""Frozen robot model composed from topology and differentiable values."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from .._validation import check_tensor
from ..lie import se3
from .frame import Frame
from .joint_models.base import JointModel

if TYPE_CHECKING:
    from ..spatial.inertia import Inertia
    from .data import Data
    from .model_structure import ModelStructure
    from .model_values import ModelValues


def _checked_value(
    name: str,
    value: torch.Tensor | None,
    current: torch.Tensor,
    event_shape: tuple[int, ...],
    exemplar: torch.Tensor,
    *,
    normalize_pose: bool = False,
) -> torch.Tensor:
    if value is None:
        return current
    tensor = check_tensor(
        name,
        value,
        shape=event_shape,
        floating=True,
        dtype=exemplar.dtype,
        device=exemplar.device,
    )
    return se3.normalize(tensor) if normalize_pose else tensor


def _move_tensor(
    tensor: torch.Tensor,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    target_dtype = dtype if tensor.is_floating_point() else tensor.dtype
    return tensor.to(device=device, dtype=target_dtype)


@dataclass(frozen=True)
class Model:
    """Shallowly frozen robot description with one owner per datum.

    The four stored fields wrap immutable topology and differentiable values.
    Explicit properties preserve flat access without copying either part.
    Contained tensors and dictionaries must be treated as read-only.
    """

    structure: ModelStructure
    values: ModelValues
    reference_configurations: dict[str, torch.Tensor] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values.validate(self.structure)

    # These deliberately compact one-line properties keep the compatibility
    # facade explicit and typed without obscuring the four stored fields.
    # fmt: off
    # Counts and names.
    @property
    def njoints(self) -> int: return self.structure.njoints
    @property
    def nbodies(self) -> int: return self.structure.nbodies
    @property
    def nframes(self) -> int: return self.structure.nframes
    @property
    def nq(self) -> int: return self.structure.nq
    @property
    def nv(self) -> int: return self.structure.nv
    @property
    def nq_full(self) -> int: return self.structure.nq_full
    @property
    def nv_full(self) -> int: return self.structure.nv_full
    @property
    def name(self) -> str: return self.structure.name
    @property
    def joint_names(self) -> tuple[str, ...]: return self.structure.joint_names
    @property
    def body_names(self) -> tuple[str, ...]: return self.structure.body_names
    @property
    def frame_names(self) -> tuple[str, ...]: return self.structure.frame_names
    @property
    def joint_name_to_id(self) -> dict[str, int]: return self.structure.joint_name_to_id
    @property
    def body_name_to_id(self) -> dict[str, int]: return self.structure.body_name_to_id
    @property
    def frame_name_to_id(self) -> dict[str, int]: return self.structure.frame_name_to_id

    # Topology and indexing.
    @property
    def parents(self) -> tuple[int, ...]: return self.structure.parents
    @property
    def children(self) -> tuple[tuple[int, ...], ...]: return self.structure.children
    @property
    def subtrees(self) -> tuple[tuple[int, ...], ...]: return self.structure.subtrees
    @property
    def supports(self) -> tuple[tuple[int, ...], ...]: return self.structure.supports
    @property
    def topo_order(self) -> tuple[int, ...]: return self.structure.topo_order
    @property
    def joint_models(self) -> tuple[JointModel, ...]: return self.structure.joint_models
    @property
    def nqs(self) -> tuple[int, ...]: return self.structure.nqs
    @property
    def nvs(self) -> tuple[int, ...]: return self.structure.nvs
    @property
    def idx_qs(self) -> tuple[int, ...]: return self.structure.idx_qs
    @property
    def idx_vs(self) -> tuple[int, ...]: return self.structure.idx_vs
    @property
    def nqs_full(self) -> tuple[int, ...]: return self.structure.nqs_full
    @property
    def nvs_full(self) -> tuple[int, ...]: return self.structure.nvs_full
    @property
    def idx_qs_full(self) -> tuple[int, ...]: return self.structure.idx_qs_full
    @property
    def idx_vs_full(self) -> tuple[int, ...]: return self.structure.idx_vs_full

    # Differentiable values.
    @property
    def joint_placements(self) -> torch.Tensor: return self.values.joint_placements
    @property
    def body_inertias(self) -> torch.Tensor: return self.values.body_inertias
    @property
    def lower_pos_limit(self) -> torch.Tensor: return self.values.lower_pos_limit
    @property
    def upper_pos_limit(self) -> torch.Tensor: return self.values.upper_pos_limit
    @property
    def velocity_limit(self) -> torch.Tensor: return self.values.velocity_limit
    @property
    def effort_limit(self) -> torch.Tensor: return self.values.effort_limit
    @property
    def rotor_inertia(self) -> torch.Tensor: return self.values.rotor_inertia
    @property
    def armature(self) -> torch.Tensor: return self.values.armature
    @property
    def friction(self) -> torch.Tensor: return self.values.friction
    @property
    def damping(self) -> torch.Tensor: return self.values.damping
    @property
    def gravity(self) -> torch.Tensor: return self.values.gravity
    @property
    def mimic_multiplier(self) -> torch.Tensor: return self.values.mimic_multiplier
    @property
    def mimic_offset(self) -> torch.Tensor: return self.values.mimic_offset
    @property
    def q_neutral(self) -> torch.Tensor: return self.values.q_neutral

    # Reduced-coordinate topology.
    @property
    def mimic_source(self) -> tuple[int, ...]: return self.structure.mimic_source
    @property
    def q_expansion(self) -> torch.Tensor: return self.structure.q_expansion
    @property
    def q_offset(self) -> torch.Tensor: return self.structure.q_offset
    @property
    def v_expansion(self) -> torch.Tensor: return self.structure.v_expansion
    @property
    def has_mimic(self) -> bool: return self.structure.has_mimic
    # fmt: on

    @property
    def frames(self) -> tuple[Frame, ...]:
        """Return frame metadata paired with the current placement values."""

        static_frames = zip(
            self.structure.frame_names,
            self.structure.frame_parent_joint_ids,
            self.structure.frame_types,
            strict=True,
        )
        return tuple(
            Frame(name, parent_joint, self.values.frame_placements[..., frame_id, :], frame_type)
            for frame_id, (name, parent_joint, frame_type) in enumerate(static_frames)
        )

    def joint_id(self, name: str) -> int:
        """Return the integer id of the named joint."""
        return self.structure.joint_id(name)

    def frame_id(self, name: str) -> int:
        """Return the integer id of the named frame."""
        return self.structure.frame_id(name)

    def body_id(self, name: str) -> int:
        """Return the integer id of the named body."""
        return self.structure.body_id(name)

    def get_subtree(self, joint_id: int) -> tuple[int, ...]:
        """Return the subtree rooted at ``joint_id``."""
        return self.structure.get_subtree(joint_id)

    def get_support(self, joint_id: int) -> tuple[int, ...]:
        """Return the joint chain from joint 0 to ``joint_id``."""
        return self.structure.get_support(joint_id)

    def q_permutation(self, other_joint_order: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return q/v gather indices from an external joint-slice order."""
        return self.structure.q_permutation(other_joint_order)

    def body_inertia(self, body_id: int) -> Inertia:
        """Return the packed inertia of one body as a typed value."""
        return self.values.body_inertia(body_id)

    def spatial_inertias(self) -> torch.Tensor:
        """Expand the current packed body inertias to spatial matrices."""
        return self.values.spatial_inertias()

    def integrate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute the universal manifold retraction ``q ⊕ v``."""
        return self.structure.integrate(q, v)

    def difference(self, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        """Compute the universal tangent ``q1 ⊖ q0``."""
        return self.structure.difference(q0, q1)

    def random_configuration(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return a random valid configuration of shape ``(nq,)``."""
        parts: list[torch.Tensor] = []
        for joint_id in range(self.njoints):
            joint_nq = self.nqs[joint_id]
            if joint_nq == 0:
                continue
            q_index = self.idx_qs[joint_id]
            lower = self.lower_pos_limit[q_index : q_index + joint_nq]
            upper = self.upper_pos_limit[q_index : q_index + joint_nq]
            parts.append(self.joint_models[joint_id].random_configuration(generator, lower, upper))
        if not parts:
            return torch.zeros(self.nq)
        return torch.cat(parts, dim=-1)

    def create_data(
        self,
        *,
        batch_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Data:
        """Allocate an empty query workspace shaped for this model."""
        from .data import Data  # noqa: PLC0415 - avoid the model/data import cycle

        target_device = device or self.joint_placements.device
        target_dtype = dtype or self.joint_placements.dtype
        return Data(q=torch.zeros(*batch_shape, self.nq, device=target_device, dtype=target_dtype))

    def with_values(
        self,
        *,
        joint_placements: torch.Tensor | None = None,
        body_inertias: torch.Tensor | None = None,
        frame_placements: torch.Tensor | None = None,
    ) -> Model:
        """Pair this topology with checked, optionally batched values."""
        current = self.values
        exemplar = current.joint_placements
        rebound = replace(
            current,
            joint_placements=_checked_value(
                "joint_placements",
                joint_placements,
                current.joint_placements,
                (self.njoints, 7),
                exemplar,
                normalize_pose=True,
            ),
            body_inertias=_checked_value(
                "body_inertias",
                body_inertias,
                current.body_inertias,
                (self.nbodies, 10),
                exemplar,
            ),
            frame_placements=_checked_value(
                "frame_placements",
                frame_placements,
                current.frame_placements,
                (self.nframes, 7),
                exemplar,
                normalize_pose=True,
            ),
        )
        rebound._execution_batch_shape(rebound.q_neutral)
        return replace(self, values=rebound)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Model:
        """Return a model whose tensor buffers live on ``device``/``dtype``."""
        references = {name: _move_tensor(value, device, dtype) for name, value in self.reference_configurations.items()}
        return replace(
            self,
            structure=self.structure.to(device=device, dtype=dtype),
            values=self.values.to(device=device, dtype=dtype),
            reference_configurations=references,
        )


__all__ = ["Model"]
