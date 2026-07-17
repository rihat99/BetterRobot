"""``Model`` — frozen kinematic-tree description (Pinocchio-style).

``Model`` is built once, shared across workers/devices, and never mutated.
Every tensor buffer is device/dtype polymorphic via ``.to()``. The static
vs floating-base distinction **disappears**: a floating base is simply
``joint_models[1] = JointFreeFlyer``.

See ``docs/concepts/model_and_data.md §2``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ..exceptions import DeviceMismatchError, DtypeMismatchError, ShapeError
from ..lie import se3, so3
from .frame import Frame
from .joint_models.base import JointModel

if TYPE_CHECKING:
    from .model_structure import ModelStructure
    from .model_values import ModelValues


@dataclass(frozen=True)
class Model:
    """Immutable kinematic-tree description.

    Joint 0 is the universe (``parents[0] == -1``). Every other joint has
    exactly one parent joint. Bodies are 1:1 with joints: ``body[i]`` is the
    body attached to joint ``i`` via ``joint_placements[i]``. A free-flyer
    root is just ``joint_models[1] = JointFreeFlyer``.
    """

    # ──────────── counts ────────────
    njoints: int
    nbodies: int
    nframes: int
    nq: int
    nv: int
    nq_full: int
    nv_full: int

    # ──────────── names & indexing ────────────
    name: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    frame_names: tuple[str, ...]

    joint_name_to_id: dict[str, int]
    body_name_to_id: dict[str, int]
    frame_name_to_id: dict[str, int]

    # ──────────── topology ────────────
    parents: tuple[int, ...]
    children: tuple[tuple[int, ...], ...]
    subtrees: tuple[tuple[int, ...], ...]
    supports: tuple[tuple[int, ...], ...]
    topo_order: tuple[int, ...]

    # ──────────── per-joint dispatch tables ────────────
    joint_models: tuple[JointModel, ...]
    nqs: tuple[int, ...]
    nvs: tuple[int, ...]
    idx_qs: tuple[int, ...]
    idx_vs: tuple[int, ...]
    nqs_full: tuple[int, ...]
    nvs_full: tuple[int, ...]
    idx_qs_full: tuple[int, ...]
    idx_vs_full: tuple[int, ...]

    # ──────────── device-resident tensors ────────────
    joint_placements: torch.Tensor  # (njoints, 7)
    body_inertias: torch.Tensor  # (nbodies, 10)
    lower_pos_limit: torch.Tensor  # (nq,)
    upper_pos_limit: torch.Tensor  # (nq,)
    velocity_limit: torch.Tensor  # (nv,)
    effort_limit: torch.Tensor  # (nv,)
    rotor_inertia: torch.Tensor  # (nv,)
    armature: torch.Tensor  # (nv,)
    friction: torch.Tensor  # (nv,)
    damping: torch.Tensor  # (nv,)
    gravity: torch.Tensor  # (6,)

    # ──────────── mimic relationship (PyRoki gather trick) ────────────
    mimic_multiplier: torch.Tensor  # (njoints,)
    mimic_offset: torch.Tensor  # (njoints,)
    mimic_source: tuple[int, ...]  # (njoints,) — src index, self-idx otherwise
    q_expansion: torch.Tensor  # (nq_full, nq)
    q_offset: torch.Tensor  # (nq_full,)
    v_expansion: torch.Tensor  # (nv_full, nv)
    has_mimic: bool

    # ──────────── frames ────────────
    frames: tuple[Frame, ...]

    # ──────────── reference configurations ────────────
    reference_configurations: dict[str, torch.Tensor] = field(default_factory=dict)
    q_neutral: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    # ──────────── optional back-references (non-tensor, not moved by .to()) ────
    meta: dict = field(default_factory=dict)

    # Canonical compute seam, derived from the compatibility fields above.
    # ``init=False`` makes dataclasses.replace rebuild both views instead of
    # accidentally carrying stale aliases after a value change.
    structure: "ModelStructure" = field(init=False, repr=False, compare=False)
    values: "ModelValues" = field(init=False, repr=False, compare=False)

    # ────────────────────────── methods ──────────────────────────

    def __post_init__(self) -> None:
        from .model_structure import ModelStructure  # noqa: PLC0415 - import cycle
        from .model_values import ModelValues  # noqa: PLC0415 - import cycle

        object.__setattr__(self, "structure", ModelStructure.from_model(self))
        object.__setattr__(self, "values", ModelValues.from_model(self))

    def _shallow_rebind(
        self,
        *,
        structure: "ModelStructure",
        values: "ModelValues",
        field_updates: dict[str, object] | None = None,
    ) -> "Model":
        """Copy immutable metadata without rebuilding topology.

        ``dataclasses.replace`` intentionally calls ``__post_init__`` and is
        therefore unsuitable for the public value-rebind hot path.
        """

        updates = field_updates or {}
        result = object.__new__(type(self))
        for model_field in dataclasses.fields(self):
            if model_field.name == "structure":
                value = structure
            elif model_field.name == "values":
                value = values
            else:
                value = updates.get(model_field.name, getattr(self, model_field.name))
            object.__setattr__(result, model_field.name, value)
        return result

    def with_values(
        self,
        *,
        joint_placements: torch.Tensor | None = None,
        body_inertias: torch.Tensor | None = None,
        frame_placements: torch.Tensor | None = None,
    ) -> "Model":
        """Pair this topology with new differentiable, optionally batched values.

        Leading dimensions follow torch's right-aligned broadcast rules.  For
        a ``(B, T, nq)`` trajectory and per-person values, pass value tables
        with an explicit singleton time axis, for example
        ``(B, 1, njoints, 7)``.
        """

        def checked(
            name: str,
            value: torch.Tensor | None,
            current: torch.Tensor,
            event_shape: tuple[int, int],
            *,
            normalize_pose: bool = False,
        ) -> torch.Tensor:
            if value is None:
                return current
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor or None")
            if value.ndim < 2 or tuple(value.shape[-2:]) != event_shape:
                raise ShapeError(f"{name} has shape {tuple(value.shape)}; expected trailing event shape {event_shape}")
            if not value.is_floating_point():
                raise DtypeMismatchError(f"{name}.dtype={value.dtype} is unsupported; use a floating dtype")
            if value.device != self.values.joint_placements.device:
                raise DeviceMismatchError(
                    f"{name}.device={value.device} != model.device="
                    f"{self.values.joint_placements.device}; move the value or "
                    "call model.to(...) first"
                )
            if value.dtype != self.values.joint_placements.dtype:
                raise DtypeMismatchError(
                    f"{name}.dtype={value.dtype} != model.dtype={self.values.joint_placements.dtype}"
                )
            return se3.normalize(value) if normalize_pose else value

        placements = checked(
            "joint_placements",
            joint_placements,
            self.values.joint_placements,
            (self.njoints, 7),
            normalize_pose=True,
        )
        inertias = checked(
            "body_inertias",
            body_inertias,
            self.values.body_inertias,
            (self.nbodies, 10),
        )
        frame_values = checked(
            "frame_placements",
            frame_placements,
            self.values.frame_placements,
            (self.nframes, 7),
            normalize_pose=True,
        )
        inertia_cache = self.values.body_inertias_6x6 if inertias is self.values.body_inertias else None
        rebound = dataclasses.replace(
            self.values,
            joint_placements=placements,
            body_inertias=inertias,
            frame_placements=frame_values,
            body_inertias_6x6=inertia_cache,
        )
        # Validate value-to-value broadcasting now; the query batch is added
        # at evaluation time.
        rebound.execution_batch_shape(self.structure, self.values.q_neutral)
        return self._shallow_rebind(
            structure=self.structure,
            values=rebound,
            field_updates={
                "joint_placements": placements,
                "body_inertias": inertias,
            },
        )

    def to(self, device=None, dtype=None) -> "Model":
        """Return a new ``Model`` with every tensor buffer moved to the given
        device and/or dtype. Topology / names / joint models are shared by
        reference (they are immutable).
        """
        structure = self.structure.to(device=device, dtype=dtype)
        values = self.values.to(device=device, dtype=dtype)

        def _t(tensor: torch.Tensor) -> torch.Tensor:
            target_dtype = dtype if tensor.is_floating_point() else tensor.dtype
            return tensor.to(device=device, dtype=target_dtype)

        field_updates: dict[str, object] = {
            name: getattr(values, name)
            for name in (
                "joint_placements",
                "body_inertias",
                "lower_pos_limit",
                "upper_pos_limit",
                "velocity_limit",
                "effort_limit",
                "rotor_inertia",
                "armature",
                "friction",
                "damping",
                "gravity",
                "mimic_multiplier",
                "mimic_offset",
                "q_neutral",
            )
        }
        field_updates.update(
            {
                "q_expansion": structure.q_expansion,
                "q_offset": structure.q_offset,
                "v_expansion": structure.v_expansion,
            }
        )
        field_updates["frames"] = tuple(
            dataclasses.replace(frame, joint_placement=_t(frame.joint_placement)) for frame in self.frames
        )
        field_updates["reference_configurations"] = {
            name: _t(value) for name, value in self.reference_configurations.items()
        }
        return self._shallow_rebind(
            structure=structure,
            values=values,
            field_updates=field_updates,
        )

    def create_data(
        self,
        *,
        batch_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):  # -> "Data"
        """Allocate an empty ``Data`` workspace shaped for this model."""
        from .data import Data  # noqa: PLC0415 - keep Data out of the model import cycle

        _device = device or self.joint_placements.device
        _dtype = dtype or self.joint_placements.dtype
        q = torch.zeros(*batch_shape, self.nq, device=_device, dtype=_dtype)
        return Data(q=q)

    def joint_id(self, name: str) -> int:
        """Return the integer id of the named joint. Raises ``KeyError`` if missing."""
        return self.joint_name_to_id[name]

    def q_permutation(
        self,
        other_joint_order: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return q/v gather indices from an external joint-slice order.

        The external vectors must concatenate the same public per-joint
        ``nqs``/``nvs`` slices as this model, in ``other_joint_order``. Names
        for zero-DOF joints may be omitted. The returned tensors make the
        remap one batched-safe trailing-dimension gather::

            perm_q, perm_v = model.q_permutation(external_joint_names)
            q_model = q_external[..., perm_q]
            v_model = v_external[..., perm_v]
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

        device = self.joint_placements.device
        return (
            torch.tensor(perm_q, dtype=torch.long, device=device),
            torch.tensor(perm_v, dtype=torch.long, device=device),
        )

    def frame_id(self, name: str) -> int:
        """Return the integer id of the named frame."""
        return self.frame_name_to_id[name]

    def body_id(self, name: str) -> int:
        """Return the integer id of the named body."""
        return self.body_name_to_id[name]

    def body_inertia(self, body_id: int):
        """Typed accessor returning a single body's :class:`~better_robot.spatial.Inertia`."""
        from ..spatial.inertia import Inertia  # noqa: PLC0415 - typed lazy accessor

        return Inertia(self.values.body_inertias[..., body_id, :])

    def get_subtree(self, joint_id: int) -> tuple[int, ...]:
        """Return the subtree rooted at ``joint_id``."""
        return self.subtrees[joint_id]

    def get_support(self, joint_id: int) -> tuple[int, ...]:
        """Return the joint chain from joint 0 to ``joint_id``."""
        return self.supports[joint_id]

    def _validate_manifold_tensor(
        self,
        name: str,
        value: torch.Tensor,
        trailing_width: int,
    ) -> None:
        if value.ndim < 1 or value.shape[-1] != trailing_width:
            raise ShapeError(f"{name} has shape {tuple(value.shape)}; expected trailing dimension {trailing_width}")
        model_device = self.structure.idx_qs_tensor.device
        if value.device != model_device:
            raise DeviceMismatchError(
                f"{name}.device={value.device} != model.device={model_device}; "
                "move the tensor or call model.to(...) first"
            )

    def integrate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Universal manifold retraction ``q ⊕ v`` grouped by joint kind.

        Leading dimensions use torch's right-aligned broadcast rules. Built-in
        manifolds execute once per semantic kind; custom and composite joints
        retain exact per-joint dispatch through the precomputed fallback group.
        """
        self._validate_manifold_tensor("q", q, self.nq)
        self._validate_manifold_tensor("v", v, self.nv)
        batch_shape = torch.broadcast_shapes(q.shape[:-1], v.shape[:-1])
        result_dtype = torch.promote_types(q.dtype, v.dtype)
        q_broadcast = q.to(dtype=result_dtype).expand(*batch_shape, self.nq)
        v_broadcast = v.to(dtype=result_dtype).expand(*batch_shape, self.nv)
        if self.nq == 0:
            return q_broadcast.clone()

        result = q_broadcast.clone()
        structure = self.structure

        q_indices = structure.manifold_euclidean_q_indices
        if q_indices.numel():
            values = q_broadcast[..., q_indices] + v_broadcast[..., structure.manifold_euclidean_v_indices]
            result = result.index_copy(-1, q_indices, values)

        q_indices = structure.manifold_spherical_q_indices
        if q_indices.numel():
            q_group = q_broadcast[..., q_indices]
            v_group = v_broadcast[..., structure.manifold_spherical_v_indices]
            values = so3.normalize(so3.compose(q_group, so3.exp(v_group)))
            result = result.index_copy(-1, q_indices.reshape(-1), values.flatten(start_dim=-2))

        q_indices = structure.manifold_free_flyer_q_indices
        if q_indices.numel():
            q_group = q_broadcast[..., q_indices]
            v_group = v_broadcast[..., structure.manifold_free_flyer_v_indices]
            values = se3.normalize(se3.compose(q_group, se3.exp(v_group)))
            result = result.index_copy(-1, q_indices.reshape(-1), values.flatten(start_dim=-2))

        q_indices = structure.manifold_unbounded_q_indices
        if q_indices.numel():
            q_group = q_broadcast[..., q_indices]
            v_group = v_broadcast[..., structure.manifold_unbounded_v_indices]
            theta = torch.atan2(q_group[..., 1], q_group[..., 0]) + v_group[..., 0]
            values = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
            result = result.index_copy(-1, q_indices.reshape(-1), values.flatten(start_dim=-2))

        q_indices = structure.manifold_planar_q_indices
        if q_indices.numel():
            q_group = q_broadcast[..., q_indices]
            v_group = v_broadcast[..., structure.manifold_planar_v_indices]
            theta = torch.atan2(q_group[..., 3], q_group[..., 2]) + v_group[..., 2]
            values = torch.stack(
                (
                    q_group[..., 0] + v_group[..., 0],
                    q_group[..., 1] + v_group[..., 1],
                    torch.cos(theta),
                    torch.sin(theta),
                ),
                dim=-1,
            )
            result = result.index_copy(-1, q_indices.reshape(-1), values.flatten(start_dim=-2))

        for fallback_index, joint_id in enumerate(structure.manifold_fallback_joint_ids):
            q_start = structure.manifold_fallback_q_offsets[fallback_index]
            q_stop = structure.manifold_fallback_q_offsets[fallback_index + 1]
            v_start = structure.manifold_fallback_v_offsets[fallback_index]
            v_stop = structure.manifold_fallback_v_offsets[fallback_index + 1]
            q_indices = structure.manifold_fallback_q_indices[q_start:q_stop]
            v_indices = structure.manifold_fallback_v_indices[v_start:v_stop]
            values = self.joint_models[joint_id].integrate(
                q_broadcast[..., q_indices],
                v_broadcast[..., v_indices],
            )
            result = result.index_copy(-1, q_indices, values)

        return result

    def difference(  # noqa: PLR0915 - explicit manifold formulas stay auditable
        self,
        q0: torch.Tensor,
        q1: torch.Tensor,
    ) -> torch.Tensor:
        """Universal tangent ``q1 ⊖ q0`` grouped by joint kind."""
        self._validate_manifold_tensor("q0", q0, self.nq)
        self._validate_manifold_tensor("q1", q1, self.nq)
        batch_shape = torch.broadcast_shapes(q0.shape[:-1], q1.shape[:-1])
        result_dtype = torch.promote_types(q0.dtype, q1.dtype)
        q0_broadcast = q0.to(dtype=result_dtype).expand(*batch_shape, self.nq)
        q1_broadcast = q1.to(dtype=result_dtype).expand(*batch_shape, self.nq)
        if self.nv == 0:
            return q0_broadcast.new_zeros(*batch_shape, self.nv)

        result = q0_broadcast.new_zeros(*batch_shape, self.nv)
        structure = self.structure

        v_indices = structure.manifold_euclidean_v_indices
        if v_indices.numel():
            values = (
                q1_broadcast[..., structure.manifold_euclidean_q_indices]
                - q0_broadcast[..., structure.manifold_euclidean_q_indices]
            )
            result = result.index_copy(-1, v_indices, values)

        q_indices = structure.manifold_spherical_q_indices
        if q_indices.numel():
            q0_group = q0_broadcast[..., q_indices]
            q1_group = q1_broadcast[..., q_indices]
            delta = so3.compose(so3.inverse(q0_group), q1_group)
            values = so3.log(so3.normalize(delta))
            v_indices = structure.manifold_spherical_v_indices
            result = result.index_copy(-1, v_indices.reshape(-1), values.flatten(start_dim=-2))

        q_indices = structure.manifold_free_flyer_q_indices
        if q_indices.numel():
            q0_group = q0_broadcast[..., q_indices]
            q1_group = q1_broadcast[..., q_indices]
            values = se3.log(se3.compose(se3.inverse(q0_group), q1_group))
            v_indices = structure.manifold_free_flyer_v_indices
            result = result.index_copy(-1, v_indices.reshape(-1), values.flatten(start_dim=-2))

        q_indices = structure.manifold_unbounded_q_indices
        if q_indices.numel():
            q0_group = q0_broadcast[..., q_indices]
            q1_group = q1_broadcast[..., q_indices]
            theta0 = torch.atan2(q0_group[..., 1], q0_group[..., 0])
            theta1 = torch.atan2(q1_group[..., 1], q1_group[..., 0])
            values = (theta1 - theta0).unsqueeze(-1)
            v_indices = structure.manifold_unbounded_v_indices
            result = result.index_copy(-1, v_indices.reshape(-1), values.flatten(start_dim=-2))

        q_indices = structure.manifold_planar_q_indices
        if q_indices.numel():
            q0_group = q0_broadcast[..., q_indices]
            q1_group = q1_broadcast[..., q_indices]
            theta0 = torch.atan2(q0_group[..., 3], q0_group[..., 2])
            theta1 = torch.atan2(q1_group[..., 3], q1_group[..., 2])
            values = torch.stack(
                (
                    q1_group[..., 0] - q0_group[..., 0],
                    q1_group[..., 1] - q0_group[..., 1],
                    theta1 - theta0,
                ),
                dim=-1,
            )
            v_indices = structure.manifold_planar_v_indices
            result = result.index_copy(-1, v_indices.reshape(-1), values.flatten(start_dim=-2))

        for fallback_index, joint_id in enumerate(structure.manifold_fallback_joint_ids):
            q_start = structure.manifold_fallback_q_offsets[fallback_index]
            q_stop = structure.manifold_fallback_q_offsets[fallback_index + 1]
            v_start = structure.manifold_fallback_v_offsets[fallback_index]
            v_stop = structure.manifold_fallback_v_offsets[fallback_index + 1]
            q_indices = structure.manifold_fallback_q_indices[q_start:q_stop]
            v_indices = structure.manifold_fallback_v_indices[v_start:v_stop]
            values = self.joint_models[joint_id].difference(
                q0_broadcast[..., q_indices],
                q1_broadcast[..., q_indices],
            )
            result = result.index_copy(-1, v_indices, values)

        return result

    def random_configuration(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return a random valid configuration ``q`` of shape ``(nq,)``."""
        parts: list[torch.Tensor] = []
        for j in range(self.njoints):
            jm = self.joint_models[j]
            nq_j = self.nqs[j]
            if nq_j == 0:
                continue
            iq = self.idx_qs[j]
            lower = self.lower_pos_limit[iq : iq + nq_j]
            upper = self.upper_pos_limit[iq : iq + nq_j]
            parts.append(jm.random_configuration(generator, lower, upper))
        if not parts:
            return torch.zeros(self.nq)
        return torch.cat(parts, dim=-1)
