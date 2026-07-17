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
from ..lie import se3
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
        from .model_structure import ModelStructure
        from .model_values import ModelValues

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
        from .data import Data

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
        from ..spatial.inertia import Inertia

        return Inertia(self.values.body_inertias[..., body_id, :])

    def get_subtree(self, joint_id: int) -> tuple[int, ...]:
        """Return the subtree rooted at ``joint_id``."""
        return self.subtrees[joint_id]

    def get_support(self, joint_id: int) -> tuple[int, ...]:
        """Return the joint chain from joint 0 to ``joint_id``."""
        return self.supports[joint_id]

    def integrate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Universal manifold retraction ``q ⊕ v``. Dispatches per joint."""
        parts: list[torch.Tensor] = []
        for j in range(self.njoints):
            jm = self.joint_models[j]
            if jm.nq == 0:
                continue
            iq = self.idx_qs[j]
            iv = self.idx_vs[j]
            qj = q[..., iq : iq + jm.nq]
            vj = v[..., iv : iv + jm.nv]
            parts.append(jm.integrate(qj, vj))
        if not parts:
            return q.clone()
        return torch.cat(parts, dim=-1)

    def difference(self, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        """Universal tangent ``q1 ⊖ q0``. Dispatches per joint."""
        parts: list[torch.Tensor] = []
        for j in range(self.njoints):
            jm = self.joint_models[j]
            if jm.nq == 0:
                continue
            iq = self.idx_qs[j]
            q0j = q0[..., iq : iq + jm.nq]
            q1j = q1[..., iq : iq + jm.nq]
            parts.append(jm.difference(q0j, q1j))
        if not parts:
            return torch.zeros(*q0.shape[:-1], self.nv, device=q0.device, dtype=q0.dtype)
        return torch.cat(parts, dim=-1)

    def random_configuration(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return a random valid configuration ``q`` of shape ``(nq,)``."""
        parts: list[torch.Tensor] = []
        for j in range(self.njoints):
            jm = self.joint_models[j]
            if jm.nq == 0:
                continue
            iq = self.idx_qs[j]
            lower = self.lower_pos_limit[iq : iq + jm.nq]
            upper = self.upper_pos_limit[iq : iq + jm.nq]
            parts.append(jm.random_configuration(generator, lower, upper))
        if not parts:
            return torch.zeros(self.nq)
        return torch.cat(parts, dim=-1)
