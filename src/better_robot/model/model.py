"""``Model``: one flat frozen record of a robot (decision records 0009, 0010, 0011)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils import _pytree

from ..exceptions import ArgumentError, ModelError, ShapeError, UnknownNameError
from ..lie import se3
from ._checks import check_tensor
from .joints.base import JointModel

_TENSOR_FIELDS = (
    "joint_placements",
    "body_inertias",
    "frame_placements",
    "lower_position_limit",
    "upper_position_limit",
    "velocity_limit",
    "effort_limit",
    "armature",
    "damping",
    "gravity",
    "q_neutral",
)
_BATCHED_FIELDS = ("joint_placements", "body_inertias", "frame_placements")


@dataclass(frozen=True, eq=False, repr=False)
class Model:
    """A robot: static topology plus the tensors that parametrize it. Build one with ``better_robot.load``.

    Static fields (plain Python, hashed into ``structure_key``) and tensor fields (differentiable, moved by ``to``,
    replaced by ``with_values``); ``I...`` is an optional batch of robots sharing one topology:

    | field | kind | shape / type |
    |---|---|---|
    | ``name`` | static | ``str`` |
    | ``joint_names``, ``body_names``, ``frame_names`` | static | ``tuple[str, ...]`` |
    | ``joints`` | static | ``tuple[JointModel, ...]``, joint 0 is the universe |
    | ``parents`` | static | ``tuple[int, ...]``, ``parents[0] == -1``, ``parents[i] < i`` |
    | ``frame_parent_joint_ids`` | static | ``tuple[int, ...]`` |
    | ``nqs``, ``nvs``, ``idx_qs``, ``idx_vs`` | static | public width and start of each joint's ``q``/``v`` slice |
    | ``nqs_full``, ``nvs_full``, ``idx_qs_full``, ``idx_vs_full`` | static | the same over every joint's coordinates |
    | ``mimic_source_ids`` | static | public joint driving each joint (itself when not a mimic target) |
    | ``mimic_multipliers``, ``mimic_offsets`` | static | ``q_full[j] = multiplier * q[source] + offset`` |
    | ``meta`` | static, until 0016 | ``dict`` (the IR and parser extras) |
    | ``joint_placements`` | tensor | ``(I..., njoints, 7)`` pose of each joint in its parent joint frame |
    | ``body_inertias`` | tensor | ``(I..., nbodies, 10)`` mass, centre of mass, symmetric inertia |
    | ``frame_placements`` | tensor | ``(I..., nframes, 7)`` pose of each frame in its parent joint frame |
    | ``lower_position_limit``, ``upper_position_limit`` | tensor | ``(nq,)`` |
    | ``velocity_limit``, ``effort_limit``, ``armature``, ``damping`` | tensor | ``(nv,)`` |
    | ``gravity`` | tensor | ``(6,)`` spatial gravity ``[linear, angular]`` |
    | ``q_neutral`` | tensor | ``(nq,)`` |
    """

    name: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    frame_names: tuple[str, ...]
    joints: tuple[JointModel, ...]
    parents: tuple[int, ...]
    frame_parent_joint_ids: tuple[int, ...]
    nqs: tuple[int, ...]
    nvs: tuple[int, ...]
    idx_qs: tuple[int, ...]
    idx_vs: tuple[int, ...]
    nqs_full: tuple[int, ...]
    nvs_full: tuple[int, ...]
    idx_qs_full: tuple[int, ...]
    idx_vs_full: tuple[int, ...]
    mimic_source_ids: tuple[int, ...]
    mimic_multipliers: tuple[float, ...]
    mimic_offsets: tuple[float, ...]
    joint_placements: torch.Tensor
    body_inertias: torch.Tensor
    frame_placements: torch.Tensor
    lower_position_limit: torch.Tensor
    upper_position_limit: torch.Tensor
    velocity_limit: torch.Tensor
    effort_limit: torch.Tensor
    armature: torch.Tensor
    damping: torch.Tensor
    gravity: torch.Tensor
    q_neutral: torch.Tensor
    meta: dict = field(default_factory=dict)
    # Plans (constant tensors derived from the static fields) keyed by (kind, dtype, device); never copied.
    _plans: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def nq(self) -> int:
        """Width of a public configuration ``q``."""
        return sum(self.nqs)

    @property
    def nv(self) -> int:
        """Width of a public tangent ``v``."""
        return sum(self.nvs)

    @property
    def nq_full(self) -> int:
        """Width of the configuration over every joint, mimic targets included."""
        return sum(self.nqs_full)

    @property
    def nv_full(self) -> int:
        """Width of the tangent over every joint, mimic targets included."""
        return sum(self.nvs_full)

    @property
    def njoints(self) -> int:
        """Number of joints, the universe included."""
        return len(self.joints)

    @property
    def nbodies(self) -> int:
        """Number of bodies, the universe included (one per joint)."""
        return len(self.body_names)

    @property
    def nframes(self) -> int:
        """Number of frames."""
        return len(self.frame_names)

    @property
    def dtype(self) -> torch.dtype:
        """dtype of every tensor field."""
        return self.joint_placements.dtype

    @property
    def device(self) -> torch.device:
        """Device of every tensor field."""
        return self.joint_placements.device

    @property
    def structure_key(self) -> tuple:
        """Hashable value of the static fields: two models with equal keys share every plan and index table."""
        return (
            self.name,
            self.joint_names,
            self.body_names,
            self.frame_names,
            self.joints,
            self.parents,
            self.frame_parent_joint_ids,
            self.nqs,
            self.nvs,
            self.idx_qs,
            self.idx_vs,
            self.nqs_full,
            self.nvs_full,
            self.idx_qs_full,
            self.idx_vs_full,
            self.mimic_source_ids,
            self.mimic_multipliers,
            self.mimic_offsets,
        )

    def __repr__(self) -> str:
        dtype = str(self.dtype).removeprefix("torch.")
        return f"Model({self.name!r}, nq={self.nq}, nv={self.nv}, njoints={self.njoints}, {dtype}, {self.device})"

    def _id(self, kind: str, names: tuple[str, ...], value: int | str) -> int:
        if isinstance(value, str):
            if value not in names:
                raise UnknownNameError(kind, value, names)
            return names.index(value)
        if not 0 <= value < len(names):
            raise ArgumentError(f"{kind} id {value} out of range [0, {len(names)})")
        return value

    def joint_id(self, joint: int | str) -> int:
        """Id of a joint given by name or by a range-checked id."""
        return self._id("joint", self.joint_names, joint)

    def frame_id(self, frame: int | str) -> int:
        """Id of a frame given by name or by a range-checked id."""
        return self._id("frame", self.frame_names, frame)

    def body_id(self, body: int | str) -> int:
        """Id of a body given by name or by a range-checked id."""
        return self._id("body", self.body_names, body)

    def q_permutation(self, other_joint_order: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather indices ``(perm_q (nq,), perm_v (nv,))`` from vectors laid out in ``other_joint_order``.

        The external vectors concatenate this model's public per-joint slices in the given order; names of
        zero-width joints may be omitted. ``q_model = q_external[..., perm_q]``.
        """
        order = tuple(other_joint_order)
        duplicates = sorted({name for name in order if order.count(name) > 1})
        if duplicates:
            raise ArgumentError(f"other_joint_order contains duplicate joint names: {duplicates}")
        joint_ids = [self.joint_id(name) for name in order]
        missing = [
            name
            for joint_id, name in enumerate(self.joint_names)
            if (self.nqs[joint_id] or self.nvs[joint_id]) and joint_id not in joint_ids
        ]
        if missing:
            raise ArgumentError(f"other_joint_order is missing joints with public q/v slices: {missing}")
        q_start, v_start = {}, {}
        q_offset = v_offset = 0
        for joint_id in joint_ids:
            q_start[joint_id], v_start[joint_id] = q_offset, v_offset
            q_offset += self.nqs[joint_id]
            v_offset += self.nvs[joint_id]
        perm_q = [q_start[j] + k for j in range(self.njoints) if self.nqs[j] for k in range(self.nqs[j])]
        perm_v = [v_start[j] + k for j in range(self.njoints) if self.nvs[j] for k in range(self.nvs[j])]
        return (
            torch.tensor(perm_q, dtype=torch.long, device=self.device),
            torch.tensor(perm_v, dtype=torch.long, device=self.device),
        )

    def random_configuration(self, *, generator: torch.Generator) -> torch.Tensor:
        """A configuration ``(nq,)`` drawn inside the position limits (revolute joints without limits: ``[-pi, pi]``)."""
        parts = []
        for joint_id, joint in enumerate(self.joints):
            width, start = self.nqs[joint_id], self.idx_qs[joint_id]
            if width:
                lower = self.lower_position_limit[start : start + width]
                upper = self.upper_position_limit[start : start + width]
                parts.append(joint.random(generator, lower=lower, upper=upper))
        if not parts:
            return torch.zeros(self.nq, dtype=self.dtype, device=self.device)
        sample = torch.cat(parts, dim=-1).to(dtype=self.dtype, device=self.device)
        if not bool(torch.isfinite(sample).all()):  # one host sync per call, off the hot path
            unlimited = [
                self.joint_names[j]
                for j, start in enumerate(self.idx_qs)
                if self.nqs[j] and not bool(torch.isfinite(sample[start : start + self.nqs[j]]).all())
            ]
            raise ModelError(f"joints without finite position limits: {unlimited}; set limits before sampling")
        return sample

    def with_values(self, **fields: torch.Tensor) -> Model:
        """Copy with the given tensor fields replaced (``joint_placements=``, ``body_inertias=``, ...).

        Placements are normalised. ``joint_placements``, ``body_inertias`` and ``frame_placements`` may carry a
        batch ``I...`` that broadcasts across the three; the other fields have exactly their event shape.
        """
        unknown = sorted(set(fields) - set(_TENSOR_FIELDS))
        if unknown:
            raise ArgumentError(f"with_values: unknown tensor fields {unknown}; valid: {list(_TENSOR_FIELDS)}")
        checked = {}
        for name, value in fields.items():
            current = getattr(self, name)
            event = tuple(current.shape[-2:]) if name in _BATCHED_FIELDS else tuple(current.shape)
            tensor = check_tensor(name, value, shape=event, dtype=self.dtype, device=self.device)
            if name not in _BATCHED_FIELDS and tensor.ndim != len(event):
                raise ShapeError(f"{name}.shape must be {event}, got {tuple(tensor.shape)}")
            checked[name] = se3.normalize(tensor) if name.endswith("placements") else tensor
        batches = {name: tuple(checked.get(name, getattr(self, name)).shape[:-2]) for name in _BATCHED_FIELDS}
        try:
            torch.broadcast_shapes(*batches.values())
        except RuntimeError as error:
            raise ShapeError(f"value batches do not broadcast: {batches}") from error
        return dataclasses.replace(self, **checked)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Model:
        """Copy with every tensor field on ``device`` in ``dtype`` (plans are rebuilt there on demand)."""
        moved = {name: getattr(self, name).to(device=device, dtype=dtype) for name in _TENSOR_FIELDS}
        return dataclasses.replace(self, **moved)

    def _plan(
        self,
        kind: str,
        build: Callable[[Model, torch.dtype, torch.device], Any],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Any:
        """The ``kind`` plan for ``dtype`` and ``device``, built by ``build(model, dtype, device)`` on first use."""
        key = (kind, dtype, device)
        if key not in self._plans:
            with torch.inference_mode(False):  # an inference tensor in a plan would break later autograd calls
                self._plans[key] = build(self, dtype, device)
        return self._plans[key]


def _flatten(model: Model) -> tuple[list[torch.Tensor], tuple]:
    static = tuple(getattr(model, f.name) for f in dataclasses.fields(model) if f.init and f.name not in _TENSOR_FIELDS)
    return [getattr(model, name) for name in _TENSOR_FIELDS], static


def _unflatten(leaves: list, context: tuple) -> Model:
    names = [f.name for f in dataclasses.fields(Model) if f.init and f.name not in _TENSOR_FIELDS]
    return Model(**dict(zip(names, context, strict=True)), **dict(zip(_TENSOR_FIELDS, leaves, strict=True)))


_pytree.register_pytree_node(Model, _flatten, _unflatten)

__all__ = ["Model"]
