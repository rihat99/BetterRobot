"""``Model``: one flat frozen record of a robot (decision records 0009, 0010, 0011)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, final

import torch
from torch.utils import _pytree

from ..exceptions import ArgumentError, ModelError, ShapeError, UnknownNameError
from ..lie import se3
from ..lie._checks import check_tensor
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


@final
@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class Model:
    """A robot: static topology plus the tensors that parametrize it. Build one with ``better_robot.load``.

    Static fields (plain Python, hashed into ``structure_key``) and tensor fields (differentiable, moved by ``to``,
    replaced by ``with_values``); ``I...`` is an optional batch of robots sharing one topology:

    | field | kind | shape / type |
    |---|---|---|
    | ``name`` | static | ``str`` |
    | ``joint_names``, ``body_names``, ``frame_names`` | static | ``tuple[str, ...]`` |
    | ``joints`` | static | ``tuple[JointModel, ...]``, joint 0 is the universe |
    | ``parent_joint_ids`` | static | ``tuple[int, ...]``, ``parent_joint_ids[0] == -1``, ``parent_joint_ids[i] < i`` |
    | ``frame_parent_joint_ids`` | static | ``tuple[int, ...]`` |
    | ``nqs``, ``nvs``, ``idx_qs``, ``idx_vs`` | static | public width and start of each joint's ``q``/``v`` slice |
    | ``nqs_full``, ``nvs_full``, ``idx_qs_full``, ``idx_vs_full`` | static | the same over every joint's coordinates |
    | ``mimic_source_ids`` | static | public joint driving each joint (itself when not a mimic target) |
    | ``mimic_multipliers``, ``mimic_offsets`` | static | ``q_full[j] = multiplier * q[source] + offset`` |
    | ``joint_placements`` | tensor | ``(I..., njoints, 7)`` pose of each joint in its parent joint frame |
    | ``body_inertias`` | tensor | ``(I..., nbodies, 10)`` mass, centre of mass, symmetric inertia |
    | ``frame_placements`` | tensor | ``(I..., nframes, 7)`` pose of each frame in its parent joint frame |
    | ``lower_position_limit``, ``upper_position_limit`` | tensor | ``(nq,)`` |
    | ``velocity_limit``, ``effort_limit``, ``armature``, ``damping`` | tensor | ``(nv,)`` |
    | ``gravity`` | tensor | ``(6,)`` spatial gravity ``[linear, angular]`` |
    | ``q_neutral`` | tensor | ``(nq,)`` |
    | ``_plan_owner`` | static, not hashed | ``Model`` whose plans a ``with_values``/``to`` copy shares, else ``None`` |
    """

    name: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    frame_names: tuple[str, ...]
    joints: tuple[JointModel, ...]
    parent_joint_ids: tuple[int, ...]
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
    # Plans (constant tensors derived from the static fields) keyed by (kind, dtype, device); never copied.
    _plans: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    # Values derived from the tensor fields keyed by (kind, grad mode), written only without a graph; never copied.
    _memo: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    # A copy keeps every static field and plans derive from those alone, so `with_values` and `to` point their copy
    # at the first model of the line, and `_plan` takes that model's plans instead of building them again. Not in
    # `structure_key`, the pytree context or the value memo. The link is strong: a copy keeps that model, with its
    # tensors and value memo, alive and pickles it along. A weak link would make every copy unpicklable and rebuild
    # the plans once the first model is dropped; `_plans` itself cannot be handed over, as it is `init=False`.
    _plan_owner: Model | None = field(default=None, repr=False, compare=False, kw_only=True)

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
            self.parent_joint_ids,
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
                raise UnknownNameError(kind, value, valid=names)
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
        """A configuration ``(nq,)`` drawn inside the position limits (revolute joints without limits: ``[-pi, pi]``).

        The generator is required (principle 05: a draw is reproducible by construction), never the global RNG.
        """
        if not isinstance(generator, torch.Generator):
            raise ArgumentError(
                f"random_configuration needs generator=torch.Generator().manual_seed(seed), got {generator!r}"
            )
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
        batch ``I...`` that broadcasts across the three; the other fields have exactly their event shape. The copy
        shares the plans of the first model of its line and keeps that model alive while it lives.
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
                raise ShapeError(f"{name} must have shape {event}, got {tuple(tensor.shape)}")
            checked[name] = se3.normalize_raw(tensor) if name.endswith("placements") else tensor
        batches = {name: tuple(checked.get(name, getattr(self, name)).shape[:-2]) for name in _BATCHED_FIELDS}
        try:
            torch.broadcast_shapes(*batches.values())
        except RuntimeError as error:
            raise ShapeError(f"value batches do not broadcast: {batches}") from error
        return dataclasses.replace(self, **checked, _plan_owner=self if self._plan_owner is None else self._plan_owner)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Model:
        """Copy with every tensor field on ``device`` in ``dtype``; it shares the plans of the first model of its line
        (a plan for a new dtype or device is built there on demand) and keeps that model, tensors included, alive
        while it lives: ``load(...).to("cuda")`` holds the CPU model too."""
        moved = {name: getattr(self, name).to(device=device, dtype=dtype) for name in _TENSOR_FIELDS}
        return dataclasses.replace(self, **moved, _plan_owner=self if self._plan_owner is None else self._plan_owner)

    def _plan(
        self,
        kind: str,
        build: Callable[[Model, torch.dtype, torch.device], Any],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Any:
        """The ``kind`` plan for ``dtype`` and ``device``, built by ``build(model, dtype, device)`` on first use.

        A ``with_values`` or ``to`` copy takes the plan of its ``_plan_owner``, building it there when missing, so
        every copy of one model shares one plan per key; a ``dataclasses.replace`` that changed a static field has
        another ``structure_key`` and builds its own.
        """
        key = (kind, dtype, device)
        if key not in self._plans:
            owner = self._plan_owner
            if owner is None or owner.structure_key != self.structure_key:
                owner = self
            if key not in owner._plans:
                with torch.inference_mode(False):  # an inference tensor in a plan would break later autograd calls
                    owner._plans[key] = build(self, dtype, device)
            self._plans[key] = owner._plans[key]
        return self._plans[key]

    def _value(self, kind: str, derive: Callable[[Model], torch.Tensor]) -> torch.Tensor:
        """``derive(model)``, memoised under ``(kind, grad mode, requires_grad flags)`` once it carries no graph.

        The flags of the tensor fields are in the key so that ``requires_grad_()`` on a model tensor after a query,
        an in-place edit C-STATE-1 forbids, still gets a fresh value. Under ``torch.inference_mode`` nothing is
        stored: an inference tensor served to a later autograd call would break it. Under ``torch.compile`` the
        derivation is traced into the graph instead (Dynamo cannot trace the inference-mode check, and the fused
        ops cost nothing there) (record 0040).
        """
        if torch.compiler.is_compiling() or torch.is_inference_mode_enabled():
            return derive(self)
        flags = tuple(getattr(self, name).requires_grad for name in _TENSOR_FIELDS)
        key = (kind, torch.is_grad_enabled(), flags)
        if key not in self._memo:
            value = derive(self)
            if value.requires_grad:  # a differentiable call owns its graph: converted per call
                return value
            self._memo[key] = value
        return self._memo[key]


#: The pytree context: the static fields (the plan link is not data of the robot).
_STATIC_FIELDS = tuple(
    f.name for f in dataclasses.fields(Model) if f.init and f.name not in (*_TENSOR_FIELDS, "_plan_owner")
)


def _flatten(model: Model) -> tuple[list[torch.Tensor], tuple]:
    return [getattr(model, name) for name in _TENSOR_FIELDS], tuple(getattr(model, name) for name in _STATIC_FIELDS)


def _unflatten(leaves: list, context: tuple) -> Model:
    return Model(**dict(zip(_STATIC_FIELDS, context, strict=True)), **dict(zip(_TENSOR_FIELDS, leaves, strict=True)))


_pytree.register_pytree_node(Model, _flatten, _unflatten)

__all__ = ["Model"]
