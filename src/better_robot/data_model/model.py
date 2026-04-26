"""``Model`` — frozen kinematic-tree description (Pinocchio-style).

``Model`` is built once, shared across workers/devices, and never mutated.
Every tensor buffer is device/dtype polymorphic via ``.to()``. The static
vs floating-base distinction **disappears**: a floating base is simply
``joint_models[1] = JointFreeFlyer``.

See ``docs/concepts/model_and_data.md §2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .frame import Frame
from .joint_models.base import JointModel


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

    # ────────────────────────── methods ──────────────────────────

    def to(self, device=None, dtype=None) -> "Model":
        """Return a new ``Model`` with every tensor buffer moved to the given
        device and/or dtype. Topology / names / joint models are shared by
        reference (they are immutable).
        """
        def _t(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device=device, dtype=dtype)

        import dataclasses
        return dataclasses.replace(
            self,
            joint_placements=_t(self.joint_placements),
            body_inertias=_t(self.body_inertias),
            lower_pos_limit=_t(self.lower_pos_limit),
            upper_pos_limit=_t(self.upper_pos_limit),
            velocity_limit=_t(self.velocity_limit),
            effort_limit=_t(self.effort_limit),
            rotor_inertia=_t(self.rotor_inertia),
            armature=_t(self.armature),
            friction=_t(self.friction),
            damping=_t(self.damping),
            gravity=_t(self.gravity),
            mimic_multiplier=_t(self.mimic_multiplier),
            mimic_offset=_t(self.mimic_offset),
            q_neutral=_t(self.q_neutral),
            reference_configurations={k: _t(v)
                                       for k, v in self.reference_configurations.items()},
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
        _dtype  = dtype  or self.joint_placements.dtype
        q = torch.zeros(*batch_shape, self.nq, device=_device, dtype=_dtype)
        return Data(_model_id=id(self), q=q)

    def joint_id(self, name: str) -> int:
        """Return the integer id of the named joint. Raises ``KeyError`` if missing."""
        return self.joint_name_to_id[name]

    def frame_id(self, name: str) -> int:
        """Return the integer id of the named frame."""
        return self.frame_name_to_id[name]

    def body_id(self, name: str) -> int:
        """Return the integer id of the named body."""
        return self.body_name_to_id[name]

    def body_inertia(self, body_id: int):
        """Typed accessor returning a single body's :class:`~better_robot.spatial.Inertia`."""
        from ..spatial.inertia import Inertia
        return Inertia(self.body_inertias[body_id])

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
            return torch.zeros(*q0.shape[:-1], self.nv,
                               device=q0.device, dtype=q0.dtype)
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
