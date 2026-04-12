"""``JointModel`` protocol — the interface every concrete joint implements.

A ``JointModel`` is a **stateless dispatch object** with a compile-time
``nq``/``nv``. It does not store per-joint runtime data — that lives on
``Model`` via ``joint_placements``, ``idx_q``, ``idx_v``. Implementations are
pure functions that operate on slices of the full ``(q, v)`` tensors.

See ``docs/02_DATA_MODEL.md §4``.
"""

from __future__ import annotations

from typing import Literal, Protocol

import torch

JointKind = Literal[
    "universe",
    "fixed",
    "revolute_rx",
    "revolute_ry",
    "revolute_rz",
    "revolute_unaligned",
    "revolute_unbounded",
    "prismatic_px",
    "prismatic_py",
    "prismatic_pz",
    "prismatic_unaligned",
    "spherical",
    "free_flyer",
    "planar",
    "translation",
    "helical",
    "composite",
    "mimic",
]


class JointModel(Protocol):
    """Protocol for per-joint dispatch objects.

    Attributes
    ----------
    kind : JointKind
        String discriminator; one of the 18 supported kinds.
    nq : int
        Size of this joint's slice in the full ``q`` vector.
    nv : int
        Size of this joint's slice in the full ``v`` vector.
    axis : Optional[torch.Tensor]
        ``(3,)`` unit axis for axis-based joints (revolute / prismatic /
        helical). ``None`` otherwise.
    """

    kind: JointKind
    nq: int
    nv: int
    axis: torch.Tensor | None

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """``(B..., nq) → (B..., 7)``. SE3 of child frame in parent frame."""
        ...

    def joint_motion_subspace(self, q_slice: torch.Tensor) -> torch.Tensor:
        """``(B..., nq) → (B..., 6, nv)``. Motion subspace ``S(q)``."""
        ...

    def joint_velocity(
        self, q_slice: torch.Tensor, v_slice: torch.Tensor
    ) -> torch.Tensor:
        """``(B..., 6)``. Spatial velocity in the joint frame."""
        ...

    def integrate(
        self, q_slice: torch.Tensor, v_slice: torch.Tensor
    ) -> torch.Tensor:
        """``(B..., nq)``. Manifold retraction ``q ⊕ v``."""
        ...

    def difference(
        self, q0_slice: torch.Tensor, q1_slice: torch.Tensor
    ) -> torch.Tensor:
        """``(B..., nv)``. Tangent ``q1 ⊖ q0``."""
        ...

    def random_configuration(
        self,
        generator: torch.Generator | None,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor:
        """``(nq,)``. Random valid configuration for this joint."""
        ...

    def neutral(self) -> torch.Tensor:
        """``(nq,)``. Neutral configuration (identity pose / midpoint)."""
        ...
