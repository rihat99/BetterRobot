"""``JointModel`` protocol — the interface every concrete joint implements.

A ``JointModel`` is a **stateless dispatch object** with a compile-time
``nq``/``nv``. It does not store per-joint runtime data — that lives on
``Model`` via ``joint_placements``, ``idx_q``, ``idx_v``. Implementations are
pure functions that operate on slices of the full ``(q, v)`` tensors.

See ``docs/design/02_DATA_MODEL.md §4``.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

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


@runtime_checkable
class JointModel(Protocol):
    """Protocol for per-joint dispatch objects.

    Marked ``@runtime_checkable`` so callers can assert ``isinstance(jm,
    JointModel)`` when wiring custom joint types (docs/conventions/15_EXTENSION.md §2).

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


# ── Dynamics hooks (P11-pre) ─────────────────────────────────────
# Joints with a ``q``-dependent motion subspace can opt in by defining
# ``joint_bias_acceleration`` / ``joint_motion_subspace_derivative``.
# The dispatch helpers below fall back to zero — correct for every joint
# kind currently shipped — and are what dynamics code calls. We keep
# these out of the ``JointModel`` Protocol so runtime_checkable
# ``isinstance`` does not pick up new mandatory methods.


def zero_joint_bias_acceleration(
    q_slice: torch.Tensor, v_slice: torch.Tensor
) -> torch.Tensor:
    """Default ``c_J`` for joints with a ``q``-linear motion subspace."""
    batch_shape = torch.broadcast_shapes(q_slice.shape[:-1], v_slice.shape[:-1])
    return torch.zeros(*batch_shape, 6, dtype=q_slice.dtype, device=q_slice.device)


def zero_joint_motion_subspace_derivative(
    q_slice: torch.Tensor, v_slice: torch.Tensor, nv: int
) -> torch.Tensor:
    """Default ``Ṡ(q, v)`` for joints with constant motion subspace."""
    batch_shape = torch.broadcast_shapes(q_slice.shape[:-1], v_slice.shape[:-1])
    return torch.zeros(*batch_shape, 6, nv, dtype=q_slice.dtype, device=q_slice.device)


def joint_bias_acceleration(
    jm: "JointModel", q_slice: torch.Tensor, v_slice: torch.Tensor
) -> torch.Tensor:
    """Dispatch ``c_J`` through ``jm`` if available, else zero."""
    impl = getattr(jm, "joint_bias_acceleration", None)
    if impl is None:
        return zero_joint_bias_acceleration(q_slice, v_slice)
    return impl(q_slice, v_slice)


def joint_motion_subspace_derivative(
    jm: "JointModel", q_slice: torch.Tensor, v_slice: torch.Tensor
) -> torch.Tensor:
    """Dispatch ``Ṡ`` through ``jm`` if available, else zero."""
    impl = getattr(jm, "joint_motion_subspace_derivative", None)
    if impl is None:
        return zero_joint_motion_subspace_derivative(q_slice, v_slice, jm.nv)
    return impl(q_slice, v_slice)
