"""``JointComposite`` — stack of sub-joints acting on the same parent/child.

See ``docs/concepts/joints_bodies_frames.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ...lie import se3 as _se3
from .base import JointModel


@dataclass(frozen=True)
class JointComposite:
    """Composite of sub-joints applied in sequence (left to right composition).

    ``nq`` and ``nv`` are set at construction time as the sum of sub-joint nq/nv.
    """

    sub_joints: tuple[JointModel, ...] = field(default_factory=tuple)
    kind: str = "composite"
    nq: int = field(init=False)
    nv: int = field(init=False)
    axis: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Derive dimensions from the immutable sub-joint sequence."""
        for joint in self.sub_joints:
            if not isinstance(joint, JointModel):
                raise TypeError(f"Composite child {joint!r} does not implement JointModel")
            if joint.kind == "mimic":
                raise ValueError(
                    "JointMimic cannot be nested in JointComposite; attach "
                    "mimic metadata to a concrete scalar joint instead"
                )
        object.__setattr__(self, "nq", sum(joint.nq for joint in self.sub_joints))
        object.__setattr__(self, "nv", sum(joint.nv for joint in self.sub_joints))

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """Compose sub-joint transforms left-to-right."""
        iq = 0
        result = _se3.identity(
            batch_shape=q_slice.shape[:-1],
            device=q_slice.device,
            dtype=q_slice.dtype,
        )
        for jm in self.sub_joints:
            qj = q_slice[..., iq : iq + jm.nq]
            Tj = jm.joint_transform(qj)
            result = _se3.compose(result, Tj)
            iq += jm.nq
        return result

    def joint_motion_subspace(self, q_slice: torch.Tensor) -> torch.Tensor:
        """Concatenate sub-joint motion subspaces along the column axis."""
        parts = []
        iq = 0
        for jm in self.sub_joints:
            qj = q_slice[..., iq : iq + jm.nq]
            parts.append(jm.joint_motion_subspace(qj))
            iq += jm.nq
        return torch.cat(parts, dim=-1)

    def joint_velocity(self, q_slice, v_slice) -> torch.Tensor:
        """Sum S_i * v_i over sub-joints."""
        *batch, _ = v_slice.shape
        out = torch.zeros(*batch, 6, dtype=v_slice.dtype, device=v_slice.device)
        iq = iv = 0
        for jm in self.sub_joints:
            qj = q_slice[..., iq : iq + jm.nq]
            vj = v_slice[..., iv : iv + jm.nv]
            out = out + jm.joint_velocity(qj, vj)
            iq += jm.nq
            iv += jm.nv
        return out

    def integrate(self, q_slice, v_slice) -> torch.Tensor:
        """Per-sub-joint retraction."""
        parts = []
        iq = iv = 0
        for jm in self.sub_joints:
            qj = q_slice[..., iq : iq + jm.nq]
            vj = v_slice[..., iv : iv + jm.nv]
            parts.append(jm.integrate(qj, vj))
            iq += jm.nq
            iv += jm.nv
        return torch.cat(parts, dim=-1)

    def difference(self, q0_slice, q1_slice) -> torch.Tensor:
        """Per-sub-joint difference."""
        parts = []
        iq = 0
        for jm in self.sub_joints:
            q0j = q0_slice[..., iq : iq + jm.nq]
            q1j = q1_slice[..., iq : iq + jm.nq]
            parts.append(jm.difference(q0j, q1j))
            iq += jm.nq
        return torch.cat(parts, dim=-1)

    def random_configuration(self, generator, lower, upper) -> torch.Tensor:
        """Per-sub-joint random config."""
        parts = []
        iq = 0
        for jm in self.sub_joints:
            lj = lower[iq : iq + jm.nq]
            uj = upper[iq : iq + jm.nq]
            parts.append(jm.random_configuration(generator, lj, uj))
            iq += jm.nq
        return torch.cat(parts, dim=-1)

    def neutral(self) -> torch.Tensor:
        """Concatenation of sub-joint neutrals."""
        return torch.cat([jm.neutral() for jm in self.sub_joints], dim=-1)
