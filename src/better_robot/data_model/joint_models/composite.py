"""``JointComposite`` — stack of sub-joints acting on the same parent/child.

See ``docs/design/02_DATA_MODEL.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class JointComposite:
    """Composite of sub-joints applied in sequence (left to right composition).

    ``nq`` and ``nv`` are set at construction time as the sum of sub-joint nq/nv.
    """

    sub_joints: tuple[Any, ...] = field(default_factory=tuple)
    kind: str = "composite"
    nq: int = 0
    nv: int = 0
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """Compose sub-joint transforms left-to-right."""
        from ...lie import se3 as _se3
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
