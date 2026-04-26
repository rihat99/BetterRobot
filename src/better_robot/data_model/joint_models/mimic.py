"""``JointMimic`` — zero-DOF joint whose value is derived from another joint.

The mimic relationship is evaluated via a vectorised gather on the ``Model``
side (``mimic_multiplier``/``mimic_offset``/``mimic_source``), not as a
per-joint branch in the hot path.

See ``docs/design/02_DATA_MODEL.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointMimic:
    """Zero-DOF mimic joint; value = ``mult * q[src] + off``.

    The actual joint expansion is done at the Model level via the vectorised
    gather (``mimic_multiplier``, ``mimic_offset``, ``mimic_source``). The
    JointMimic object itself is a stateless placeholder that the FK loop
    handles specially.
    """

    kind: str = "mimic"
    nq: int = 0
    nv: int = 0
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """Identity — the mimic offset is encoded in model.joint_placements."""
        return torch.tensor([0., 0., 0., 0., 0., 0., 1.], dtype=torch.float32)

    def joint_motion_subspace(self, q_slice: torch.Tensor) -> torch.Tensor:
        return torch.zeros(6, 0)

    def joint_velocity(self, q_slice, v_slice) -> torch.Tensor:
        return torch.zeros(6)

    def integrate(self, q_slice, v_slice) -> torch.Tensor:
        return q_slice.clone()

    def difference(self, q0_slice, q1_slice) -> torch.Tensor:
        return torch.zeros(0, dtype=q0_slice.dtype, device=q0_slice.device)

    def random_configuration(self, generator, lower, upper) -> torch.Tensor:
        return torch.zeros(0)

    def neutral(self) -> torch.Tensor:
        return torch.zeros(0)
