"""``ResidualSpec`` — optional sparsity / structural metadata on a residual.

Solvers can look at ``.spec`` to pre-build sparse Jacobian masks (trajopt
especially). Opt-in: solvers that ignore ``.spec`` still work.

Per ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §7`` the full shape carries:

* ``output_dim`` — residual output dimension (alias kept as ``dim``).
* ``tangent_dim`` — variable space ``nv`` this residual touches; ``None``
  means the full variable.
* ``structure`` — dense block layout. ``"dense"``, ``"diagonal"``,
  ``"block"`` (collision-pair-style), ``"banded"`` (temporal smoothing).
* ``time_coupling`` — for trajectory residuals: ``"single"``,
  ``"5-point"``, ``"custom"``.
* ``affected_knots`` — tuple of knot indices touched (trajopt).
* ``affected_joints`` / ``affected_frames`` — index sets, used to compose
  block-sparse Jacobians.
* ``dynamic_dim`` — ``True`` if the residual length can change between
  evaluations (collision residuals during a solve).

Backwards compatible: ``input_indices`` and ``is_diagonal`` are kept as
deprecated aliases for tests that still set them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ResidualStructure = Literal["dense", "diagonal", "block", "banded"]
TimeCoupling = Literal["single", "5-point", "custom"]


@dataclass
class ResidualSpec:
    """Optional structural metadata attached to a residual."""

    dim: int
    output_dim: int | None = None
    tangent_dim: int | None = None
    structure: ResidualStructure = "dense"
    time_coupling: TimeCoupling = "single"
    affected_knots: tuple[int, ...] = ()
    affected_joints: tuple[int, ...] = ()
    affected_frames: tuple[int, ...] = ()
    dynamic_dim: bool = False
    # Deprecated aliases — older tests set these:
    input_indices: tuple[int, ...] | None = None
    is_diagonal: bool = False

    def __post_init__(self) -> None:
        if self.output_dim is None:
            self.output_dim = self.dim
        if self.is_diagonal and self.structure == "dense":
            self.structure = "diagonal"
