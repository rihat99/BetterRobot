"""``ResidualSpec`` — optional sparsity / structural metadata on a residual.

Solvers can look at ``.spec`` to pre-build sparse Jacobian masks (trajopt
especially). Opt-in: solvers that ignore ``.spec`` still work.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §7``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ResidualSpec:
    """Optional structural metadata attached to a residual."""

    dim: int
    input_indices: tuple[int, ...] | None = None
    is_diagonal: bool = False
    time_coupling: Literal["single", "5-point", "custom"] = "single"
