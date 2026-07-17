"""Reduced/full coordinate transforms for mimic-constrained models.

The public configuration and tangent spaces omit mimic targets.  Whole-body
recursions still run over every concrete joint, so this module owns the small
set of trailing-dimension transforms shared by kinematics and dynamics.
"""

from __future__ import annotations

import torch

from .model_structure import ModelStructure


def expand_configuration(
    structure: ModelStructure,
    q: torch.Tensor,
) -> torch.Tensor:
    """Map public ``(..., nq)`` configurations to ``(..., nq_full)``."""

    if not structure.has_mimic:
        return q
    return q @ structure.q_expansion.mT + structure.q_offset


def expand_tangent(
    structure: ModelStructure,
    value: torch.Tensor,
) -> torch.Tensor:
    """Map a public velocity/acceleration to the full joint tangent."""

    if not structure.has_mimic:
        return value
    return value @ structure.v_expansion.mT


def reduce_generalized_force(
    structure: ModelStructure,
    value_full: torch.Tensor,
) -> torch.Tensor:
    """Apply the mimic chain rule to full-space force or Jacobian rows."""

    if not structure.has_mimic:
        return value_full
    return value_full @ structure.v_expansion


# Both names remain readable at their call sites while sharing one implementation.
reduce_jacobian = reduce_generalized_force


def reduce_mass_matrix(
    structure: ModelStructure,
    mass_full: torch.Tensor,
) -> torch.Tensor:
    """Project a full joint-space matrix as ``G_v.T @ M @ G_v``."""

    if not structure.has_mimic:
        return mass_full
    return structure.v_expansion.mT @ mass_full @ structure.v_expansion


__all__ = [
    "expand_configuration",
    "expand_tangent",
    "reduce_generalized_force",
    "reduce_jacobian",
    "reduce_mass_matrix",
]
