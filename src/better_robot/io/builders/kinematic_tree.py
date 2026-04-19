"""Generic kinematic-tree builder — array-driven construction.

Takes a tree defined by ``(joint_names, parents, translations)`` and produces
an ``IRModel`` / ``Model`` via the low-level ``ModelBuilder``. The root
attaches to the universe via ``root_kind`` (default ``"free_flyer"``); every
non-root joint uses ``child_kind`` (default ``"spherical"``). Rotations at
each joint origin are identity — supply custom orientations by composing
origins manually through ``ModelBuilder`` if needed.

This is the common primitive behind ``make_smpl_like_body`` and the
recommended entry point for programmatic construction of any tree of bodies
connected by uniform free-flyer / spherical / fixed joints.

See ``docs/04_PARSERS.md §6``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...data_model.model import Model
from ..build_model import build_model
from ..ir import IRModel
from ..parsers.programmatic import ModelBuilder


def _origin_from_translation(xyz: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return a ``(7,)`` SE3 origin from a 3-vec translation (identity rotation)."""
    o = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=dtype)
    o[:3] = xyz.to(dtype=dtype)
    return o


def _normalize_mass(mass_per_body: float | Sequence[float], n: int) -> list[float]:
    if isinstance(mass_per_body, (int, float)):
        return [float(mass_per_body)] * n
    masses = [float(m) for m in mass_per_body]
    if len(masses) != n:
        raise ValueError(
            f"mass_per_body length {len(masses)} does not match number of bodies {n}"
        )
    return masses


def build_kinematic_tree_body(
    *,
    name: str,
    joint_names: Sequence[str],
    parents: Sequence[int],
    translations: torch.Tensor,
    root_kind: str = "free_flyer",
    child_kind: str = "spherical",
    mass_per_body: float | Sequence[float] = 0.0,
) -> IRModel:
    """Build an ``IRModel`` for a kinematic tree described by arrays.

    Parameters
    ----------
    name
        Name of the model.
    joint_names
        ``(N,)`` body / joint names. ``joint_names[0]`` is the root body.
    parents
        ``(N,)`` parent indices. ``parents[0]`` must be ``-1``; every
        non-root entry must reference an earlier index.
    translations
        ``(N, 3)`` per-joint parent→child translations. ``translations[0]``
        is the origin of the world → root joint (typically zeros for
        ``free_flyer``, since a free-flyer carries root placement in ``q``).
    root_kind
        Joint kind for the world → root edge. Default ``"free_flyer"``.
    child_kind
        Joint kind for every non-root joint. Default ``"spherical"``.
    mass_per_body
        Uniform mass in kg, or a length-``N`` sequence for per-body mass.
    """
    n = len(joint_names)
    if len(parents) != n:
        raise ValueError(
            f"parents length {len(parents)} does not match joint_names length {n}"
        )
    if parents[0] != -1:
        raise ValueError(f"parents[0] must be -1 (root), got {parents[0]}")
    if tuple(translations.shape) != (n, 3):
        raise ValueError(
            f"translations shape {tuple(translations.shape)} must be ({n}, 3)"
        )
    masses = _normalize_mass(mass_per_body, n)
    dtype = translations.dtype

    b = ModelBuilder(name)
    for jname, m in zip(joint_names, masses):
        b.add_body(jname, mass=m)

    b.add_joint(
        "root",
        kind=root_kind,
        parent="world",
        child=joint_names[0],
        origin=_origin_from_translation(translations[0], dtype),
    )

    for idx in range(1, n):
        pidx = parents[idx]
        if pidx < 0 or pidx >= idx:
            raise ValueError(
                f"parents[{idx}] = {pidx} must reference an earlier index in [0, {idx})"
            )
        b.add_joint(
            joint_names[idx],
            kind=child_kind,
            parent=joint_names[pidx],
            child=joint_names[idx],
            origin=_origin_from_translation(translations[idx], dtype),
        )

    return b.finalize()


def build_kinematic_tree_model(
    *,
    name: str,
    joint_names: Sequence[str],
    parents: Sequence[int],
    translations: torch.Tensor,
    root_kind: str = "free_flyer",
    child_kind: str = "spherical",
    mass_per_body: float | Sequence[float] = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Build a frozen ``Model`` for a kinematic tree. See ``build_kinematic_tree_body``."""
    ir = build_kinematic_tree_body(
        name=name,
        joint_names=joint_names,
        parents=parents,
        translations=translations,
        root_kind=root_kind,
        child_kind=child_kind,
        mass_per_body=mass_per_body,
    )
    return build_model(ir, device=device, dtype=dtype)
