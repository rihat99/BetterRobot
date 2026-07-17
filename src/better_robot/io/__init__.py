"""``better_robot.io`` — loaders, parsers, IR, and ``build_model``.

The public entry point is ``load`` — a thin suffix/type dispatcher.

See ``docs/concepts/parsers_and_ir.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import torch

from ..data_model.joint_models.base import JointModel
from ..data_model.model import Model
from .assets import (
    AssetResolver,
    CachedDownloadResolver,
    CompositeResolver,
    FilesystemResolver,
    PackageResolver,
)
from .build_model import build_model
from .ir import (
    IRBody,
    IRError,
    IRFrame,
    IRGeom,
    IRJoint,
    IRModel,
)
from .parsers import ModelBuilder, parse_mjcf, parse_urdf

_PARSERS: dict[str, Callable[..., IRModel]] = {
    "urdf": parse_urdf,
    "mjcf": parse_mjcf,
    "xml": parse_mjcf,
}


def register_parser(suffix: str, fn: Callable[..., IRModel]) -> None:
    """Register a new format parser at runtime. See docs/concepts/parsers_and_ir.md §7."""
    _PARSERS[suffix] = fn


def _detect_format(source: Any, hint: str) -> str:
    """Infer the format string from the source path or explicit hint."""
    if hint != "auto":
        return hint
    if isinstance(source, (str, Path)):
        suffix = Path(source).suffix.lstrip(".").lower()
        if suffix in _PARSERS:
            return suffix
        raise ValueError(
            f"Cannot infer format from suffix {suffix!r}; pass format= explicitly. Known: {list(_PARSERS)}"
        )
    raise ValueError("Cannot infer format for non-path source; pass format= explicitly.")


def load(
    source: str | Path | Any | Callable[[], IRModel],
    *,
    format: Literal["auto", "urdf", "mjcf", "builder"] = "auto",
    root_joint: JointModel | None = None,
    free_flyer: bool = False,
    preserve_joint_order: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Load a robot description into a frozen ``Model``.

    Dispatches by file suffix, argument type, or explicit ``format`` hint.
    Passing ``free_flyer=True`` (or ``root_joint=JointFreeFlyer()``) turns
    any fixed-base URDF into a floating-base robot.
    ``preserve_joint_order=True`` retains already-topological source joint
    order; the default keeps the historical DFS layout.

    See docs/concepts/parsers_and_ir.md §1.
    """
    from ..data_model.joint_models import JointFreeFlyer

    # Resolve root_joint
    if free_flyer and root_joint is None:
        root_joint = JointFreeFlyer()

    # Parse → IRModel
    if callable(source) and not isinstance(source, (str, Path)):
        # Programmatic builder: callable that returns an IRModel
        ir = source()
        if not isinstance(ir, IRModel):
            raise ValueError(f"Programmatic builder must return an IRModel, got {type(ir)}")
    else:
        # Try to detect if it's a yourdfpy.URDF object
        try:
            import yourdfpy

            if isinstance(source, yourdfpy.URDF):
                ir = parse_urdf(source)
            else:
                suffix = _detect_format(source, format)
                ir = _PARSERS[suffix](source)
        except ImportError:
            # yourdfpy not available; just try the suffix route
            suffix = _detect_format(source, format)
            ir = _PARSERS[suffix](source)

    return build_model(
        ir,
        root_joint=root_joint,
        preserve_joint_order=preserve_joint_order,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "load",
    "build_model",
    "register_parser",
    "IRModel",
    "IRBody",
    "IRJoint",
    "IRFrame",
    "IRGeom",
    "IRError",
    "ModelBuilder",
    "AssetResolver",
    "FilesystemResolver",
    "PackageResolver",
    "CompositeResolver",
    "CachedDownloadResolver",
]
