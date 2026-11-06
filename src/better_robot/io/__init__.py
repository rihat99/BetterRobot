"""``better_robot.io`` — loaders, parsers, IR, and ``build_model``.

The public entry point is ``load`` — a thin suffix/type dispatcher.

See ``docs/concepts/parsers_and_ir.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import torch

from ..exceptions import ModelError
from ..model.geometry import Geometry
from ..model.joints import JointModel
from ..model.model import Model
from .build_geometry import build_geometry
from .build_model import build_model
from .ir import (
    IRBody,
    IRFrame,
    IRJoint,
    IRModel,
    IRShape,
)
from .parsers import ModelBuilder, parse_mjcf, parse_urdf


def _parser(path: str | Path, format: str) -> Callable[[str | Path], IRModel]:
    """The parser of ``path``: by the explicit ``format``, else by its suffix (``.urdf``; ``.xml`` or ``.mjcf``)."""
    suffix = format if format != "auto" else Path(path).suffix.lstrip(".").lower()
    if suffix == "urdf":
        return parse_urdf
    if suffix in ("mjcf", "xml"):
        return parse_mjcf
    raise ModelError(f"{path}: no parser for the format {suffix!r}; pass format='urdf' or format='mjcf'")


def parse(
    source: str | Path | Callable[[], IRModel],
    *,
    format: Literal["auto", "urdf", "mjcf"] = "auto",
) -> IRModel:
    """Parse a robot description (a file path, or a callable returning the IR) into an editable ``IRModel``.

    Composition lives on the IR: parse several sources, merge or edit their
    ``bodies``/``joints``/``frames`` records, then call ``build_model(ir)``.
    ``load(source)`` is ``build_model(parse(source))`` plus root-joint options.
    """
    if callable(source) and not isinstance(source, (str, Path)):
        # Programmatic builder: callable that returns an IRModel
        ir = source()
        if not isinstance(ir, IRModel):
            raise ModelError(f"a callable source must return an IRModel, got {type(ir).__name__}")
        return ir
    return _parser(source, format)(source)


def load(
    source: str | Path | Callable[[], IRModel],
    *,
    format: Literal["auto", "urdf", "mjcf"] = "auto",
    root_joint: JointModel | None = None,
    free_flyer: bool = False,
    preserve_joint_order: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Load a robot description into a frozen ``Model``.

    Dispatches by file suffix or explicit ``format`` hint.
    Passing ``free_flyer=True`` (or ``root_joint=JointFreeFlyer()``) turns
    any fixed-base URDF into a floating-base robot.
    ``preserve_joint_order=True`` retains already-topological source joint
    order; the default keeps the historical DFS layout.
    ``dtype`` and ``device`` set where the model's tensors are built
    (default ``torch.float32`` on the CPU), like ``model.to(...)`` after loading.

    See docs/concepts/parsers_and_ir.md §1.
    """
    from ..model.joints import JointFreeFlyer  # noqa: PLC0415 - only the free_flyer=True path needs it

    # Resolve root_joint
    if free_flyer and root_joint is None:
        root_joint = JointFreeFlyer()

    ir = parse(source, format=format)

    return build_model(
        ir,
        root_joint=root_joint,
        preserve_joint_order=preserve_joint_order,
        device=device,
        dtype=dtype,
    )


def load_geometry(path: str | Path, model: Model) -> Geometry:
    """The ``Geometry`` of the description at ``path`` for ``model``, which ``load(path)`` built (record 0016)."""
    return build_geometry(parse(path), model)


__all__ = [
    "parse",
    "load",
    "load_geometry",
    "build_model",
    "build_geometry",
    "IRModel",
    "IRBody",
    "IRJoint",
    "IRFrame",
    "IRShape",
    "ModelBuilder",
]
