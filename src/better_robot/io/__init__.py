"""``better_robot.io``: ``load`` a description into a ``Model``, in two stages anyone can split.

``parse(path)`` reads a URDF or MJCF file into the editable ``IRModel``; ``build_model(ir)`` turns an IR (parsed or
written with ``ModelBuilder``) into the frozen ``Model``; ``load(path)`` is both. ``load_geometry(path, model)``
reads the shapes of the same description. See ``docs/concepts/io.md``.
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
from .builder import ModelBuilder
from .ir import (
    IRBody,
    IRFrame,
    IRJoint,
    IRModel,
    IRShape,
)
from .parser_mjcf import parse_mjcf
from .parser_urdf import parse_urdf


def _parser(path: str | Path, format: str | None) -> Callable[[str | Path], IRModel]:
    """The parser of ``path``: by the explicit ``format``, else by its suffix (``.urdf``; ``.xml`` or ``.mjcf``)."""
    suffix = format if format is not None else Path(path).suffix.lstrip(".").lower()
    if suffix == "urdf":
        return parse_urdf
    if suffix in ("mjcf", "xml"):
        return parse_mjcf
    raise ModelError(f"{path}: no parser for the format {suffix!r}; pass format='urdf' or format='mjcf'")


def parse(path: str | Path, *, format: Literal["urdf", "mjcf"] | None = None) -> IRModel:
    """The editable ``IRModel`` of the description at ``path``, read by the parser of its suffix or of ``format``.

    Composition lives on the IR: parse several files, merge or edit their ``bodies``, ``joints`` and ``frames``,
    then ``build_model(ir)``. An unknown suffix is a ``ModelError``.
    """
    return _parser(path, format)(path)


def load(
    path: str | Path,
    *,
    format: Literal["urdf", "mjcf"] | None = None,
    root_joint: JointModel | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """The frozen ``Model`` of the description at ``path``: ``build_model(parse(path, format=format), ...)``.

    ``root_joint`` replaces the joint that holds the root body under the universe (``JointFreeFlyer()`` makes the
    base float; the description's own root joint, or a ``JointFixed``, otherwise). ``device`` and ``dtype`` set
    where and in what precision the model's tensors are built, like ``model.to(...)`` after loading.
    """
    return build_model(parse(path, format=format), root_joint=root_joint, device=device, dtype=dtype)


def load_geometry(path: str | Path, model: Model) -> Geometry:
    """The ``Geometry`` of the description at ``path`` for ``model``, which ``load(path)`` built (record 0016)."""
    return build_geometry(parse(path), model)


__all__ = [
    "load",
    "parse",
    "parse_urdf",
    "parse_mjcf",
    "build_model",
    "build_geometry",
    "load_geometry",
    "ModelBuilder",
    "IRModel",
    "IRBody",
    "IRJoint",
    "IRFrame",
    "IRShape",
]
