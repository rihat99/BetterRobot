"""``build_geometry(ir, model)``: the ``Geometry`` of a parsed description for the model built from it (record 0016)."""

from __future__ import annotations

from ..model.geometry import AttachedShape, Geometry
from ..model.model import Model
from .ir import IRModel


def build_geometry(ir: IRModel, model: Model) -> Geometry:
    """The shapes of ``ir`` attached by body id to ``model``; a body that ``model`` lacks raises ``UnknownNameError``."""
    return Geometry(
        shapes=tuple(
            AttachedShape(body=model.body_id(shape.body), shape=shape.shape, collision=shape.collision)
            for shape in ir.shapes
        )
    )
