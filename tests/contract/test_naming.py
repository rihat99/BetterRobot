"""The field names of ``Model`` and ``KinematicState`` tell their shape and type (C-NAME-2, record 0023, principle 3.3).

A tensor field is plural exactly when its documented shape has an axis over named things (``njoints``, ``nframes``,
``nbodies``, ``J``, ``V``); a tuple field holds one entry per joint, body or frame, so it is plural. A field of
integer indices ends in ``_ids``, and a ``_ids`` field holds integers. The shapes are read from the class docstring's
field table and from each field's ``#:`` comment.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from better_robot.kinematics import KinematicState
from better_robot.model import Model

ENTITY_AXES = re.compile(r"\b(njoints|nframes|nbodies|J|V)\b")
COORDINATE_FIELDS = {
    name: "the width and start of each joint's slice of q or v, not ids of joints, bodies or frames (record 0009)"
    for name in ("nqs", "nvs", "idx_qs", "idx_vs", "nqs_full", "nvs_full", "idx_qs_full", "idx_vs_full")
}


def _public_fields(cls: type) -> dict[str, str]:
    """Name to annotation of each public dataclass field."""
    return {field.name: str(field.type) for field in dataclasses.fields(cls) if not field.name.startswith("_")}


def _documented_shapes(cls: type) -> dict[str, str]:
    """Field to the text documenting it: its docstring table row (``| ``a``, ``b`` | ... |``) or ``#:`` comment."""
    shapes: dict[str, str] = {}
    for row in (inspect.getdoc(cls) or "").splitlines():
        cells = row.split("|")
        if len(cells) > 3:
            for name in re.findall(r"``(\w+)``", cells[1]):
                shapes[name] = "|".join(cells[2:])
    for line in inspect.getsource(cls).splitlines():
        match = re.match(r"\s+(\w+):[^#]*#:(.*)", line)
        if match:
            shapes[match.group(1)] = match.group(2)
    return shapes


def _is_plural(name: str) -> bool:
    """The head noun ends in ``s``; ``_full`` qualifies a field (``nqs_full``) and is not its noun."""
    return name.removesuffix("_full").rsplit("_", 1)[-1].endswith("s")


@pytest.mark.parametrize("cls", [Model, KinematicState], ids=lambda cls: cls.__name__)
def test_a_field_is_plural_exactly_when_it_has_an_axis_over_named_things(cls: type) -> None:
    shapes = _documented_shapes(cls)
    wrong = []
    for name, annotation in _public_fields(cls).items():
        if annotation.startswith("tuple["):
            if not _is_plural(name):
                wrong.append(f"{name}: one entry per joint, body or frame, but singular")
        elif "Tensor" in annotation:
            assert name in shapes, f"{cls.__name__}.{name}: no documented shape"
            if bool(ENTITY_AXES.search(shapes[name])) != _is_plural(name):
                wrong.append(f"{name}: shape {shapes[name].strip()!r}")
    assert not wrong, f"{cls.__name__} names whose number does not match the shape:\n  " + "\n  ".join(wrong)


@pytest.mark.parametrize("cls", [Model, KinematicState], ids=lambda cls: cls.__name__)
def test_integer_index_fields_end_in_ids_and_ids_fields_hold_integers(cls: type) -> None:
    fields = _public_fields(cls)
    integers = {name for name, annotation in fields.items() if annotation == "tuple[int, ...]"}
    assert sorted(integers - COORDINATE_FIELDS.keys() - {name for name in integers if name.endswith("_ids")}) == []
    assert sorted(name for name in fields if name.endswith("_ids") and name not in integers) == []
