"""``Model`` is one flat frozen record (decision records 0009, 0016).

The class docstring table is the field contract: every dataclass field appears there as ``static`` or ``tensor``,
the tensor fields hold tensors and the static ones do not, no field is geometry, and no property forwards to a
part (there are no parts).
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re

import pytest
import torch

from better_robot import load
from better_robot.model import Model
from tests.support.robots import panda_urdf


def _documented_fields() -> dict[str, str]:
    """``{field: "static" | "tensor"}`` read from the docstring table."""
    table: dict[str, str] = {}
    for line in inspect.getdoc(Model).splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not (cells[1] == "tensor" or cells[1].startswith("static")):
            continue
        for name in re.findall(r"``(\w+)``", cells[0]):
            table[name] = "tensor" if cells[1] == "tensor" else "static"
    return table


@pytest.fixture(scope="module")
def panda() -> Model:
    return load(panda_urdf())


def test_every_field_is_in_the_docstring_table_with_its_kind(panda: Model) -> None:
    documented = _documented_fields()
    assert set(documented) == {f.name for f in dataclasses.fields(Model) if f.init}
    for name, kind in documented.items():
        assert isinstance(getattr(panda, name), torch.Tensor) == (kind == "tensor"), name


def test_no_field_is_geometry(panda: Model) -> None:
    """Meshes and shapes live outside the model (record 0016)."""
    names = {f.name for f in dataclasses.fields(Model)}
    assert not {name for name in names if any(word in name for word in ("geom", "mesh", "visual", "collision"))}
    assert not any(dataclasses.is_dataclass(getattr(panda, name)) for name in names)


def test_properties_derive_and_never_forward() -> None:
    """A property body ``return self.<a>.<b>`` is only allowed when ``<a>`` is a tensor field (``dtype``, ``device``)."""
    tensor_fields = {name for name, kind in _documented_fields().items() if kind == "tensor"}
    for node in ast.walk(ast.parse(inspect.getsource(Model))):
        is_property = isinstance(node, ast.FunctionDef) and any(
            isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in node.decorator_list
        )
        if not is_property:
            continue
        body = [statement for statement in node.body if not isinstance(statement, ast.Expr)]  # drop the docstring
        if len(body) == 1 and isinstance(body[0], ast.Return):
            value = body[0].value
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Attribute):
                assert value.value.attr in tensor_fields, f"forwarding property {node.name}"
