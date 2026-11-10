"""A public function parameter that names a frame, joint or body takes the id or the name, ``int | str`` (C-API-4).

The signatures come from the tracked snapshot ``tests/contract/public_api.json``. ``ModelBuilder`` is the permanent
exception: it names bodies before any id exists, so its parameters take names only (record 0029). An annotation
can say ``int | str`` while the body refuses one, so every ``_public_ops`` case with an identifier also runs with
the id and with the name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from ._public_ops import CASE_NAMES, build_case

SNAPSHOT: dict[str, str] = json.loads((Path(__file__).parent / "public_api.json").read_text())
IDENTIFIER = re.compile(r"[(,]\s*(frame|joint|body|frames|joints|bodies): '([^']*)'")
NAMES_ONLY = ("better_robot.ModelBuilder.", "better_robot.io.ModelBuilder.")
KIND = {"frame": "frame", "frames": "frame", "joint": "joint", "joints": "joint", "body": "body", "bodies": "body"}


def test_identifier_parameters_take_an_id_or_a_name() -> None:
    """Every ``frame``, ``joint`` or ``body`` parameter (and its plural) of a public function accepts ``int | str``."""
    found = [
        f"{name}({parameter}: {annotation})"
        for name, signature in SNAPSHOT.items()
        if signature.startswith("function(") and not name.startswith(NAMES_ONLY)
        for parameter, annotation in IDENTIFIER.findall(signature)
        if "int | str" not in annotation and annotation != "JointModel"  # a joint object, not an identifier
    ]
    assert not found, found


def _spelled(model, kind: str, value, *, as_name: bool):
    """``value`` (an id, a name or a list of them) written as names or as ids of ``kind``."""
    if isinstance(value, list):
        return [_spelled(model, kind, item, as_name=as_name) for item in value]
    index = getattr(model, f"{kind}_id")(value)
    return getattr(model, f"{kind}_names")[index] if as_name else index


def test_identifier_arguments_accept_the_id_and_the_name_alike() -> None:
    """A case whose function refuses the name (or the id) of its frame, joint or body fails, whatever it is annotated."""
    cases = [(case, *build_case(case, torch.float64, torch.device("cpu"))) for case in CASE_NAMES]
    cases = [(case, fn, args, kwargs) for case, fn, args, kwargs in cases if KIND.keys() & kwargs.keys()]
    assert cases
    for case, fn, (model, *rest), kwargs in cases:
        spellings = [
            {
                key: _spelled(model, KIND[key], value, as_name=as_name) if key in KIND else value
                for key, value in kwargs.items()
            }
            for as_name in (False, True)
        ]
        by_id, by_name = (fn(model, *rest, **spelling) for spelling in spellings)
        torch.testing.assert_close(by_name, by_id, msg=case)
