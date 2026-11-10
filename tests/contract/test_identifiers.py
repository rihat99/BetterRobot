"""A public function parameter that names a frame, joint or body takes the id or the name, ``int | str`` (C-API-4).

The signatures come from the tracked snapshot ``tests/contract/public_api.json``. ``ModelBuilder`` is the permanent
exception: it names bodies before any id exists, so its parameters take names only (record 0029).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SNAPSHOT: dict[str, str] = json.loads((Path(__file__).parent / "public_api.json").read_text())
IDENTIFIER = re.compile(r"[(,]\s*(frame|joint|body|frames|joints|bodies): '([^']*)'")
NAMES_ONLY = ("better_robot.ModelBuilder.", "better_robot.io.ModelBuilder.")


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
