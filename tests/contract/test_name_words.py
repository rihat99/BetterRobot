"""Words in public names: layouts follow the data order (C-NAME-3), units are SI and *frame* is a robot frame (C-NAME-4).

The public names and their signatures come from the tracked snapshot ``tests/contract/public_api.json``, which
``tests/contract/test_api_snapshot.py`` keeps equal to the package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SNAPSHOT: dict[str, str] = json.loads((Path(__file__).parent / "public_api.json").read_text())
PARAMETER = re.compile(r"[(,]\s*\*{0,2}(\w+)\s*[:=,)]")
FOREIGN_LAYOUT = re.compile(r"wxyz|angular_linear|angular_first")
NON_SI_UNIT = re.compile(r"(?:^|_)(?:deg|degrees|ms|millis|milliseconds|mm|cm|minutes|hz)(?:_|$)")


def _words() -> list[tuple[str, str]]:
    """``(public dotted name, word)`` for every name's last segment and every parameter name in its signature."""
    words = []
    for name, signature in SNAPSHOT.items():
        words.append((name, name.rsplit(".", 1)[-1]))
        words += [(name, parameter) for parameter in PARAMETER.findall(signature.split("->", 1)[0])]
    return words


def test_layout_words_name_converters_only() -> None:
    """C-NAME-3: ``wxyz`` or angular-first appears only in a ``to_``/``from_`` converter's name."""
    found = [f"{name}: {word}" for name, word in _words() if FOREIGN_LAYOUT.search(word)]
    found = [entry for entry in found if not entry.split(": ")[1].startswith(("to_", "from_"))]
    assert not found, found


def test_public_quantities_are_in_si_units() -> None:
    """C-NAME-4: no public name or parameter carries a unit other than radians, metres and seconds."""
    found = [f"{name}: {word}" for name, word in _words() if NON_SI_UNIT.search(word.lower())]
    assert not found, found


def test_frame_means_a_robot_frame_in_the_viewer() -> None:
    """C-NAME-4: the viewer's time index is ``step``; ``frame`` appears only in ``FramesLayer``, which draws frames."""
    found = [
        f"{name}: {word}"
        for name, word in _words()
        if name.startswith("better_robot.viewer.")
        and "frame" in word.lower()
        and not name.startswith("better_robot.viewer.FramesLayer")
    ]
    assert not found, found
