"""C-API-2 (principle 2.3): a public callable takes at most two positional arguments.

Every function, class constructor and public method that the ``__all__`` tree lists (``better_robot.testing.
public_objects``, each object once) is inspected; parameters after the second must be keyword-only, and an
identifier argument after ``(model, state)`` of a kinematics or dynamics function is never positional.
``PERMANENT`` holds the reasoned exceptions. ``tests/contract/ratchets/keyword_only.json`` holds the violations whose
fix belongs to another work package (``path`` to reason); it only shrinks.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from better_robot.testing import public_objects

MAX_POSITIONAL = 2
PERMANENT: dict[str, str] = {
    "better_robot.kinematics.integrate": "record 0013 decides the call integrate(model, q, v): the step q (+) v",
    "better_robot.kinematics.difference": "record 0013 decides the call difference(model, q0, q1): q1 (-) q0",
}
RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "keyword_only.json").read_text())
ID_PARAMETERS = {"frame", "joint", "body", "frame_id", "joint_id", "body_id", "frames", "joints"}


def _callables() -> dict[str, Callable[..., object]]:
    """Each public function, class (its constructor) and method once, under the first path that lists it."""
    found: dict[int, tuple[str, Callable[..., object]]] = {}
    for path, listed in public_objects("better_robot").items():
        value = listed.__func__ if isinstance(listed, staticmethod | classmethod) else listed
        if inspect.isfunction(value) or (inspect.isclass(value) and inspect.isfunction(value.__init__)):
            found.setdefault(id(value), (path, value))
    return dict(found.values())


def _positional_count(function: Callable[..., object]) -> int:
    parameters = list(inspect.signature(function).parameters.values())
    if parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]
    return sum(
        parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD) for parameter in parameters
    )


CALLABLES = _callables()
OVER = sorted(path for path, function in CALLABLES.items() if _positional_count(function) > MAX_POSITIONAL)


@pytest.mark.parametrize("path", sorted(CALLABLES))
def test_public_callable_is_keyword_only_after_its_second_argument(path: str) -> None:
    count = _positional_count(CALLABLES[path])
    assert count <= MAX_POSITIONAL or path in PERMANENT or path in RATCHET, (
        f"{path} takes {count} positional parameters; make the ones after the second keyword-only (`*`, or "
        "`kw_only=True` on a dataclass)"
    )


def test_exceptions_and_ratchet_name_only_violations() -> None:
    """A fixed entry leaves the ratchet with its fix; a permanent exception exists while its callable does."""
    assert sorted(PERMANENT.keys() - set(OVER)) == []
    assert sorted(RATCHET.keys() - set(OVER)) == []


def test_kinematics_and_dynamics_identifiers_are_keyword_only() -> None:
    """Identifier arguments of a package-level function are never positional, so call sites name them."""
    offenders = [
        f"{path}({parameter.name})"
        for path, function in CALLABLES.items()
        if path.startswith(("better_robot.kinematics.", "better_robot.dynamics.")) and path.count(".") == 2
        for parameter in inspect.signature(function).parameters.values()
        if parameter.name in ID_PARAMETERS and parameter.kind != parameter.KEYWORD_ONLY
    ]
    assert offenders == []
