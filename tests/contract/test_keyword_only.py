"""C-API-2 (principle 2.3): a public callable takes at most two positional arguments.

Every function, class constructor and public method that the ``__all__`` tree lists (``better_robot.testing.
public_objects``, each object once) is inspected; parameters after the second must be keyword-only, and an
identifier argument after ``(model, state)`` of a kinematics or dynamics function is never positional.
``PERMANENT`` holds the reasoned exceptions (``path`` to its positional count and the reason).
``tests/contract/ratchets/keyword_only.json`` holds the violations whose fix belongs to another work package (``path``
to its positional count and the reason); it only shrinks. A listed callable may not take one more positional
parameter than its entry says.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

from better_robot.testing import public_objects

from ._ratchet_sites import counts

MAX_POSITIONAL = 2
PERMANENT: dict[str, tuple[int, str]] = {
    "better_robot.kinematics.integrate": (3, "record 0013 decides the call integrate(model, q, v): the step q (+) v"),
    "better_robot.kinematics.difference": (3, "record 0013 decides the call difference(model, q0, q1): q1 (-) q0"),
}
ALLOWED: dict[str, int] = {
    **{path: count for path, (count, _) in PERMANENT.items()},
    **counts(Path(__file__).parent / "ratchets" / "keyword_only.json"),
}
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


@pytest.mark.parametrize("path", sorted(CALLABLES))
def test_public_callable_is_keyword_only_after_its_second_argument(path: str) -> None:
    count = _positional_count(CALLABLES[path])
    assert count <= ALLOWED.get(path, MAX_POSITIONAL), (
        f"{path} takes {count} positional parameters; make the ones after the second keyword-only (`*`, or "
        "`kw_only=True` on a dataclass)"
    )


def test_exceptions_and_ratchet_name_only_violations() -> None:
    """A fixed entry leaves the ratchet with its fix; a permanent exception exists while its callable does, with
    the positional count its entry states."""
    stale = sorted(
        f"{path}: entry {allowed}, found {_positional_count(CALLABLES[path]) if path in CALLABLES else 'no callable'}"
        for path, allowed in ALLOWED.items()
        if path not in CALLABLES or _positional_count(CALLABLES[path]) != allowed
    )
    assert stale == []


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
