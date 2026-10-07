"""Tests for :mod:`better_robot.exceptions` — docs/conventions/contracts.md §2."""

from __future__ import annotations


import pytest

from better_robot import exceptions as exc


def test_all_errors_inherit_from_root() -> None:
    """Every exception class derives from :class:`BetterRobotError`."""
    for name in exc.__all__:
        obj = getattr(exc, name)
        assert issubclass(obj, exc.BetterRobotError), name
        assert issubclass(obj, Exception), name


@pytest.mark.parametrize(
    ("cls", "stdlib_parent"),
    [
        (exc.ModelInconsistencyError, ValueError),
        (exc.DeviceMismatchError, ValueError),
        (exc.DtypeMismatchError, ValueError),
        (exc.QuaternionNormError, ValueError),
        (exc.ShapeError, ValueError),
        (exc.BackendNotAvailableError, ImportError),
    ],
)
def test_stdlib_parent(cls: type, stdlib_parent: type) -> None:
    """Existing ``except ValueError`` etc. blocks keep catching us."""
    assert issubclass(cls, stdlib_parent)
