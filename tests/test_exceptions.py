"""Tests for :mod:`better_robot.exceptions` — docs/17_CONTRACTS §2."""

from __future__ import annotations

import warnings

import pytest

import better_robot
from better_robot import exceptions as exc


def test_module_accessible_from_top_level() -> None:
    """``import better_robot`` exposes the exceptions sub-module."""
    assert better_robot.exceptions is exc


def test_all_errors_inherit_from_root() -> None:
    """Every exception class derives from :class:`BetterRobotError`."""
    for name in exc.__all__:
        obj = getattr(exc, name)
        if name == "SingularityWarning":
            assert issubclass(obj, Warning)
            continue
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
        (exc.ConvergenceError, RuntimeError),
        (exc.BackendNotAvailableError, ImportError),
        (exc.UnsupportedJointError, ValueError),
    ],
)
def test_stdlib_parent(cls: type, stdlib_parent: type) -> None:
    """Existing ``except ValueError`` etc. blocks keep catching us."""
    assert issubclass(cls, stdlib_parent)


def test_errors_carry_messages() -> None:
    """Each exception accepts a message and surfaces it on ``str()``."""
    e = exc.ShapeError("q has trailing size 8, expected 7")
    assert "expected 7" in str(e)


def test_singularity_warning_is_filterable() -> None:
    """``SingularityWarning`` behaves like a standard ``UserWarning``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("ill-conditioned", exc.SingularityWarning)
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
