"""Contract test: every public symbol has a non-empty docstring.

The public API from ``better_robot.__all__`` is a contract surface with
users; every listed name must carry a one-line
summary. See ``docs/conventions/extension.md`` and ``docs/conventions/contracts.md``.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import better_robot
import pytest
from better_robot import exceptions, optim, terms


def _undocumented(module: ModuleType) -> list[str]:
    return [name for name in module.__all__ if not (inspect.getdoc(getattr(module, name)) or "").strip()]


def test_every_public_symbol_has_a_docstring() -> None:
    offenders = _undocumented(better_robot)
    assert not offenders, (
        f"Public symbols missing a docstring: {offenders}. "
        f"Every entry in better_robot.__all__ must carry a one-line summary."
    )


@pytest.mark.parametrize("module", [exceptions, optim, terms], ids=lambda module: module.__name__)
def test_public_submodule_symbols_have_docstrings(module: ModuleType) -> None:
    """Every entry in a supported public submodule's ``__all__`` is documented."""
    assert not (offenders := _undocumented(module)), offenders
