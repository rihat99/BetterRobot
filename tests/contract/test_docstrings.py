"""Contract test: every public symbol has a non-empty docstring.

The public API from ``better_robot.__all__`` is a contract surface with
users; every listed name must carry a one-line
summary. See ``docs/conventions/extension.md`` and ``docs/conventions/contracts.md``.
"""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import get_origin

import better_robot
import pytest
from better_robot import exceptions, optim, residuals

PUBLIC_TYPE_ALIASES: dict[ModuleType, frozenset[str]] = {
    optim: frozenset({"LinearSystem", "LinearizationMode", "JacobianStrategy"}),
}


def _undocumented(module: ModuleType) -> list[str]:
    aliases = PUBLIC_TYPE_ALIASES.get(module, frozenset())
    return [
        name
        for name in module.__all__
        if name not in aliases and not (inspect.getdoc(getattr(module, name)) or "").strip()
    ]


def test_every_public_symbol_has_a_docstring() -> None:
    offenders = _undocumented(better_robot)
    assert not offenders, (
        f"Public symbols missing a docstring: {offenders}. "
        f"Every entry in better_robot.__all__ must carry a one-line summary."
    )


@pytest.mark.parametrize("module", [exceptions, optim, residuals], ids=lambda module: module.__name__)
def test_public_submodule_symbols_have_docstrings(module: ModuleType) -> None:
    """Every entry in a supported public submodule's ``__all__`` is documented."""
    assert not (offenders := _undocumented(module)), offenders


@pytest.mark.parametrize(
    ("module", "name"),
    [(module, name) for module, names in PUBLIC_TYPE_ALIASES.items() for name in sorted(names)],
)
def test_public_type_alias_metadata_is_valid(module: ModuleType, name: str) -> None:
    """Docstring exemptions are explicit public typing aliases, never ordinary objects."""
    assert name in module.__all__
    assert get_origin(getattr(module, name)) is not None
