"""Contract test: every public symbol has a non-empty docstring.

The 25-symbol public API from ``better_robot.__all__`` is the contract
surface with users; every one of those names must carry a one-line
summary. See ``docs/conventions/15_EXTENSION.md`` and ``docs/conventions/17_CONTRACTS.md``.
"""

from __future__ import annotations

import inspect

import better_robot


def test_every_public_symbol_has_a_docstring() -> None:
    offenders: list[str] = []
    for name in better_robot.__all__:
        obj = getattr(better_robot, name)
        doc = inspect.getdoc(obj)
        if not doc or not doc.strip():
            offenders.append(name)
    assert not offenders, (
        f"Public symbols missing a docstring: {offenders}. "
        f"Every entry in better_robot.__all__ must carry a one-line summary."
    )


def test_public_all_has_26_symbols() -> None:
    """``better_robot.__all__`` is pinned at 26; see docs/design/01_ARCHITECTURE.md."""
    assert len(better_robot.__all__) == 26, (
        f"__all__ has {len(better_robot.__all__)}; expected 26 "
        f"(add or remove a symbol and update 01_ARCHITECTURE.md if intentional)."
    )


def test_exceptions_module_symbols_have_docstrings() -> None:
    """Every entry in ``better_robot.exceptions.__all__`` is documented."""
    from better_robot import exceptions

    offenders: list[str] = []
    for name in exceptions.__all__:
        obj = getattr(exceptions, name)
        if not (inspect.getdoc(obj) or "").strip():
            offenders.append(name)
    assert not offenders, offenders


def test_solver_state_is_documented() -> None:
    """``SolverState`` is the shared optimiser record — must be documented."""
    from better_robot.optim.state import SolverState

    doc = inspect.getdoc(SolverState)
    assert doc and "SolverState" not in doc.split("\n")[0] or "terminal" in (doc or ""), (
        "SolverState needs a purpose-statement docstring."
    )
