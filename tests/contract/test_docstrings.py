"""Contract test: every public symbol has a non-empty docstring.

The public API from ``better_robot.__all__`` is a contract surface with
users; every listed name must carry a one-line
summary. See ``docs/conventions/extension.md`` and ``docs/conventions/contracts.md``.
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
    assert doc, "SolverState needs a purpose-statement docstring."
    summary = doc.splitlines()[0].lower()
    assert "iteration" in summary and "terminal" in summary, (
        "SolverState's summary must describe its iteration and terminal roles."
    )
