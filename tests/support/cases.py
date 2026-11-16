"""One test over a table of cases: every case runs, and the one assertion names each case that failed.

A sweep over every public object or page is one test, not one test id per case, so the suite's count tracks its
behaviours; the failure message still lists every offender by name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest


def failures[Case](check: Callable[[Case], object], cases: Iterable[Case]) -> list[str]:
    """``"<case>: <error>"`` for every case on which ``check`` raises, in order; a failure does not stop the rest.

    An empty table fails, as an empty parametrization does (``empty_parameter_set_mark`` in ``pyproject.toml``).
    """
    cases = list(cases)
    if not cases:
        pytest.fail("the table has no case, so the test checks nothing")
    found = []
    for case in cases:
        try:
            check(case)
        except (Exception, pytest.fail.Exception) as error:  # noqa: BLE001 - reported below, with every other case
            found.append(f"{case}: {type(error).__name__}: {error}")
    return found
