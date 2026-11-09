"""The pytest plugin that fails every skip and xfail (principle 6.5, decision record 0033).

Load it from a test suite's root ``conftest.py`` with ``pytest_plugins = ["better_robot.testing.no_skips"]``. A skipped
or xfailed gate reports green while checking nothing, so each one becomes a failure that names it. The hooks read the
test reports, so every spelling fails alike: ``pytest.skip``, an aliased ``pytest`` or ``mark``, ``unittest.skip``,
``raise SkipTest``, ``importorskip`` (in a test or at collection) and ``xfail`` of either strictness. A test that
cannot run on every machine carries a marker its tier command deselects (``cuda``; BetterHuman's ``licensed``). This
module imports pytest (the ``testing`` extra), so ``better_robot.testing`` loads it only when it is asked for.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

__all__: list[str] = []


def _fail_skip(report: pytest.TestReport | pytest.CollectReport) -> None:
    """Turn a skip or xfail into a failure that names it."""
    if hasattr(report, "wasxfail"):
        what = f"xfail ({report.wasxfail or 'no reason given'})"
        del report.wasxfail
    elif report.skipped:
        what = f"skip ({report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr})"
    else:
        return
    report.outcome = "failed"
    report.longrepr = (
        f"{report.nodeid}: {what}. No test skips or xfails (BetterRobot decision record 0033): mark a test that "
        "needs a GPU or licensed weights with the marker its tier deselects; a missing oracle, fixture or browser "
        "is an error to fix."
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport() -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Fail a test whose setup, call or teardown skipped or xfailed."""
    report = yield
    _fail_skip(report)
    return report


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_make_collect_report() -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    """Fail a module that skips at collection (``importorskip`` or ``skip(allow_module_level=True)``)."""
    report = yield
    _fail_skip(report)
    return report
