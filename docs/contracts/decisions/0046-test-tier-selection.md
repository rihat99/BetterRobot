# 0046 The fast tier leaves out `cuda`; the full tier keeps the browser test and the docs runs

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record [0033](0033-test-tiers.md) selects the fast tier as the contract tests without `slow`. Contract tests marked
`cuda` then ran in the pre-commit hook and failed on a machine without a GPU. The full tier takes most of its time in
two places: the viewer screenshot test, which drives a headless Chromium, and the docs and examples runs, which load
their robots through the `robot_descriptions` cache (the tests themselves load tracked fixtures, record
[0039](0039-test-robot-fixtures.md)). Leaving both to the release tier was proposed to shorten the full run. The
no-skip rule of 0033 was a hook in BetterRobot's own `conftest.py`, which BetterHuman copied.

## Decision
We will select the fast tier as `tests/contract -m "not slow and not cuda"`, keep the Chromium screenshot test and the
docs and examples runs in the full tier, and ship the no-skip rule as the pytest plugin
`better_robot.testing.no_skips` (with the `testing` extra), which both repositories load from their root
`conftest.py`.

## Example
```bash
uv run pytest tests/contract -q -m "not slow and not cuda"              # fast
uv run pytest tests -q -m "not bench and not cuda"     # full (BetterHuman, which has licensed tests, adds "and not licensed")
```
```python { .no-run }
# tests/conftest.py
pytest_plugins = ["better_robot.testing.no_skips"]
```

## Consequences
The fast tier runs on any machine. The full tier needs a Chromium and a filled `robot_descriptions` cache, and fails
when either is missing (a missing oracle fails, 0033); the browser path is named by `BETTER_ROBOT_CHROME`. Its time is
recorded with each run. BetterHuman's skip rule cannot drift from BetterRobot's.

## Alternatives
Move the browser test and the docs and examples runs to the release tier: a shorter full run, but the viewer's
recording and every page's snippets would run only by hand. Keep the hook as copied code: two copies drift.

## Compatibility
Test selection only. A user of `better_robot.testing` installs the `testing` extra to load the plugin.

## Enforcement
`tests/contract/test_pytest_config.py`; `tests/contract/test_no_skips.py`; `docs/guides/run-the-tests.md` (the tier
commands).

## Open questions
None.
