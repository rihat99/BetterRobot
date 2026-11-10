# Run the tests

The test suite is the contract: Pinocchio parity tests pin the numbers, contract tests pin the architecture, and
docs tests keep every page runnable. This guide shows how to run it; [Test suite](../reference/tests.md) lists what
each group guards.

## Run a tier

Install everything the tests need with one command: `uv sync --all-extras --all-groups` (the extras, the `dev`
tools and the `test-oracles` group with Pinocchio and SciPy). Then each tier is one command:

| Tier | Command | Time | Run by |
|------|---------|------|--------|
| fast | `uv run pytest tests/contract -q -m "not slow and not cuda"` | about 15 s | the pre-commit hook, after every edit |
| full | `uv run pytest tests -q -m "not bench and not cuda"` | about 4 min (219 s) | the pre-push hook, the release checklist |
| release | `CUDA_VISIBLE_DEVICES=<gpu> uv run pytest tests -q -m cuda` | seconds | by hand on a machine with a free GPU |

The tiers are those of records [0033](../contracts/decisions/0033-test-tiers.md) and
[0046](../contracts/decisions/0046-test-tier-selection.md); the counters that 0033 adds to the release tier are not
in the repository yet. Beside the tiers, the pre-push hook runs the coverage gate of `lie` and `spatial`, both
torch-only mathematics with no robot in sight, so every line and branch is reachable from a unit test and the gate
holds them at 100 % (about 5 s):

```bash
uv run coverage run --branch --source=src/better_robot/lie,src/better_robot/spatial -m pytest tests/lie tests/spatial -q -m "not cuda" && uv run coverage report --fail-under=100
```

Every tier runs the internal invariants (decision record 0021): `tests/conftest.py` sets
`BETTER_ROBOT_INTERNAL_CHECKS=1` unless the environment sets it, so a library bug that breaks an invariant fails
a test. The benchmarks set it to `0`: an invariant reads tensor values, a host sync per call.

While you edit one package, run the fast tier plus that package's folder, for example
`uv run pytest tests/kinematics -q`. The hooks live in `.pre-commit-config.yaml`;
`uv run pre-commit install` installs them.

## What the full tier needs

Nothing skips: the pytest plugin `better_robot.testing.no_skips`, which `tests/conftest.py` loads (BetterHuman's too),
turns every skip or xfail into a failure, whatever spells it, so a missing oracle, fixture or browser fails the test
(`tests/contract/test_no_skips.py` plants each spelling). The plugin needs pytest, the `testing` extra. The test
robots are tracked under `tests/fixtures/robots/`, so the unit, contract and parity tests run offline. The full tier
does not: the docs pages (`tests/docs/test_snippets.py`) and the examples (`tests/examples/test_examples_run.py`) load
robots through `robot_descriptions`, which downloads each robot into `~/.cache` on first use and needs that cache from
then on. One full-tier test,
`tests/viewer/test_capture.py::test_record_screenshots_a_real_scene`, screenshots a real scene through a headless
Chromium: put `chromium`, `google-chrome` or `chrome-headless-shell` on `PATH`, or point `BETTER_ROBOT_CHROME` at
one. On a machine without `libasound.so.2` Chromium exits at start; point `BETTER_ROBOT_CHROME` at a wrapper script
that adds a copy of the library to `LD_LIBRARY_PATH` (for example extracted from the `libasound2` package) and
then runs the browser. Both stay in the full tier on purpose (record
[0046](../contracts/decisions/0046-test-tier-selection.md)): the pages, the examples and the screenshot are what a
reader runs and sees.

## Check the docs pages

`tests/docs/test_snippets.py` runs every `python` fence of each page and compares every `output` fence with what
the fence prints (the rules are in [Test suite](../reference/tests.md#docs-tests)):

```bash
uv run pytest tests/docs -q -k load-a-robot                                     # one page
BR_DOCS_CAPTURE=1 BR_DOCS_CAPTURE_DIR=/tmp/out uv run pytest tests/docs -q -k load-a-robot   # write real outputs
uv run mkdocs build --strict                                                    # broken links fail the build
BETTER_ROBOT_CHROME=<chromium> node tools/docs_screenshots.mjs /tmp/shots       # every page as light/dark PNGs
```

The capture run writes `<page>.outputs.json` with each fence's output; copy it into the `output` fence. Look at
the screenshots of the pages you changed before you open a pull request.

## Never weaken a test

Do not skip, `xfail`, loosen a tolerance or delete an assertion to land a change. If a parity,
architecture or contract test fails, the change is wrong or the contract must change on purpose; in
the second case update the test, the docs and the nearest `CLAUDE.md` together, and say so in the
pull request.

## Where next

- [Test suite](../reference/tests.md): the markers, the parity tolerances, every contract test and the docs rules.
- [Extend BetterRobot](extend.md): the five kinds of brick and their tests.
- [Design contracts](../contracts/00-index.md): the rules the contract tests enforce.
