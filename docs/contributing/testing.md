# Testing

The test suite is the contract: Pinocchio parity tests pin the numbers, contract tests pin the
architecture, and docs tests keep every page runnable. This page shows how to run it and what each
group guards.

## Run the suite

Install everything the tests need with one command: `uv sync --all-extras --all-groups` (the extras, the `dev`
tools and the `test-oracles` group with Pinocchio and SciPy). Then each tier is one command:

| Tier | Command | Time | Run by |
|------|---------|------|--------|
| fast | `uv run pytest tests/contract -q -m "not slow and not cuda"` | about 2 s | the pre-commit hook, after every edit |
| full | `uv run pytest tests -q -m "not bench and not cuda"` | about 3 min (184 s) | the pre-push hook |
| CUDA | `CUDA_VISIBLE_DEVICES=<gpu> uv run pytest tests -q -m cuda` | seconds | by hand on a machine with a GPU |
| release | `uv run coverage run --branch --source=src/better_robot/lie,src/better_robot/spatial -m pytest tests/lie tests/spatial -q -m "not cuda" && uv run coverage report --fail-under=100` | about 5 s | the pre-push hook, the release checklist |

The release row is the coverage gate of `lie` and `spatial`: both are torch-only mathematics with no robot in
sight, so every line and branch is reachable from a unit test, and the gate holds them at 100 %.

Every tier runs the internal invariants (decision record 0021): `tests/conftest.py` sets
`BETTER_ROBOT_INTERNAL_CHECKS=1` unless the environment sets it, so a library bug that breaks an invariant fails
a test. The benchmarks set it to `0`: an invariant reads tensor values, a host sync per call.

While you edit one package, run the fast tier plus that package's folder, for example
`uv run pytest tests/kinematics -q`. The hooks live in `.pre-commit-config.yaml`;
`uv run pre-commit install` installs them.

| Marker | Meaning |
|--------|---------|
| `slow` | takes seconds; deselected in the fast tier, run in the full tier |
| `cuda` | needs a CUDA device; deselected in the fast and full tiers |
| `bench` | micro-benchmarks in `tests/bench/`; deselected in every tier |

Nothing skips: the pytest plugin `better_robot.testing.no_skips`, which `tests/conftest.py` loads (BetterHuman's too),
turns every skip or xfail into a failure, whatever spells it, so a missing oracle, fixture or browser fails the test
(`tests/contract/test_no_skips.py` plants each spelling). The plugin needs pytest, the `testing` extra. The test
robots are tracked under `tests/fixtures/robots/`, so the unit, contract and parity tests run offline. The full tier
does not: the docs pages (`tests/docs/test_snippets.py`), the examples (`tests/examples/test_examples_run.py`) and the
benchmark smoke test (`tests/bench/test_run_smoke.py`) load robots through `robot_descriptions`, which downloads each
robot into `~/.cache` on first use and needs that cache from then on. One full-tier test,
`tests/viewer/test_capture.py::test_record_screenshots_a_real_scene`, screenshots a real scene through a headless
Chromium: put `chromium`, `google-chrome` or `chrome-headless-shell` on `PATH`, or point `BETTER_ROBOT_CHROME` at
one. On a machine without `libasound.so.2` Chromium exits at start; point `BETTER_ROBOT_CHROME` at a wrapper script
that adds a copy of the library to `LD_LIBRARY_PATH` (for example extracted from the `libasound2` package) and
then runs the browser.

## Pinocchio parity is a gate

`tests/parity/pinocchio/` compares kinematics, Jacobians and dynamics against Pinocchio on the same
URDFs; it is part of the full tier and runs alone with `uv run pytest tests/parity/pinocchio -q`. The tolerances
are part of the contract:

| Inputs | Typical `atol` |
|--------|----------------|
| float64 | `1e-12` to `1e-9` for kinematics and Jacobians; up to `2e-6` for a few dynamics cases (`test_rnea_missing_joint_kinds.py`, `test_centroidal.py`) |
| float32 | `2e-6` |

A parity failure means the library changed its answer. Fix the code, not the tolerance.

## Contract tests

`tests/contract/` guards properties that no unit test sees:

| Test | Guards |
|------|--------|
| `test_layer_dependencies.py` | every import between packages is in the `ALLOWED` edge table (`optim` imports only `exceptions`) |
| `test_public_surface.py` | the root `__all__` is the decided list; every name of the `__all__` tree resolves |
| `test_submodule_public_imports.py` | only the decided packages exist; no raw `*_raw` kernel is public |
| `test_docstrings.py` | every public object has a docstring that gives each tensor parameter's shape |
| `test_boundary_validation_count.py` | public functions validate inputs once; raw passes below do not re-validate |
| `test_hot_path_lint.py` | no `torch.compile`-breaking patterns (host syncs, `.item()`, Python branching on tensors) in kinematics and dynamics |
| `test_naming.py` | old Pinocchio-style storage names do not come back |
| `test_optional_imports.py` | each package imports only the outside packages `pyproject.toml` declares for it; `lie`, `spatial` and `optim` import torch only |
| `test_no_skips.py` | the `no_skips` plugin fails every skip and xfail, whatever spells it |
| `test_pytest_config.py` | pytest runs strictly: unknown markers and config keys, passing xfails, empty parameter sets and new warnings fail |
| `test_python_version.py` | `.python-version`, `requires-python` and ruff's target agree on Python 3.12 |
| `test_fixtures.py` | every test robot has its licence and provenance |
| `test_citations.py` | every repository path and `path::symbol` cited in tracked text resolves; no line number is cited |
| `test_protocols.py` | extension protocols keep their shape (see [Extending](extending.md)) |
| `test_roadmap_stub_inventory.py` | every `NotImplementedError` stub is listed in the [roadmap](../reference/roadmap.md) |

A rule that still has known violations keeps them in a ratchet, `tests/contract/ratchets/<name>.json`: each key
is one violation with its reason or witness. A ratchet only shrinks; the fix that removes a violation removes its
entry, and a stale-entry test fails until it does.

`tests/autograd/` checks that gradients reach `q` and the model tensors through every public
kinematics and dynamics function (see [Differentiate](../guides/differentiate.md)).

## Docs snippets run

`tests/docs/test_snippets.py` runs every `python` block of each page (and `README.md`) in order, in one
namespace, on CPU, with a fake viewer server, 120 s per page. It also checks that every image, poster and
clip a page references exists. The rules for a page:

| Fence | Meaning |
|---|---|
| ```` ```python ```` | runs in the full tier, in page order |
| ```` ```python { .no-run } ```` | shown but not run (licensed data, a live server) |
| ```` ```output ```` | directly after a block that prints: its real output |

```bash
uv run pytest tests/docs -q -k load-a-robot                                     # one page
BR_DOCS_CAPTURE=1 BR_DOCS_CAPTURE_DIR=/tmp/out uv run pytest tests/docs -q -k load-a-robot   # write real outputs
uv run mkdocs build --strict                                                    # broken links fail the build
BETTER_ROBOT_CHROME=<chromium> node scripts/docs_screenshots.mjs /tmp/shots     # every page as light/dark PNGs
```

The capture run writes `<page>.outputs.json` with each block's stdout; paste it into the `output` fence. Look at
the screenshots of the pages you changed before you open a pull request.

## Never weaken a test

Do not skip, `xfail`, loosen a tolerance or delete an assertion to land a change. If a parity,
architecture or contract test fails, the change is wrong or the contract must change on purpose; in
the second case update the test, the docs and the nearest `CLAUDE.md` together, and say so in the
pull request.

## Where next

- [Extending BetterRobot](extending.md): the four extension points and their tests.
- [Design contracts](../contracts/00-index.md): the rules the contract tests enforce.
