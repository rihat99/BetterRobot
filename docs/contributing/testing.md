# Testing

The test suite is the contract: Pinocchio parity tests pin the numbers, contract tests pin the
architecture, and docs tests keep every page runnable. This page shows how to run it and what each
group guards.

## Run the suite

```bash
UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q -m "not bench and not cuda"
uv run pytest tests/kinematics -q            # one package while you edit
uv run ruff check src/ tests/
```

| Marker | Meaning |
|--------|---------|
| `slow` | long-running tests; skip with `-m "not slow"` |
| `cuda` | needs a CUDA device |
| `bench` | micro-benchmarks in `tests/bench/`; advisory, not part of the contract |

Tests that need optional packages (`robot_descriptions`, `pin`, `mujoco`, `viser`) call
`pytest.importorskip`, so a minimal install skips them instead of failing. The `dev` extra installs
all of them.

## Pinocchio parity is a gate

`tests/test_pinocchio/` compares kinematics, Jacobians and dynamics against Pinocchio on the same
URDFs. The tolerances are part of the contract:

| Inputs | Typical `atol` |
|--------|----------------|
| float64 | `1e-12` to `1e-9` for kinematics and Jacobians; up to `2e-6` for a few dynamics cases (`test_rnea_missing_joint_kinds.py`, `test_centroidal.py`) |
| float32 | `2e-6` |

A parity failure means the library changed its answer. Fix the code, not the tolerance.

## Contract tests

`tests/contract/` guards properties that no unit test sees:

| Test | Guards |
|------|--------|
| `test_layer_dependencies.py` | imports follow the package graph (`viewer -> terms -> core`, `optim` imports only torch) |
| `test_public_api.py` | the top-level `better_robot` API surface |
| `test_submodule_public_imports.py` | symbols documented under `better_robot.<submodule>` import from there |
| `test_docstrings.py` | every public symbol has a docstring |
| `test_boundary_validation_count.py` | public functions validate inputs once; raw passes below do not re-validate |
| `test_hot_path_lint.py` | no `torch.compile`-breaking patterns (host syncs, `.item()`, Python branching on tensors) in kinematics and dynamics |
| `test_naming.py` | old Pinocchio-style storage names do not come back |
| `test_optional_imports.py` | optional dependencies are imported lazily at their integration boundary |
| `test_protocols.py` | extension protocols keep their shape (see [Extending](extending.md)) |
| `test_roadmap_stub_inventory.py` | every `NotImplementedError` stub is listed in the [roadmap](../reference/roadmap.md) |

`tests/autograd/` checks that gradients reach `q` and the model tensors through every public
kinematics and dynamics function (see [Differentiate](../guides/differentiate.md)).

## Docs snippets run

`tests/docs/test_snippets.py` runs every `python` block of each page (and `README.md`) in order, in one
namespace, on CPU, with a fake viewer server, 120 s per page. It also checks that every image, poster and
clip a page references exists. The rules for a page:

| Fence | Meaning |
|---|---|
| ```` ```python ```` | runs in CI, in page order |
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
