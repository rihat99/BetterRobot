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
| float64 | `1e-12` to `1e-9` |
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

`tests/docs/test_snippets.py` runs every `python` block of each page in order, in one namespace, on
CPU, with a fake viewer server, 120 s per page. Tag a block that must not run with `python no-run`.

```bash
uv run pytest tests/docs -q -k load-a-robot   # one page
uv run mkdocs build --strict                  # broken links fail the build
```

## Never weaken a test

Do not skip, `xfail`, loosen a tolerance or delete an assertion to land a change. If a parity,
architecture or contract test fails, the change is wrong or the contract must change on purpose; in
the second case update the test, the docs and the nearest `CLAUDE.md` together, and say so in the
pull request.
