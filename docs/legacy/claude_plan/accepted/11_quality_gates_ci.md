# 11 · Quality gates: CI, contract tests, pre-commit, type-check

★★ **Structural.** Operationalises the contract tests already
specified in
[16_TESTING.md §5](../../conventions/16_TESTING.md). Closes the
"contract tests are spec'd but not all wired up" gap noted in
[18_ROADMAP.md §3](../../status/18_ROADMAP.md).

## Problem

[16 §5](../../conventions/16_TESTING.md) lists six contract tests:

1. `test_layer_dependencies.py` — DAG enforcement.
2. `test_public_api.py` — `__all__` ceiling.
3. `test_skeleton_signatures.py` — every symbol importable.
4. `test_hot_path_lint.py` — perf anti-patterns.
5. `test_naming.py` — no Pinocchio cryptic names.
6. `test_docstrings.py` — every public symbol has a docstring.

The codebase has tests with most of these names, but per
[18 §3](../../status/18_ROADMAP.md):

> | `@torch.compile(fullgraph=True)` on FK / Jacobian / CostStack | — | **missing** |
> | `@graph_capture` context manager (docs 14 §5)               | `backends/warp/graph_capture.py` | **stub** |
> | `@cache_kernel` adaptive dispatch (docs 10/14)              | — | **missing** |
> | Hot-path anti-pattern AST linter (docs 14 §6)               | — | **missing** |
> | `BR_PROFILE` env hook                                        | — | **missing** |

`test_hot_path_lint.py` is one of those missing pieces. There is also
no formal CI surface — no GitHub Actions workflow, no pre-commit
hook config, no mypy/pyright config that the contract tests depend
on.

## Goal

Every PR runs every quality gate. Failures block merge. The list of
gates is short and each gate is fast (< 60s for the contract bundle,
< 5min for unit tests, < 15min for the CUDA path).

## The proposal

### 11.A Contract test bundle (the must-pass set)

Locked under `tests/contract/`. Run on every PR:

| Test file | What it checks | Status |
|-----------|----------------|--------|
| `test_layer_dependencies.py` | AST: no module imports anything above its layer in the DAG. | EXISTS — extend per [Proposal 02](02_backend_abstraction.md) |
| `test_public_api.py` | `__all__` matches the EXPECTED set. | EXISTS — update per [Proposal 06](06_public_api_audit.md) |
| `test_skeleton_signatures.py` | Every symbol in `__all__` is importable and has the documented signature. | EXISTS |
| `test_naming.py` | No Pinocchio cryptic names in new code. | EXISTS |
| `test_docstrings.py` | Every `__all__` symbol has a docstring with at least one example. | EXISTS — extend per [Proposal 06 §6.E](06_public_api_audit.md) |
| `test_protocols.py` | Every documented Protocol has the documented members; runtime_checkable stays runtime_checkable. | EXISTS |
| `test_solver_state.py` | `SolverState` carries the spec'd fields. | EXISTS |
| `test_shape_annotations.py` | Public symbols use `jaxtyping`-style annotations on tensor params. | NEW — [Proposal 04](04_typing_shapes_and_enums.md) |
| `test_no_legacy_strings.py` | No `reference="..."` literals in `src/`. | NEW — [Proposal 04](04_typing_shapes_and_enums.md) |
| `test_backend_boundary.py` | Only `lie/`, `kinematics/`, `dynamics/` cross the backend boundary. | NEW — [Proposal 02](02_backend_abstraction.md) |
| `test_hot_path_lint.py` | AST: no `.item()`, `.cpu()`, per-iter alloc, dim-branching, etc. in hot files. | NEW — [16 §5.4](../../conventions/16_TESTING.md) |
| `test_cache_invariants.py` | `Data._kinematics_level` enforced. | NEW — [Proposal 07](07_data_cache_invariants.md) |

The bundle runs in **<60 seconds** because all of it is AST walks,
no GPU, no model loads.

### 11.B Hot-path lint — the missing piece

```python
# tests/contract/test_hot_path_lint.py
"""AST walk over kinematics/, dynamics/, optim/optimizers/ enforcing
the perf anti-patterns from docs/conventions/14_PERFORMANCE.md §3."""
import ast, pathlib

HOT_PATH_FILES = (
    "src/better_robot/kinematics/forward.py",
    "src/better_robot/kinematics/jacobian.py",
    "src/better_robot/dynamics/rnea.py",
    "src/better_robot/dynamics/aba.py",
    "src/better_robot/dynamics/crba.py",
    "src/better_robot/costs/stack.py",
    "src/better_robot/optim/optimizers/levenberg_marquardt.py",
    "src/better_robot/optim/optimizers/gauss_newton.py",
)

FORBIDDEN_ATTRS = ("item", "cpu", "numpy")          # from §3 forbid-sync
FORBIDDEN_CALLS = {                                 # from §3 forbid-hot-alloc
    "torch.zeros", "torch.empty", "torch.ones",
}

def test_no_dim_branching():
    for path in HOT_PATH_FILES:
        tree = ast.parse(pathlib.Path(path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # `x.dim() == 1`, etc. — flag.
                ...

def test_no_sync_methods():
    """No .item() / .cpu() / .numpy() inside HOT_PATH_FILES."""

def test_no_hot_alloc_in_loop():
    """No torch.zeros/empty/ones inside a `for ... in range(...)` body
    in HOT_PATH_FILES."""
```

`# bench-ok: <reason>` comments suppress a violation
([16 §10.3](../../conventions/16_TESTING.md)). The lint counts them; a
PR adding more than three new bench-ok comments fails.

### 11.C Type checks

`mypy --strict` on `src/better_robot/`. Per
[Proposal 04 §4.D](04_typing_shapes_and_enums.md), internal modules
opt out via `pyproject.toml`. CI runs:

```bash
uv run mypy src/better_robot/ --strict
```

Pyright is recommended for editors only; CI uses mypy because it's
more thorough and we want the dispute resolved one way.

### 11.D Lint and format

- `ruff format .` — formatting (drop-in replacement for Black,
  faster).
- `ruff check .` — linting (drop-in replacement for flake8 + isort
  + pyupgrade + several others).
- Both rule sets configured in `pyproject.toml`.

Pre-commit (`pre-commit install`) runs both on every commit. CI
runs them on every PR for redundancy.

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "RUF", "SIM", "PTH", "NPY", "PLR0913"]
ignore = []
```

### 11.E Pre-commit configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff-format
      - id: ruff
        args: [--fix]
  - repo: local
    hooks:
      - id: contract-fast
        name: contract tests (fast subset)
        entry: uv run pytest tests/contract/test_layer_dependencies.py tests/contract/test_naming.py tests/contract/test_public_api.py
        language: system
        types: [python]
        pass_filenames: false
```

The fast subset runs in <10 seconds locally; the full suite runs in
CI.

### 11.F GitHub Actions matrix

```yaml
# .github/workflows/ci.yml
name: ci

on: [pull_request, push]

jobs:
  contract:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
      - run: uv sync --group dev
      - run: uv run pytest tests/contract/ -v

  unit:
    needs: contract
    runs-on: ubuntu-22.04
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
        torch-version: ["stable"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
      - run: uv sync --group dev
      - run: uv run pytest tests/ --ignore=tests/contract --ignore=tests/bench

  cuda:
    needs: contract
    runs-on: gpu-runner          # self-hosted or GitHub Actions GPU
    strategy:
      matrix:
        python-version: ["3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
      - run: uv sync --group dev
      - run: uv run pytest tests/ -m gpu --ignore=tests/bench

  # CPU benchmark — advisory while runner variance is being measured.
  # Records numbers, posts a delta comment on the PR, never fails.
  bench-cpu-advisory:
    needs: contract
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: uv run pytest tests/bench/ -m 'not gpu' --benchmark-compare || true

  # CUDA benchmarks — nightly only at first; promoted to PR-blocking
  # once we have at least one cycle of stable variance data on the
  # self-hosted GPU runner. See Proposal 12 §12.D.
  bench-baseline-nightly:
    if: github.event_name == 'schedule'
    runs-on: gpu-runner
    steps:
      - uses: actions/checkout@v4
      - run: uv run pytest tests/bench/ --benchmark-compare --benchmark-fail=mean:20%

  docs:
    needs: contract
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: uv sync --group docs
      - run: sphinx-build -W -b html docs/site/ build/html/

  type-check:
    needs: contract
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - run: uv run mypy src/better_robot/ --strict
```

The dependency graph: contract first (fast, blocks everything), then
the heavier jobs run in parallel.

**Advisory-then-blocking ladder.** Several gates start advisory (do
not block PRs) and only flip to blocking once they have collected
enough signal:

| Gate | Initial mode | Promotion criterion |
|------|--------------|---------------------|
| Contract bundle | **Blocking from day 1** | — |
| `mypy --strict` | **Blocking from day 1**, scope-limited per [Proposal 04 §4.D](04_typing_shapes_and_enums.md) | — |
| Hot-path lint | **Blocking from day 1** | — |
| `test_shape_annotations.py` | Advisory (coverage report) | Stage 3 of [Proposal 04 §4.A](04_typing_shapes_and_enums.md): all public symbols annotated. |
| CPU bench | Advisory (PR comment) | Two release cycles of stable runner variance < 5% noise floor. |
| CUDA bench | Nightly only | One cycle of stable self-hosted-runner data. |
| `mem_watermark` | Nightly only | Same as CUDA bench. |

Promotion is a separate PR per gate that flips the workflow from
`continue-on-error: true` (or `|| true`) to a real fail. The
flip-PR's description records the variance data that justifies it.

### 11.G Coverage gates

Per [16 §3](../../conventions/16_TESTING.md):

```toml
# pyproject.toml
[tool.coverage.report]
fail_under = 80
exclude_lines = ["raise NotImplementedError", "pragma: no cover"]

[tool.coverage.run]
source = ["src/better_robot"]
omit = ["src/better_robot/dynamics/*"]   # until phase D
```

Per-package thresholds in [16 §3] are enforced by `pytest-cov`
modules in CI.

### 11.H BR_STRICT mode

[17 §7.2](../../conventions/17_CONTRACTS.md) defines `BR_STRICT=1`
which promotes `DeprecationWarning` to an error. Add `BR_STRICT=1`
to the contract job; a PR that adds a deprecated alias call fails CI
unless explicitly suppressed.

## Tradeoffs

| For | Against |
|-----|---------|
| Every contract becomes load-bearing — they're not just specs. | A small one-off setup cost. Once green, adds < 60s to PR latency. |
| Pre-commit catches issues on the contributor side, before CI. | Contributors must run `pre-commit install` once; the README documents this. |
| `mypy --strict` catches signature regressions before review. | Some loud type-check failures during the migration; mitigated by per-module opt-out. |
| Advisory-then-blocking ladder collects variance data before forcing a budget. | Two release cycles of "soft" gating. Mitigation: the contract bundle blocks correctness from day one; the soft gates are perf/coverage signals, not safety. |
| The CI matrix covers Python 3.10–3.12 × stable PyTorch + CUDA. | Cost ~20 minutes of runner time per PR. Mitigation: contract job blocks the rest. |

## Acceptance criteria

- `tests/contract/` runs in < 60 seconds and includes every test in
  the table above.
- `.github/workflows/ci.yml` exists and runs contract + unit + CUDA
  + docs + type-check on every PR.
- `pre-commit install` produces a working hook that runs ruff and
  the fast contract subset.
- `BR_STRICT=1` is set in the contract job; a PR using a deprecated
  alias fails.
- Coverage gates fail PRs that drop a per-package threshold.
- README documents `pre-commit install` as the contributor on-ramp.

## Cross-references

- [16_TESTING.md](../../conventions/16_TESTING.md) — the spec these
  proposals operationalise.
- [Proposal 02](02_backend_abstraction.md) — adds
  `test_backend_boundary.py`.
- [Proposal 04](04_typing_shapes_and_enums.md) — adds
  `test_shape_annotations.py`, `test_no_legacy_strings.py`.
- [Proposal 06](06_public_api_audit.md) — updates the public-API
  test EXPECTED set.
- [Proposal 07](07_data_cache_invariants.md) — adds
  `test_cache_invariants.py`.
- [Proposal 12](12_regression_and_benchmarks.md) — wires the
  benchmark gate.
