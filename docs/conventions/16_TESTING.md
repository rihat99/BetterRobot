# Testing Strategy

> **Status:** normative. Every PR is expected to satisfy this doc.

## 1 · What we test, and why

| Category | Purpose | Tolerance for breakage |
|----------|---------|------------------------|
| **Unit** | Smallest provable behaviours (e.g. `se3.exp ∘ log == identity`) | Zero — a unit failure is a bug |
| **Integration** | Cross-module correctness (e.g. `solve_ik` + Panda URDF) | Zero |
| **Contract** | DAG, public-API ceiling, registry sanity | Zero |
| **Regression** | Numerical equivalence against a frozen reference output | Tolerance: 1e-5 (fp32), 1e-10 (fp64) |
| **Benchmark** | Wall-clock budgets from [14_PERFORMANCE.md §1](14_PERFORMANCE.md) | 20% regression window |
| **Example** | Every runnable `examples/*.py` imports and runs headless | Zero |

No single category is "optional." CI runs all six on every PR.

## 2 · Directory layout

```
tests/
├── lie/                                # unit (SE3/SO3, tangents, Jr/Jl)
├── spatial/                            # unit (Motion/Force/Inertia ops)
├── data_model/                         # unit (joint models, Model/Data roundtrip)
├── kinematics/                         # unit + regression
├── dynamics/                           # unit + regression (phase D)
├── residuals/                          # unit (analytic vs FD jacobian per residual)
├── costs/                              # unit (CostStack shape/weight/active)
├── optim/                              # unit (solvers on synthetic problems)
├── tasks/                              # integration (Panda, G1 IK; trajopt later)
├── io/                                 # integration (URDF / MJCF roundtrip)
├── collision/                          # unit (SDF pairs)
├── viewer/                             # unit with MockBackend, smoke test with ViserBackend
├── bench/                              # benchmarks (see §6)
├── contract/
│   ├── test_layer_dependencies.py      # DAG enforcement (AST walk)
│   ├── test_backend_boundary.py        # only lie/kinematics/dynamics cross the backend Protocol
│   ├── test_public_api.py              # __all__ matches frozen EXPECTED set (26 symbols)
│   ├── test_submodule_public_imports.py # symbols not in __all__ stay reachable from documented submodule paths
│   ├── test_skeleton_signatures.py     # every public symbol is importable
│   ├── test_hot_path_lint.py           # perf anti-patterns (see 14_PERFORMANCE §3)
│   ├── test_naming.py                  # no pinocchio cryptic names in new code (see 13_NAMING)
│   ├── test_docstrings.py              # every public symbol has a docstring + example
│   ├── test_protocols.py               # documented Protocols carry the documented members
│   ├── test_solver_state.py            # SolverState carries the spec'd fields
│   ├── test_shape_annotations.py       # jaxtyping coverage on public surface (advisory until stage 3)
│   ├── test_no_legacy_strings.py       # no reference="..." literals in src/
│   ├── test_cache_invariants.py        # Data._kinematics_level enforced; in-place mutation limitation codified
│   ├── test_optional_imports.py        # `import better_robot` does not pull yourdfpy/mujoco/viser/warp/...
│   └── test_deprecations.py            # deprecation shims removed on schedule
└── examples/
    └── test_examples.py                # imports and runs each examples/*.py headless
```

## 3 · Coverage budgets per layer

Enforced by `pytest-cov --cov-fail-under`:

| Layer | Minimum line coverage |
|-------|-----------------------|
| `lie/` | 95% |
| `spatial/` | 95% |
| `data_model/` | 90% |
| `kinematics/` | 90% |
| `residuals/` | 90% |
| `costs/`, `optim/` | 85% |
| `tasks/` | 80% |
| `io/` | 80% |
| `collision/` | 85% |
| `viewer/` | 60% (hard to test headlessly) |
| `dynamics/` | 80% *once the body is implemented*; phase-D only |

"Below budget" fails CI. We never lower a budget to pass; we add tests.

## 4 · Test style

### 4.1 No mocks for numerical code

Real Panda URDF (`robot_descriptions.panda_description.URDF_PATH`), real
floating-base G1, real MJCF where applicable. Mocking breaks the
reference numerics.

### 4.2 Parametrise over shapes

Every hot-path test runs at **(CPU, fp32), (CPU, fp64), (CUDA, fp32)**
and at **batch sizes 1, 8, 1024** where memory permits:

```python
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 8, 1024])
def test_fk_shape(panda, device, dtype, batch):
    ...
```

A helper `tests/conftest.py::gpu` auto-skips CUDA-marked tests on CI
runners without a GPU.

### 4.3 Manifold-aware assertions

For tensors that live on SE(3) / SO(3), comparison uses a dedicated
helper:

```python
from better_robot.utils.testing import assert_close_manifold
assert_close_manifold(T_got, T_ref, atol=1e-5, rtol=1e-5)
```

Why not `torch.testing.assert_close`? Two unit quaternions `±q`
represent the same rotation; a raw element-wise compare would falsely
fail. The helper normalises both and compares via geodesic distance.

### 4.4 Analytic vs. autodiff equivalence

Every residual with a hand-coded `.jacobian()` ships a test that
compares the analytic Jacobian to `torch.func.jacrev` of the residual:

```python
def test_pose_residual_jacobian_matches_autodiff(panda_fk_data):
    res = PoseResidual(target_pose, frame_id)
    J_analytic = res.jacobian(state)
    J_auto = torch.func.jacrev(lambda x: res(state.with_x(x)))(state.x)
    assert_close(J_analytic, J_auto, atol=1e-4)
```

This is the contract the library relies on for performance; it must
never silently drift.

### 4.5 Regression oracles

`tests/kinematics/fk_reference.npz` holds FK outputs for Panda and G1
at a pinned set of 50 random `q`s — fp64 throughout. A regression test
fails if the current FK diverges from the stored reference beyond
`atol=1e-10`, unless the developer *intentionally* regenerates the file
via `tests/kinematics/_generate_fk_reference.py`.

Metadata captured in the npz:

```
oracle_version : "1.0"            # bumped when the FK convention changes
generation_seed: 0                 # the only thing that selects the q's
generated_with : "torch=2.5.0, …"  # pinned dep versions for reproducibility
fk_dtype       : "float64"
generated_at   : "2026-04-25T12:34:56Z"
```

**No git SHA** — reproducibility hinges on
`oracle_version + generation_seed + dep versions`, not on a SHA that
becomes useless after a rebase. When the FK convention itself changes,
`oracle_version` bumps and the new file is committed.

A sibling `_pinocchio_oracle.npz` (when Pinocchio is available — opt-in
via the `[test]` extra; test `skipif`s otherwise) is generated *from*
Pinocchio with the same `q` samples. The cross-check test compares both
oracles to within `1e-8` (fp64), which makes the FK convention provably
identical to Pinocchio's.

The regression files are cheap (~50 KB per robot) and catch subtle
changes in Lie conventions, joint-model fixes, and quaternion-sign-bit
flips.

### 4.6 Gate-promotion ladder

Several gates land **advisory** and only flip to blocking once they have
collected enough signal:

| Gate | Initial mode | Promotion criterion |
|------|--------------|---------------------|
| Contract bundle (correctness, DAG, hot-path lint, mypy strict, cache invariants, optional imports) | **Blocking from day 1** | — |
| `test_shape_annotations.py` (jaxtyping coverage) | Advisory (coverage report) | Stage 3 of [Proposal 04 §4.A](../legacy/claude_plan/accepted/04_typing_shapes_and_enums.md): all public symbols annotated |
| CPU bench | Advisory (PR comment) | Two release cycles of stable runner variance < 5% |
| CUDA bench | Nightly only | One cycle of stable self-hosted-runner data |
| `mem_watermark` | Nightly only | Pre-1.0 stays nightly; promotion at v1 release |
| Pinocchio cross-check | Opt-in (skip if missing) | Always opt-in until we ship a hosted Pinocchio runner |

Promotion is a separate PR per gate that flips the workflow from
`continue-on-error: true` (or `|| true`) to a real fail. The flip-PR's
description records the variance data that justifies it.

Hard CUDA gates *before* runner stability is measured produce flaky CI
that gets muted, defeating the gate's purpose. The ladder is the
discipline that prevents that.

## 5 · Contract tests (DAG, API, naming, lint)

These tests don't exercise numerical code; they enforce the **rules**
elsewhere in `docs/`. They are deliberately separated in
`tests/contract/` so they're the first thing CI reports.

### 5.1 Layer DAG

`test_layer_dependencies.py` AST-parses every `src/better_robot/**/*.py`
and walks the import graph. Any edge that violates the DAG (see
[01_ARCHITECTURE.md §Dependency rule](../design/01_ARCHITECTURE.md)) fails.
Reported with the offending file and import line number.

### 5.2 Public API

`test_public_api.py`:
- `better_robot.__all__` matches the **frozen `EXPECTED` set** (26 symbols).
  Adding/removing public symbols requires updating `EXPECTED` in the same
  PR — the audit is the diff, not a magic number.
- Each is importable.
- Each has a non-empty docstring with at least one example block.
- Each class is a `@dataclass` or has `__slots__` (memory discipline).

`test_submodule_public_imports.py`:
- The documented submodule-only public symbols stay reachable:

```python
from better_robot.lie         import SO3, Pose
from better_robot.spatial     import Motion, Force, Inertia, Symmetric3
from better_robot.kinematics  import ReferenceFrame
from better_robot.optim.state import SolverState
from better_robot.tasks.ik    import IKResult, IKCostConfig, OptimizerConfig
```

If a refactor accidentally moves one, the test fails — silent moves are
caught.

### 5.3 Naming

`test_naming.py` greps `src/better_robot/` for any identifier matching
the deprecated-name patterns in [13_NAMING.md §4](13_NAMING.md)
*outside* the migration shim allowlist (`data_model/data.py` during the
deprecation window).

### 5.4 Hot-path lint

`test_hot_path_lint.py` AST-walks the designated hot-path files and
fails on the patterns listed in
[14_PERFORMANCE.md §3](14_PERFORMANCE.md): `.item()`, `.cpu()`,
per-iteration `torch.zeros`, branching on `tensor.dim()`, etc.

### 5.5 Docstring coverage

`test_docstrings.py` asserts every symbol in `__all__`, every `Protocol`,
and every `@register_*` decorator has a docstring. The check runs at
import time.

## 6 · Benchmarking (how numbers become gates)

Benchmarks live under `tests/bench/` and use `pytest-benchmark`.

### 6.1 Running

```bash
uv run pytest tests/bench/ -v --benchmark-only
uv run pytest tests/bench/ --benchmark-compare --benchmark-fail=mean:20%
```

### 6.2 Baseline management

- `tests/bench/baseline_cpu.json` — committed, auto-bumped on perf PRs.
- `tests/bench/baseline_cuda.json` — committed per hardware generation
  (RTX 40-series today; future GPUs get a new file).

A perf PR pipeline:
1. Run baseline: `pytest --benchmark-save=new`.
2. Compare against committed: `pytest --benchmark-compare=baseline_cpu`.
3. If win > 5%, commit `baseline_cpu.json` bump in the same PR.
4. If loss > 20%, PR fails; investigate or justify.

### 6.3 Budgets from [14_PERFORMANCE.md](14_PERFORMANCE.md)

Each benchmark asserts its own budget:

```python
def test_bench_panda_fk_cuda_b1(benchmark, panda):
    q = panda.q_neutral.cuda()
    r = benchmark(lambda: br.forward_kinematics(panda, q))
    assert benchmark.stats.mean < 150e-6   # § 14.1
```

## 7 · Example-as-test

Every file in `examples/` must be importable from `tests/examples/test_examples.py`:

```python
@pytest.mark.parametrize("example", [
    "01_basic_ik", "02_g1_ik",
])
def test_example_imports(example):
    mod = importlib.import_module(f"examples.{example}")
    assert hasattr(mod, "main")

@pytest.mark.parametrize("example", [...])
@pytest.mark.gpu
def test_example_runs_headless(example, monkeypatch):
    # disable viser, run main() with max_iter=3
    ...
```

Rationale: docs that bit-rot take users down; tests that import every
example prevent this for free. cuRobo does the same.

## 8 · CI matrix

| Axis | Values |
|------|--------|
| OS | Ubuntu 22.04, macOS 14 |
| Python | 3.10, 3.11, 3.12 |
| PyTorch | stable, nightly |
| Device | CPU (all matrices), CUDA (Ubuntu + Python 3.11 + stable) |
| dtype | fp32 in most tests; fp64 in `tests/lie/`, `tests/kinematics/` |

A full matrix is ~30 jobs; we run the reduced matrix (one OS × one
Python × both devices) on every PR and the full matrix nightly. PR
failures in the reduced matrix block merge; nightly failures open
issues.

## 9 · What "green" means

A PR is "ready to merge" when:

- All tests in all six categories pass in the reduced CI matrix.
- No coverage budget dropped (§3).
- No benchmark regressed beyond the window (§6.2).
- All contract tests green (§5).
- For a docs-only PR: contract tests + example imports still green.

No green, no merge — including for documentation-only PRs (the
contract tests guard against copy-pasted cryptic names in new docs).

## 10 · Debugging failures

### 10.1 Numerical drift in regression

1. Check whether `model.gravity`, `joint_placements`, `body_inertias`
   were regenerated (URDF hash changed).
2. Diff the analytic and autodiff Jacobians on the offending frame.
3. Re-run with `dtype=torch.float64` — if the drift disappears, the
   root cause is fp32 quantization and the tolerance may be raised
   with a justifying note in the test.

### 10.2 Flaky benchmark

Benchmarks use `@pytest.mark.benchmark(warmup=True, min_rounds=10)`.
If variance is high:
1. Check GPU is at fixed clocks (`nvidia-smi -q -d PERFORMANCE`).
2. Pin the process to a single NUMA node (`numactl --cpunodebind=0`).
3. Re-run with `--benchmark-autosave` and inspect the histogram.

### 10.3 Hot-path lint false positive

If a legitimate `.item()` is needed (e.g. one-time shape extraction in
construction), add a `# bench-ok: <reason>` comment on that line. The
lint pass respects it but records the count; a PR adding more than
**three** new `bench-ok` comments fails.

## 11 · Writing a new test — style guide

- **One concept per file.** Not `test_kinematics.py`; instead
  `test_forward_kinematics_shape.py`, `test_jacobian_equivalence.py`.
- **Fixtures in `conftest.py`.** Robot loaders, `q_neutral`,
  pre-computed `Data` — share via fixtures, not copy-paste.
- **No `@pytest.mark.skip` without a linked issue.** A skipped test
  is a bug with a mask on. If it's not, remove it.
- **Assert on meaning, not numbers.** `assert_close_manifold`, not
  element-wise `torch.testing.assert_close` for SE(3)-valued outputs.
- **Tight tolerances.** fp32: 1e-5 atol, 1e-5 rtol; fp64: 1e-10.
  Looser bounds are flags, not conveniences — explain them in a
  comment.
