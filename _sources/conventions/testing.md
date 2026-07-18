# Testing

> **Status:** normative test expectations plus a description of current
> automation. The workflow is manual-only; this page does not imply an
> automatic pull-request, scheduled, coverage, benchmark, or GPU gate.

Tests are part of BetterRobot's compatibility surface. Numerical tests pin
robotics conventions and gradients, contract tests pin architecture and API
rules, and benchmark definitions keep performance claims reproducible. The
test files and ``.github/workflows/ci.yml`` are the source of truth for what is
executed; this guide explains how those pieces fit together.

## 1 · Test categories

| Category | Current role |
|---|---|
| **Unit** | Small tensor, manifold, model, residual, solver, and viewer behaviours. |
| **Integration** | Public operations on programmatic and real robot descriptions, including task facades and Pinocchio comparisons. |
| **Contract** | Layer DAG, public imports, naming, optional imports, cache invariants, protocols, roadmap inventory, and hot-path rules. |
| **Regression** | Frozen numerical output such as the committed FK oracle. |
| **Benchmark** | Advisory definitions and evidence artifacts; not a merge gate. |
| **Documentation** | The published front-page example plus strict Sphinx builds. |

The repository currently has these top-level test areas:

```text
tests/
├── bench/             # advisory microbenchmarks and measurement contracts
├── contract/          # source/API/architecture contracts
├── data_model/
├── docs/              # executable published examples
├── dynamics/
├── io/
├── kinematics/        # includes fk_reference.npz
├── lie/
├── optim/
├── residuals/
├── spatial/
├── tasks/
├── test_pinocchio/    # optional Pinocchio oracles
├── viewer/
└── warp/              # optional Warp CPU and CUDA coverage
```

Small programmatic models are appropriate for isolating topology and shape
contracts. Tests that claim compatibility with a real robot or Pinocchio use
the corresponding real description/reference rather than a mock substitute.

## 2 · Current automation

GitHub Actions is intentionally stopped for automatic events. The ``CI``
workflow has only a ``workflow_dispatch`` trigger and must be started by a
maintainer. It currently defines five Ubuntu/Python 3.12 jobs:

| Manual job | What it runs |
|---|---|
| ``full-non-warp`` | Full non-Warp, non-benchmark, non-CUDA suite with development/integration extras. |
| ``core-only`` | Fresh core install, public import, and dependency-boundary assertions. |
| ``docs`` | Published front-page test and Sphinx HTML build. |
| ``benchmarks`` | Advisory pytest benchmarks plus the M1 eager-CPU harness; uploads evidence without comparing a blocking baseline. |
| ``warp-cpu`` | Bounded Warp-CPU contracts and the Warp-FK/compiled-Torch definition; this is correctness evidence, not CUDA evidence. |

There is no automatic PR or nightly matrix, hosted CUDA job, coverage
threshold, static-type job, or blocking benchmark comparison at present. CUDA
validation recorded for M6 was run on the documented local host; see
``tests/bench/definitions.md``. Re-enabling or promoting a remote gate is a
separate owner decision.

## 3 · Running locally

Install the extras needed by the surface under test. A close equivalent of
the manual non-Warp job is:

```bash
uv sync --extra dev --extra viewer --extra io-mjcf --extra meshes --extra demos
uv run pytest tests \
  --ignore=tests/warp \
  --ignore=tests/data_model/test_layout_aliasing.py \
  -m "not bench and not cuda"
```

The ignored layout test imports Warp. With the optional runtime installed, run
the bounded Warp-CPU contracts separately:

```bash
uv sync --extra dev --extra warp
uv run pytest tests/data_model/test_layout_aliasing.py tests/warp -q \
  --ignore=tests/warp/test_fk_cuda.py
```

CUDA-marked tests require a visible CUDA device and their declared optional
dependencies. They skip when CUDA is unavailable; a skip is not GPU evidence.
On the M6 measurement host, the focused suite is:

```bash
uv run pytest tests/optim/test_graph_capture.py tests/warp/test_fk_cuda.py -q
```

The default agent sandbox on that host hides NVIDIA device nodes. A CUDA result
is valid only from the approved host context described in
``tests/bench/definitions.md``.

For documentation and benchmarks:

```bash
uv run pytest tests/docs/test_front_page.py -q
uv run make -C docs strict
uv run pytest tests/bench/bench_*.py -m bench --benchmark-only
```

``uv lock --check`` verifies that the committed lock still matches project
metadata. Run ``git diff --check`` before committing documentation or code.

## 4 · Coverage

``pyproject.toml`` configures branch-aware coverage for
``better_robot`` when ``pytest-cov`` is requested:

```bash
uv run pytest tests --cov=better_robot --cov-report=term-missing
```

No ``fail_under`` value or per-layer budget is configured, and the manual CI
workflow does not currently collect coverage. Coverage is therefore a review
signal, not an enforced gate. Add tests for changed behaviour rather than
treating a percentage as the only adequacy criterion.

## 5 · Numerical and differentiation style

### 5.1 Shapes, batches, devices, and dtypes

Test the combinations promised by the boundary being changed. Public tensor
paths normally need unbatched and representative leading-batch cases, fp32
and fp64 where supported, and dtype/device preservation. Value-batched work
uses ``tests/conftest.py::assert_value_batched_matches_loop`` to compare one
batched call with scalar calls over the resolved execution batch.

CUDA-specific behaviour is marked ``@pytest.mark.cuda`` and guarded by an
explicit availability skip. Do not report a CPU or skipped run as CUDA
validation.

### 5.2 Manifold-aware comparisons

Quaternions have a double cover, so ``q`` and ``-q`` represent the same
rotation. Tests of rotational meaning should compare a relative rotation or
geodesic error. Raw component comparisons are appropriate only when the
component representation itself is the contract, as in a frozen algorithm
regression oracle.

### 5.3 Derivative checks

Use the oracle that matches the claimed derivative:

- compare declared analytic named-block Jacobian blocks with ``jacrev`` and/or
  ``jacfwd``;
- use ``torch.autograd.gradcheck`` in fp64 for differentiable boundaries and
  singular seams;
- use finite differences when they are the independent reference, with a
  documented step and tolerance;
- include broadcast/shared-value reductions when a compute lane owns a custom
  VJP.

Not every legacy residual has an analytic Jacobian. The legacy
``JacobianStrategy.AUTO`` prefers an analytic method and otherwise uses its
documented unbatched central-finite-difference fallback. The named-block
``Problem`` surface separately supports analytic, ``jacrev``, ``jacfwd``, and
finite-difference strategies.

### 5.4 Frozen FK regression oracle

``tests/kinematics/fk_reference.npz`` stores fp64 Panda and G1 FK outputs for
a pinned set of configurations. The Panda fixture intentionally preserves the
historical full-coordinate gripper representation rather than public reduced
mimic coordinates; the generator and regression test strip the mimic metadata
the same way. ``test_fk_regression.py`` compares current joint and frame
placements at ``atol=rtol=1e-10``. Regenerate it only after an intentional
algorithm/convention change:

```bash
uv run python tests/kinematics/_generate_fk_reference.py
```

Review the resulting binary diff together with the generator metadata. There
is no sibling committed ``_pinocchio_oracle.npz``. Live Pinocchio comparisons
are under ``tests/test_pinocchio/`` and skip if their optional dependency is
unavailable.

## 6 · Contract tests

``tests/contract/`` contains the current executable rules:

```text
test_boundary_validation_count.py
test_cache_invariants.py
test_cost_stack_location.py
test_docstrings.py
test_hot_path_lint.py
test_layer_dependencies.py
test_naming.py
test_no_legacy_strings.py
test_optional_imports.py
test_pluggable_protocols.py
test_protocols.py
test_public_api.py
test_roadmap_stub_inventory.py
test_solver_state.py
test_submodule_public_imports.py
```

Run them directly while changing architecture, exports, optional imports, or
documentation tied to the roadmap:

```bash
uv run pytest tests/contract -q
```

The hot-path lint AST-walks the paths declared in
``test_hot_path_lint.py``. A legitimate eager/static boundary may use
``# bench-ok: <reason>``; the exemption is reviewed as ordinary source code.
There is no rule that fails merely because more than a fixed number of such
comments were added.

## 7 · Benchmarks and evidence

Benchmarks are advisory. The canonical definitions and artifact status live
in ``tests/bench/README.md`` and ``tests/bench/definitions.md``. In
particular:

- ``baseline_cpu.json`` is placeholder pytest-benchmark scaffolding and is not
  a regression gate;
- hardware-named CUDA JSON files are evidence only for the exact host, commit,
  workload, dtype, and batch they record;
- the M6 144-selector matrix is defined but only a filtered subset has been
  measured;
- ``test_mem_watermark.py`` is available for explicit local/manual runs and is
  not scheduled nightly.

Never replace an unsupported workload with an easier one under the same
label. Record raw samples, cold-start policy, synchronization, versions,
hardware identity, and source commit before quoting a ratio.

## 8 · Documentation and examples

``tests/docs/test_front_page.py`` executes the fenced example published from
``docs/index.md``. Sphinx's strict target catches internal references and
warnings. The repository does not currently import and execute every file in
``examples/`` automatically, so examples changed outside the front page need
an explicit smoke run or a focused test.

## 9 · What a verified change means

A change is ready for review when the tests relevant to its claimed surface
pass in a suitable environment and the result is reported precisely. Examples:

- a docs-only change: front-page test when touched, strict Sphinx, contract
  tests affected by documented inventories, and ``git diff --check``;
- a Torch numerical change: focused unit/regression/gradient tests plus the
  non-CUDA suite in the supported environment;
- a Warp change: shared Torch-oracle parity and Warp-CPU contracts, plus real
  CUDA tests before making any CUDA or performance claim;
- a performance change: functional tests plus a fresh artifact following the
  relevant committed measurement definition.

Because CI is manual-only, “verified locally” and “passed a manually
dispatched workflow” are distinct statements. Neither should be described as
an automatic merge gate.

## 10 · Debugging failures

For numerical drift, first separate dtype tolerance from a convention change,
then compare the analytic/autodiff blocks or current/frozen oracle at the
smallest failing model and batch. Do not regenerate a frozen oracle merely to
make an unexplained failure disappear.

For noisy benchmarks, preserve the committed workload and inspect affinity,
warmup, synchronization, compiler/kernel caches, GPU clocks, and raw-sample
distribution. A quick or filtered harness run is a smoke test unless the
definition explicitly promotes it to canonical evidence.

For an optional-dependency failure, reproduce in the matching core-only,
integration, Warp-CPU, or CUDA environment. Importing ``better_robot`` must
remain independent of viewer, MJCF, demo, and Warp extras.
