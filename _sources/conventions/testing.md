# Testing

Tests protect mathematical conventions as well as Python behavior. A passing
shape test is not enough when a change can alter a frame convention,
quaternion sign, gradient, or solver status.

The test files and `.github/workflows/ci.yml` are the source of truth. The
workflow is started manually; there is no automatic pull-request, scheduled,
coverage, benchmark, or hosted-CUDA gate.

## Test layers

| Layer | What it protects |
|---|---|
| unit | one tensor formula, class, or failure mode |
| integration | several public layers on a complete robot |
| contract | imports, dependency direction, cache rules, protocols, and documented inventories |
| regression | a reviewed numerical output that should not drift |
| parity | an independent implementation such as Pinocchio |
| gradient | analytic or automatic derivatives |
| benchmark | reproducible measurements; currently advisory |
| documentation | examples and Sphinx references |

Use a small programmatic model to isolate a rule. Use a real robot description
when the claim concerns a real model or an external reference.

## The normal local gate

Install the development extras needed by the changed surface, then run:

```bash
UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q +  -m "not bench and not cuda"
```

The manually started non-Warp workflow uses a fresh CPU installation and
skips the Warp folder plus the one layout test that imports Warp. A close
local equivalent is:

```bash
uv sync --extra dev --extra viewer --extra io-mjcf --extra meshes --extra demos
uv run pytest tests +  --ignore=tests/warp +  --ignore=tests/data_model/test_layout_aliasing.py +  -m "not bench and not cuda"
```

Also run `uv lock --check` when project metadata changes and
`git diff --check` before handing off a patch.

## Optional runtimes

Warp CPU checks need the `warp` extra:

```bash
uv sync --extra dev --extra warp
uv run pytest tests/data_model/test_layout_aliasing.py tests/warp -q +  --ignore=tests/warp/test_fk_cuda.py
```

CUDA-marked tests require a visible CUDA device and their optional
dependencies. A skip is not CUDA evidence. Report the GPU model, relevant
software versions, commit, dtype, and selected tests with every CUDA result.
The normal sandbox can hide device nodes, so do not infer a GPU failure from
that environment.

## Shapes, batches, devices, and dtypes

Test the combinations promised by the boundary:

- unbatched input and representative leading batches;
- shared values that broadcast across another batch;
- fp32 and fp64 where supported;
- dtype and device preservation; and
- a clear failure for incompatible event shapes.

`assert_value_batched_matches_loop` in `tests/conftest.py` compares one
batched call with scalar calls over the same resolved execution batch.

## Rotation-aware comparisons

`q` and `-q` describe the same rotation. Compare a relative rotation,
rotation matrix, or geodesic error when rotational meaning is the contract.
Compare raw quaternion components only when storage itself is under test.

State the tolerance and why it is appropriate for the dtype and operation.
Do not reuse an fp32 tolerance for an fp64 identity without checking the
expected error.

## Derivative tests

Choose an independent reference:

- compare an analytic Jacobian with `jacrev` or `jacfwd`;
- use `torch.autograd.gradcheck` in fp64 at differentiable boundaries;
- use `gradgradcheck` for a claimed second derivative;
- use finite differences as a debugging reference with a stated step; and
- test reduction back to the original shape for broadcasted custom gradients.

Include small-angle branches and samples near, but not exactly on, genuine
nonsmooth points. Do not require a derivative at the quaternion principal-log
cut, a robust-loss kink, or an active-set change.

## Frozen FK data

`tests/kinematics/fk_reference.npz` stores reviewed fp64 Panda and G1 FK
outputs. The matching regression test uses tight tolerances. Regenerate the
file only for an intentional convention or algorithm change:

```bash
uv run python tests/kinematics/_generate_fk_reference.py
```

Review the generator metadata and binary diff together. Live Pinocchio
comparisons are under `tests/test_pinocchio/` and skip when the optional
dependency is unavailable.

## Contract tests

`tests/contract/` checks rules that are easy to break without changing one
numerical fixture: boundary validation counts, cache levels, docstrings,
dependency direction, naming, optional imports, protocols, public exports,
stub inventory, and hot-path source rules.

Run that folder while changing architecture, exports, validation, optional
imports, or the roadmap:

```bash
uv run pytest tests/contract -q
```

The hot-path check watches the paths listed in its test source. A necessary
eager decision may carry `# bench-ok: <reason>`; the reason is reviewed like
any other code.

## Documentation

`tests/docs/test_front_page.py` executes the marked example on the landing
page. The custom-residual test executes the marked guide example. When a page
path or marker changes, update the corresponding test in the same patch.

```bash
uv run pytest tests/docs/test_front_page.py tests/optim/test_vertical_slice.py -q
uv run sphinx-build -b html docs docs/_build/html
```

Strict Sphinx builds catch broken internal references and undocumented
orphans. Examples outside the executable markers need either a focused test
or an explicit smoke run.

## Coverage

`pytest-cov` can report branch coverage:

```bash
uv run pytest tests --cov=better_robot --cov-report=term-missing
```

No minimum percentage is configured. Coverage is a review aid, not a
substitute for tests of mathematical invariants and failure behavior.

## Benchmarks

Benchmarks are advisory. Definitions and artifact requirements live in
`tests/bench/README.md` and `tests/bench/definitions.md`.

A useful performance report includes:

- the unchanged workload definition;
- raw samples and synchronization policy;
- warm or cold compile policy;
- CPU/GPU identity and software versions;
- dtype, batch, and robot model; and
- the source commit.

Do not replace an unsupported workload with an easier one under the same
label. A benchmark becomes a gate only after the repository has stable runner
evidence and explicitly enables that comparison.

## Reporting a verified change

Say exactly what ran and where:

- “focused CPU tests passed” is different from the full non-CUDA suite;
- “CUDA tests skipped” is not a CUDA pass;
- “verified locally” is different from a manually started remote workflow;
- a smoke benchmark is not a canonical measurement.

When a test fails numerically, reduce it to the smallest model and batch before
changing a tolerance or fixture. Never regenerate a reference merely to hide
an unexplained difference.
