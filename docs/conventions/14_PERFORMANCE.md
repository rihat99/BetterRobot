# 14 · Performance

> **Status:** normative goals + informative guidance.

This doc answers three questions:
1. **What performance must the library deliver at v1 and v1.x?** (budgets)
2. **How is performance won?** (techniques — kernel fusion, graph capture,
   adaptive dispatch, memory reuse)
3. **How is performance measured and regressed?** (benchmarks, CI guard)

It is the authority for any perf-related decision elsewhere in the docs.
If another doc contradicts this one, this one wins.

## 1 · Performance budgets (targets, not promises)

### 1.1 Hot-path budgets (CUDA, fp32, Panda-class arm, single batch)

| Operation | Target (RTX 4090 / L40) | Notes |
|-----------|-------------------------|-------|
| `forward_kinematics(model, q)` | **≤ 150 µs** per call, `B=1` | Topological walk over 7 joints |
| `forward_kinematics(model, q)` | **≤ 250 µs** per call, `B=1024` | Batched; cost is mostly launch overhead |
| `compute_joint_jacobians(model, data)` | **≤ 300 µs** per call, `B=1` | Analytic, world frame |
| `get_frame_jacobian(model, data, fid)` | **≤ 50 µs** per call | Cache hit — pure gather + rotate |
| `solve_ik(model, targets, max_iter=30)` | **≤ 8 ms** per call, `B=1` | 30 LM iterations, pose cost only |
| `solve_ik(...)` | **≤ 40 ms** per call, `B=1024` | Batched warm-starts; same iter count |

Humanoid (`G1`, 36 DoF, floating-base):

| Operation | Target | Notes |
|-----------|--------|-------|
| `forward_kinematics(model, q)` | **≤ 400 µs** per call, `B=1` | 30+ joints |
| `solve_ik(model, targets, max_iter=60)` | **≤ 25 ms** per call, `B=1` | Whole-body, pose + limits + rest |

### 1.2 CPU targets (fp32, single thread)

CPU is the fallback path. We don't optimise for it aggressively but we
don't let it rot either.

| Operation | Target (Threadripper 5995WX / M3 Pro) |
|-----------|---------------------------------------|
| `forward_kinematics(model, q)`, `B=1` | ≤ 1 ms |
| `solve_ik`, Panda, `B=1`, 30 iters | ≤ 50 ms |

### 1.3 Memory budgets

| Quantity | Target |
|----------|--------|
| `Model` (tensors only) for Panda | ≤ 50 KiB |
| `Data` for `B=1024, njoints=8` | ≤ 5 MiB |
| Peak GPU working set during IK, `B=1024` | ≤ 200 MiB |

### 1.4 Non-targets

We do **not** set targets for:
- First-call compile time (may include `torch.compile` warmup).
- Mesh/URDF load time (once per process; bounded by disk I/O).
- Viewer render loop (measured separately; see 12).

## 2 · How performance is won

### 2.1 Batching is the default, not a mode

Every hot-path function accepts `(B..., feature)` tensors and walks the
robot topology **once** per call, regardless of `B`. No Python loop over
the batch axis. No `if` on `tensor.dim()`. See
[10_BATCHING_AND_BACKENDS.md](../design/10_BATCHING_AND_BACKENDS.md).

### 2.2 Static topology, dynamic values

`Model` is frozen (`@dataclass(frozen=True)`). `model.topo_order`,
`model.parents`, `model.joint_models` are Python tuples. They **unroll**
under `torch.compile`: the FK kernel compiles once per (shape, dtype,
device) triple, not per query.

### 2.3 Analytic derivatives where it matters

- FK Jacobian: **analytic** (one pass, 1 ulp accuracy).
- Pose residual Jacobian: **analytic via `Jr_inv(log_err)`** — no
  autodiff through `Log`, kept analytic for accuracy and speed (see
  [03_LIE_AND_SPATIAL.md §Numerical stability](../design/03_LIE_AND_SPATIAL.md)).
- Position / orientation residual Jacobians: analytic slices of the FK
  Jacobian.
- RNEA / ABA Jacobians (phase D3): **analytic via Carpentier & Mansard**.
  Not autodiff.
- Everything else: autodiff or central finite differences as a fallback.
  `JacobianStrategy.AUTO` dispatches.

**Rationale:** analytic derivatives are 3–10× faster than autodiff for
rigid-body routines and 100× faster than finite differences.

### 2.4 Kernel fusion (torch.compile boundary)

`@torch.compile(fullgraph=True)` is applied at **three boundaries**:

1. `forward_kinematics.inner(q, joint_placements, ...)` — the topo walk.
2. `compute_joint_jacobians.inner(joint_pose_world_stack, motion_subspaces, ...)`.
3. `CostStack.__call__.inner(state)` — residual concatenation loop.

Each compiled region:
- has no Python control flow on tensor values,
- has no `.item()`, `.cpu()`, `.numpy()`,
- has no `new_tensor`, no un-guarded `.to(device, dtype)`,
- caches its compiled artefact per `(nq, nv, njoints, dtype, device)` key.

The **outer Python** code — `Model` construction, `Data` allocation,
solver iteration — stays eager. Optimisers that call the compiled FK
many times per step inherit fusion for free.

### 2.5 Adaptive kernel dispatch (cuRobo pattern)

For routines whose cost depends on a discrete size parameter — number of
collision spheres, trajectory horizon — the library picks a specialised
kernel at first call.

| Regime | Kernel |
|--------|--------|
| `n_spheres ≤ 100` | single-fused-kernel SDF (one launch) |
| `n_spheres > 100` | dual-kernel (per-body transform + per-pair distance) |
| `horizon ≤ 32` | rolled Python loop inside one compile block |
| `horizon > 32` | scan-style parallel prefix kernel |

Dispatch is keyed on the `Model` (or `LeastSquaresProblem`) identity, so
each problem compiles once.

### 2.6 CUDA graph capture for hot solver loops

Once LM (or any solver) is warm, its per-iteration call graph is stable.
We wrap the iteration in a `@graph_capture` context:

```python
with graph_capture(replay_n=max_iter) as ctx:
    step_fn = optimizer.step_compiled(problem, state)
    for k in range(max_iter):
        state = step_fn(state)
```

Under the hood: first iteration records the CUDA graph; subsequent
iterations replay it. Typical win: 30–50% latency reduction for
`max_iter ≥ 10`, `B=1`.

Graph capture is **opt-in** and lives in `backends/warp/graph_capture.py`
behind a flag — it is not on by default because it interacts with
autograd (replay nukes the grad tape).

### 2.7 Memory reuse and matrix-free trajopt

- `Data` is allocated once per problem; fields are reset in place via
  `Data.reset()` rather than allocating new tensors.
- The LM solver pre-allocates `JᵀJ` and `Jᵀr` buffers sized from the
  residual spec (see [07_RESIDUALS_COSTS_SOLVERS.md](../design/07_RESIDUALS_COSTS_SOLVERS.md)).
- Trajectory optimisation preallocates the per-knot cost buffer. No
  per-iteration `torch.zeros`.
- **Matrix-free trajopt.** Long horizons (T > 100) never materialise the
  dense Jacobian. Adam and L-BFGS read
  `LeastSquaresProblem.gradient(x)` (J^T r matrix-free); temporal
  residuals override `apply_jac_transpose` for banded `J^T r` in
  `O(T·nv)` memory. See [07 §4](../design/07_RESIDUALS_COSTS_SOLVERS.md).
  This is what keeps a 200-knot G1 trajopt under the 200 MiB CUDA peak
  watermark in §1.3.

### 2.8 Sparse Jacobian for collision

Self-collision residuals are sparse: only the two kinematic chains
connecting a colliding pair contribute non-zero columns. The collision
residual exposes a **sparsity mask** (see
[09_COLLISION_GEOMETRY.md](../design/09_COLLISION_GEOMETRY.md)); the LM solver
skips zero blocks in the normal-equation assembly. Measured speed-up on
G1: ~6× for the collision-Jacobian step.

### 2.9 Warp backend (future)

Phase `W` migrates the three hot kernels (FK, spatial Jacobian, RNEA) to
`warp.kernel`. Contract:
- User sees `torch.Tensor` in and out. Conversion lives in
  `backends/warp/bridge.py`.
- Differentiability: `torch.autograd.Function.apply` wraps the kernel; the
  backward is a second Warp kernel (analytic, hand-written).
- Graph capture extends from the torch path to the Warp path with no
  API change.

No public surface depends on Warp being available; `import warp` is
local to `backends/warp/`.

## 3 · Performance anti-patterns (forbidden)

These patterns ship with a failing CI check. Each has a linter rule.

| Pattern | Rule | Bad | Good |
|---------|------|-----|------|
| Branching on `tensor.dim()` | `forbid-dim-branch` | `if x.dim()==2: ... else: ...` | Rely on leading-batch convention; `batch_shape = x.shape[:-1]` |
| `.item()` / `.cpu()` in hot path | `forbid-sync` | `if loss.item() < tol` | `if (loss < tol).all()` after graph-wide reduce |
| Per-joint `.to(device, dtype)` | `forbid-redundant-to` | `for j: S_j.to(device, dtype)` | Move once during `Model.to(...)` and cache |
| `torch.zeros` in solver iteration | `forbid-hot-alloc` | `for k in range(N): torch.zeros(...)` | Allocate in `create_data` / `SolverState`, reset in place |
| Python `if` on tensor value | `forbid-tensor-cond` | `if data.mass_matrix.det() > 0: ...` | Use `torch.where` or assert as contract |
| `torch.cat` inside a fused region | `prefer-stack` | `cat([a, b, c])` (allocates view) | `stack` + reshape where shapes match |

The linter lives in `tests/test_hot_path_lint.py` (AST walks the
`kinematics/` and `optim/` trees).

## 4 · Measurement — how we know

### 4.1 Microbenchmarks (`tests/bench/`)

One file per public operation. Shape:

```python
# tests/bench/bench_forward_kinematics.py
def test_panda_fk_cpu_b1(benchmark):      ...  # ≤ 1 ms
def test_panda_fk_cuda_b1(benchmark):     ...  # ≤ 150 µs
def test_panda_fk_cuda_b1024(benchmark):  ...  # ≤ 250 µs
```

All targets from §1 are encoded as `benchmark.extra_info["budget_us"]`
and asserted.

### 4.2 Regression guard — advisory-then-blocking ladder

Benchmarks land **advisory** and only flip to blocking once they have
collected enough signal:

| Gate | Initial mode | Promotion criterion |
|------|--------------|---------------------|
| Contract bundle (correctness, DAG, hot-path lint, mypy strict, cache invariants, optional imports) | Blocking from day 1 | — |
| `test_shape_annotations.py` | Advisory (coverage report) | Stage 3 of typing migration |
| CPU bench | Advisory (PR comment) | Two release cycles of stable runner variance < 5% |
| CUDA bench | Nightly only | One cycle of stable self-hosted-runner data |
| `mem_watermark` | Nightly only, advisory | Pre-1.0 stays nightly; promotion at v1 release |

Once promoted, `pytest tests/bench/ --benchmark-compare
--benchmark-fail=mean:20%` against the committed baseline
(`tests/bench/baseline_cpu.json`, `tests/bench/baseline_cuda_l40.json`) is
the gate. Hard CUDA gates *before* runner stability is measured produce
flaky CI that gets muted; the ladder above is the discipline that prevents
that. See [16_TESTING.md](16_TESTING.md) and the implementation phases in
[../UPDATE_PHASES.md](../UPDATE_PHASES.md).

Baseline is bumped only when:
- A performance PR improves the number (new lower bound).
- A hardware change is announced and all budgets re-measured.

### 4.3 Profiling (opt-in)

`BR_PROFILE=1 python ...` enables `torch.profiler` with NVTX ranges
around every compiled region. Output is consumable in Chrome Trace or
Nsight Systems. No runtime cost when the flag is off.

### 4.4 Memory watermark

`tests/bench/mem_watermark.py` runs the hot scenarios under
`torch.cuda.memory._record_memory_history()` and asserts peak GPU
working set against §1.3.

## 5 · Compile/JIT lifecycle

### 5.1 Cold start

On first call, every `@torch.compile` block records shapes and compiles.
On Panda at `B=1024`, cold start adds ~800 ms to the first FK call.
This is a one-time cost, not repeated across queries.

### 5.2 Recompile triggers

Recompilation happens when:
- `B` (the batch prefix) changes across queries → compile per shape.
  Mitigated by the shape-specialisation cache; if the user cycles through
  many batch sizes we fall back to dynamic shapes (slower, no recompile).
- `dtype` / `device` changes → new cache entry.
- `Model` topology changes → never, because `Model` is frozen.

### 5.3 Cache location

`TORCHINDUCTOR_CACHE_DIR` (default `~/.cache/torch_inductor/better_robot`).
For CI, set to a persistent path to avoid re-compiling across jobs.

## 6 · Per-module performance ownership

| Module | Owns | Primary optimisation technique |
|--------|------|--------------------------------|
| `backends/` | Backend Protocol; per-backend kernel implementations | Single-dispatch through Protocol; explicit `backend=` kwargs avoid global state in compiled code |
| `lie/` | SE3/SO3 group ops, typed wrappers | Pure-PyTorch (default); short closed-form; compile-friendly |
| `spatial/` | 6D operators | Dataclass wrappers; no branching |
| `kinematics/forward.py` | FK topo walk | `@torch.compile(fullgraph=True)`; unroll on `topo_order` |
| `kinematics/jacobian.py` | Spatial Jacobian | Analytic; compiled |
| `dynamics/*.py` | RNEA/ABA/CRBA | Analytic derivatives (phase D6); compile-friendly recursion |
| `residuals/*.py` | Pure functions | Analytic `.jacobian()` and `apply_jac_transpose` where possible; `ResidualSpec` advertises sparsity |
| `costs/stack.py` | Concatenation | Flat buffer, write-into-slice |
| `optim/optimizers/*.py` | Solver loops | Pre-allocated buffers; graph-capture-ready; matrix-free path via `problem.gradient(x)` |
| `optim/linear_solvers/*.py` | Linear solves | `torch.linalg.cholesky_ex`; fall back to `lstsq`; block-Cholesky for trajopt |
| `tasks/parameterization.py` | Trajectory parameterisations | B-spline gives ~4× fewer optim variables than knot-based |
| `collision/*.py` | SDF pairs | Sparsity-aware residual; vectorised pairs; stable `dim` across iterations |
| `io/*.py` | One-shot parse | Not hot; readability > speed; `AssetResolver` Protocol |
| `viewer/*.py` | Scene updates | 60 fps budget (see [12_VIEWER.md](../design/12_VIEWER.md)) |

## 7 · Checklist for a "fast" PR

Every PR that claims "perf" answers **all** of these in the description:

1. What measurement changed? (cite the benchmark file and number)
2. Is the change portable across GPU generations, or RTX-specific?
3. Does it introduce recompilation when batch or dtype cycles?
4. Is the memory watermark unchanged or better?
5. Does `pytest tests/bench/` pass with the new baseline?
6. Is there a new lint rule needed to keep the win?

If any answer is "don't know," the PR is not ready.
