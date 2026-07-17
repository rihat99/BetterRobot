# Performance

> **Status:** normative goals + informative guidance.

A robotics library that is "correct but slow" is a teaching tool, not a
production dependency. Inverse kinematics for an interactive viewer
needs to close the loop in milliseconds; trajectory optimisation for a
humanoid needs to fit a 200-knot horizon in a single GPU's memory; the
benchmarks that prove either one of those keeps shipping have to run in
finite CI time. Performance is therefore not a nice-to-have add-on —
it is a contract the library has to defend on every PR.

We defend it with three things working together. The first is a set of
**budgets** — concrete latency and memory targets that say "FK on a
Panda must complete in 150 µs on an L40, full stop." Without that
number, "is this fast enough?" becomes opinion; with that number, it
becomes a passing or failing benchmark. The second is a small set of
**techniques** that earn the budget back: leading-batch shapes that
avoid Python loops, frozen `Model` topology that lets `torch.compile`
unroll cleanly, analytic Jacobians for the routines that dominate the
hot path, and representations that avoid dense Jacobians where that contract
is implemented. Today, named-block Adam is matrix-free, temporal LM can use
block-banded normal storage, and explicit operator LM uses `NormalCG`.
Undeclared problems retain the dense correctness path; these capabilities are
not by themselves a benchmark-certified long-horizon performance claim. The
third is a
**gate-promotion ladder** — benchmarks land
advisory, collect signal, and only flip to blocking once their variance
is low enough that flipping does not produce flaky CI.

When another doc seems to imply a different perf rule, this one wins.
The targets here are the contract; the techniques here are how the
contract is met; the lint rules are the discipline that keeps it met.

## 1 · Performance budgets (targets, not promises)

### 1.1 Hot-path budgets (CUDA, fp32, Panda-class arm, single batch)

| Operation | Target (RTX 4090 / L40) | Notes |
|-----------|-------------------------|-------|
| `forward_kinematics(model, q)` | **≤ 150 µs** per call, `B=1` | Topological walk over 7 joints |
| `forward_kinematics(model, q)` | **≤ 250 µs** per call, `B=1024` | Batched; cost is mostly launch overhead |
| `compute_joint_jacobians(model, data)` | **≤ 300 µs** per call, `B=1` | Analytic, world frame |
| `get_frame_jacobian(model, data, fid)` | **≤ 50 µs** per call | Cache hit — pure gather + rotate |
| `solve_ik(model, targets, max_iter=30)` | **≤ 8 ms** per call, `B=1` | 30 LM iterations, pose cost only |

Humanoid (G1, 36 DoF, floating-base):

| Operation | Target | Notes |
|-----------|--------|-------|
| `forward_kinematics(model, q)` | **≤ 400 µs** per call, `B=1` | 30+ joints |
| `solve_ik(model, targets, max_iter=60)` | **≤ 25 ms** per call, `B=1` | Whole-body, pose + limits + rest |

### 1.2 CPU targets (fp32, single thread)

CPU is the fallback path. We do not optimise for it aggressively but we
do not let it rot.

| Operation | Target (Threadripper 5995WX / M3 Pro) |
|-----------|---------------------------------------|
| `forward_kinematics(model, q)`, `B=1` | ≤ 1 ms |
| `solve_ik`, Panda, `B=1`, 30 iters | ≤ 50 ms |

### 1.3 Memory budgets

| Quantity | Target |
|----------|--------|
| `Model` (tensors only) for Panda | ≤ 50 KiB |
| `Data` for `B=1024, njoints=8` | ≤ 5 MiB |
| Batched IK working set, `B=1024` | ≤ 200 MiB (tracked target; not yet benchmark-certified) |

### 1.4 Non-targets

We do **not** set targets for:

- First-call compile time (may include `torch.compile` warmup).
- Mesh / URDF load time (once per process; bounded by disk I/O).
- Viewer render loop (measured separately; see {doc}`/concepts/viewer`).

## 2 · How performance is won

### 2.1 Batching is the default, not a mode

Tensor kernels such as FK, residual evaluation, and analytic Jacobians
accept `(B..., feature)` tensors and walk the robot topology **once** per
call, regardless of `B`. Named-block Adam/LM/GN, `solve_ik`, and
`solve_trajopt` preserve the same leading axes with per-element state; legacy
flat optimizers remain single-problem. See
{doc}`/concepts/batching_and_backends`.

### 2.2 Static topology, dynamic values

``ModelStructure`` carries immutable topology in two equivalent forms:
Python tuples for statically unrolled Torch loops and flat device tables for
whole-pass kernels. ``ModelValues`` carries the differentiable tensors as a
registered pytree. Raw passes consume these two objects explicitly, so a
compiled function does not discover tensor inputs through mutable model
state.

### 2.3 Analytic derivatives where it matters

- FK Jacobian: **analytic** (one pass).
- Pose residual Jacobian: **analytic via `Jr_inv(log_err)`** — no
  autodiff through `Log`, kept analytic for accuracy and speed.
- Position / orientation residual Jacobians: analytic slices of the FK
  Jacobian.
- RNEA / ABA Jacobians: spec'd analytic via Carpentier–Mansard; the
  current shipping body uses autograd through the differentiable RNEA /
  ABA / CRBA passes (live, gradcheck-clean) and will switch to the
  analytic recursion in a future minor release.
- Everything else: unbatched central finite differences as the current
  fallback. `JacobianStrategy.AUTO` prefers analytic evaluation; real
  `torch.func` strategies are scheduled for M2.

**Rationale:** analytic derivatives are 3–10× faster than autodiff for
rigid-body routines and 100× faster than finite differences.

### 2.4 Kernel fusion (torch.compile boundary)

No automatic `@torch.compile` decorator is installed today.
`forward_kinematics_raw` is tested as compatible with an explicit caller-side
`torch.compile(..., fullgraph=True)` wrapper. The next intended compilation
boundaries are:

1. `forward_kinematics.inner(q, joint_placements, ...)` — the topo walk.
2. `compute_joint_jacobians.inner(joint_pose_world_stack, motion_subspaces, ...)`.
3. `CostStack.__call__.inner(state)` — residual concatenation loop.

The Jacobian and `CostStack` boundaries remain roadmap work. The outer
Python — `Model` construction, `Data` allocation, and solver iteration —
stays eager.

### 2.5 Adaptive kernel dispatch

For routines whose cost depends on a discrete size parameter — number
of collision spheres, trajectory horizon — the library picks a
specialised kernel at first call.

| Regime | Kernel |
|--------|--------|
| `n_spheres ≤ 100` | single-fused-kernel SDF (one launch) |
| `n_spheres > 100` | dual-kernel (per-body transform + per-pair distance) |
| `horizon ≤ 32` | rolled Python loop inside one compile block |
| `horizon > 32` | scan-style parallel prefix kernel |

Dispatch is keyed on the `Model` (or `LeastSquaresProblem`) identity,
so each problem compiles once.

### 2.6 CUDA graph capture for hot solver loops

CUDA graph capture is roadmap work, not a helper that ships today. A
capture-ready solver needs fixed storage, a warm-up phase, explicit
invalidation when shapes or storage change, and a replay lifecycle that
records forward **and backward together**. Capturing only the forward pass
would not preserve the intended autograd work on replay.

Named-block LM now supplies the fixed-structure tensor state and pure,
sync-free `update` required by that checklist; the CPU fullgraph smoke is only
a graph-break proxy. Actual capture remains opt-in until M6 records and
replays the full solver lifecycle and any custom-kernel adjoints with parity.
There is currently no public capture decorator or context manager.

### 2.7 Current allocation and matrix-free limits

- Named-block `Adam` differentiates `Problem.objective` through tangent
  retractions and does not assemble a Jacobian. This is the shipped
  matrix-free first-order path.
- Named-block LM/GN retain dense assembly as the correctness route. A problem
  with one declared temporal block can instead assemble block-banded normal
  storage or use the explicit `NormalOperator`/`NormalCG` route.
- `optim/cost_stack.py` assembles legacy residuals and Jacobians with
  `torch.cat`; it does not own a persistent flat buffer.
- Knot-based `solve_trajopt` uses route-aware named-block LM and reports the
  chosen dense or banded path. Undeclared residuals fall back to dense;
  structured routing is never inferred from numerical zeros. The legacy Adam
  and L-BFGS classes remain direct-use compatibility APIs. Manifold-safe spline
  integration is still deferred.

The memory values in §1.3 are tracked targets, not evidence that a 200-knot
trajectory solve currently meets them.

### 2.8 Sparse collision roadmap

Temporal residuals can declare symbolic banded support, but the collision
residuals do not currently provide such declarations and self-collision
optimisation is not a measured structured-solver path. Collision integration
must supply fixed rows and explicit temporal blocks before it can become
eligible; no collision speed-up claim is certified here.

### 2.9 Opt-in Warp whole-pass lane

Warp integration is a whole-pass optimisation, not an interchangeable math
layer. A supported FK, spatial-Jacobian, or RNEA kernel consumes
``ModelStructure`` plus ``ModelValues`` and returns Torch-compatible tensors
at the pass boundary. The Torch raw pass remains the default and correctness
oracle.

Each kernel requires explicit eligibility checks and forward/backward parity.
A custom autograd wrapper owns the analytic adjoint; optional runtime array
types never cross the public boundary. The optional import stays beside the
pass that uses it. Until a pass satisfies those requirements it remains an
opt-in prototype, and ordinary public calls continue down the Torch lane.

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

The linter lives in `tests/contract/test_hot_path_lint.py` (AST walks
the `kinematics/` and `optim/` trees).

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

Benchmarks land advisory and only flip to blocking once they have
collected enough signal:

| Gate | Initial mode | Promotion criterion |
|------|--------------|---------------------|
| Contract bundle (correctness, DAG, hot-path lint, mypy strict, cache invariants, optional imports) | Blocking from day 1 | — |
| CPU bench | Advisory (PR comment) | Two release cycles of stable runner variance < 5% |
| CUDA bench | Nightly only | One cycle of stable self-hosted-runner data |
| `mem_watermark` | Nightly only, advisory | Promoted at v1 release |

Once promoted, `pytest tests/bench/ --benchmark-compare
--benchmark-fail=mean:20%` against the committed baseline
(`tests/bench/baseline_cpu.json`, `tests/bench/baseline_cuda_l40.json`)
is the gate. Hard CUDA gates *before* runner stability is measured
produce flaky CI that gets muted; the ladder is the discipline that
prevents that.

Baseline is bumped only when:

- A performance PR improves the number (new lower bound), or
- A hardware change is announced and all budgets re-measured.

### 4.3 Profiling (opt-in)

The proposed `BR_PROFILE=1` profiler/NVTX hook is not wired yet. Use
`torch.profiler` directly until that M1 roadmap item lands.

### 4.4 Memory watermark

A blocking CUDA memory-watermark benchmark is not present yet. The budgets
in §1.3 remain targets until that benchmark and CI gate land.

## 5 · Compile / JIT lifecycle

### 5.1 Cold start

When a caller explicitly wraps a compatible kernel with `torch.compile`,
the first call records shapes and compiles. Cold-start cost depends on the
PyTorch version, compiler toolchain, device, and input shape and is not
currently gated.

### 5.2 Recompile triggers

For caller-compiled kernels, recompilation can happen when:

- `B` (the batch prefix) changes across queries → compile per shape.
  Mitigated by the shape-specialisation cache; if the user cycles
  through many batch sizes we fall back to dynamic shapes (slower, no
  recompile).
- `dtype` / `device` changes → new cache entry.
- `Model` topology changes → never, because `Model` is frozen.

### 5.3 Cache location

`TORCHINDUCTOR_CACHE_DIR` (default
`~/.cache/torch_inductor/better_robot`). For CI, set to a persistent
path to avoid re-compiling across jobs.

## 6 · Per-module performance ownership

| Module | Owns | Primary technique |
|--------|------|-------------------|
| `lie/` | SE3/SO3 group ops, typed wrappers | Pure-PyTorch; closed-form; compile-friendly |
| `data_model/model_structure.py` | Static topology and device-kernel tables | Validated dual representation |
| `data_model/model_values.py` | Differentiable model tensors | Registered tensor pytree |
| `data_model/execution_batch.py` | Broadcast-to-flat execution ABI | Index maps avoid repeating shared inputs |
| `spatial/` | 6D operators | Dataclass wrappers; no branching |
| `kinematics/forward.py` | FK topo walk and lane boundary | Torch raw pass unrolls on static topology; whole-pass kernels stay local |
| `kinematics/jacobian.py` | Spatial Jacobian | Analytic; automatic compilation is roadmap work |
| `dynamics/*.py` | RNEA / ABA / CRBA | Analytic derivatives; compile-friendly recursion |
| `residuals/*.py` | Residual evaluation | Analytic blocks where implemented; temporal residuals declare exact knot offsets |
| `optim/cost_stack.py` | Legacy concatenation | Fresh `torch.cat` assembly; no persistent flat buffer |
| `optim/blocks/solver_adam.py` | Named-block first-order solve | Tangent objective VJP; no Jacobian assembly; CUDA replay certification deferred to M6 |
| `optim/blocks/solver_lm.py` | Named-block LM/GN | Dense/banded/operator routing with fixed-shape tensor state; CUDA replay certification deferred to M6 |
| `optim/optimizers/*.py` | Legacy flat solver loops | Dense Jacobian path, including legacy Adam/L-BFGS; eager and single-problem |
| `optim/solvers/*.py` | Linear solves | Dense Cholesky/LSTSQ, block-banded Cholesky, and preconditioned normal CG |
| `tasks/parameterization.py` | Numerical trajectory bases | B-spline compression utility; robot-manifold integration requires a separate reviewed design |
| `collision/*.py` | Geometry primitives and roadmap residuals | Stable-shape/sparsity contracts; solver integration is not yet shipped |
| `io/*.py` | One-shot parse | Not hot; readability > speed; `AssetResolver` Protocol |
| `viewer/*.py` | Scene updates | 60 fps budget |

## 7 · Checklist for a "fast" PR

Every PR that claims "perf" answers all of these in the description:

1. What measurement changed? (cite the benchmark file and number)
2. Is the change portable across GPU generations, or RTX-specific?
3. Does it introduce recompilation when batch or dtype cycles?
4. Is the memory watermark unchanged or better?
5. Does `pytest tests/bench/` pass with the new baseline?
6. Is there a new lint rule needed to keep the win?

If any answer is "don't know," the PR is not ready.
