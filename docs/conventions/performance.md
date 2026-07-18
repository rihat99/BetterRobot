# Performance

> **Status:** normative goals + informative guidance.

A robotics library that is "correct but slow" is a teaching tool, not a
production dependency. Inverse kinematics for an interactive viewer
needs to close the loop in milliseconds; trajectory optimisation for a
humanoid needs to fit a 200-knot horizon in a single GPU's memory; the
benchmarks that prove either one of those keeps shipping have to remain
finite and reproducible. Performance is therefore not a nice-to-have add-on.
The targets below guide review, but the current manual-only workflow does not
enforce them on every pull request.

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
third is a **gate-promotion ladder** — benchmarks land advisory, collect
signal, and may flip to blocking only after their variance is low enough. No
performance gate is currently promoted.

When another doc seems to imply a different perf rule, this one wins.
The targets here are the contract; the techniques here are how the
contract is met; the lint rules are the discipline that keeps it met.

## 1 · Performance budgets (targets, not promises)

### 1.1 Hot-path budgets (CUDA, fp32, Panda-class arm, single batch)

| Operation | Target (RTX 4090 / L40) | Notes |
|-----------|-------------------------|-------|
| `forward_kinematics(model, q)` | **≤ 150 µs** per call, `B=1` | Locked Panda model: `nq=nv=8`, `njoints=14` |
| `forward_kinematics(model, q)` | **≤ 250 µs** per call, `B=1024` | Batched; cost is mostly launch overhead |
| `compute_joint_jacobians(model, data)` | **≤ 300 µs** per call, `B=1` | Analytic, world frame |
| `get_frame_jacobian(model, data, fid)` | **≤ 50 µs** per call | Cache hit — pure gather + rotate |
| `solve_ik(model, targets, max_iter=30)` | **≤ 8 ms** per call, `B=1` | 30 LM iterations, pose cost only |

Humanoid (locked G1, `nq=36`, `nv=35`, floating-base):

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
| `Data` for `B=1024, njoints=14` | ≤ 5 MiB |
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
`solve_trajopt` preserve the same leading axes with per-element state. See
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
- Everything else in the legacy residual stack: unbatched central finite
  differences as the current fallback. `JacobianStrategy.AUTO` prefers
  analytic evaluation. Named-block `Problem` evaluation separately supports
  analytic, `jacrev`, `jacfwd`, and finite-difference strategies.

**Rationale:** analytic derivatives avoid graph construction or repeated
residual evaluations on the paths where their implementation and maintenance
cost is justified. Any numerical speed ratio still requires the committed
benchmark definition and host evidence.

### 2.4 Kernel fusion (torch.compile boundary)

No automatic `@torch.compile` decorator is installed today.
`forward_kinematics_raw` is tested as compatible with an explicit caller-side
`torch.compile(..., fullgraph=True)` wrapper. The next intended compilation
boundaries are:

1. `forward_kinematics.inner(q, joint_placements, ...)` — the topo walk.
2. `compute_joint_jacobians.inner(joint_pose_world_stack, motion_subspaces, ...)`.
3. Named-block `Problem` residual and Jacobian evaluation.

The Jacobian and `Problem` boundaries remain roadmap work. The outer
Python — `Model` construction, `Data` allocation, and solver iteration —
stays eager.

### 2.5 Adaptive kernel dispatch (design target)

Future routines whose cost depends on a discrete size parameter — number of
collision spheres or trajectory horizon — may select a specialised kernel at
first call. The following table is a proposed policy, not shipping dispatch;
collision residuals are currently stubs and no horizon-specialised CUDA kernel
is installed.

| Regime | Kernel |
|--------|--------|
| `n_spheres ≤ 100` | single-fused-kernel SDF (one launch) |
| `n_spheres > 100` | dual-kernel (per-body transform + per-pair distance) |
| `horizon ≤ 32` | rolled Python loop inside one compile block |
| `horizon > 32` | scan-style parallel prefix kernel |

If this policy is implemented, dispatch and cache lifetime must be keyed on
the static model/problem structure and measured before it becomes a default.

### 2.6 CUDA graph capture for hot solver loops

No captured solver driver ships. Named-block LM `update` is fixed-shape and
sync-free for eligible problems, but the public `run` loop is eager and no
end-to-end IK graph benchmark is committed. A future capture path must own
fixed storage, warmup, invalidation, and a replay lifecycle that covers the
required forward and backward work together.

### 2.7 Current allocation and matrix-free limits

- Named-block `Adam` differentiates `Problem.objective` through tangent
  retractions and does not assemble a Jacobian. This is the shipped
  matrix-free first-order path.
- Named-block LM/GN retain dense assembly as the correctness route. A problem
  with one declared temporal block can instead assemble block-banded normal
  storage or use the explicit `NormalOperator`/`NormalCG` route.
- Knot-based `solve_trajopt` uses route-aware named-block LM and reports the
  chosen dense or banded path. Undeclared residuals fall back to dense;
  structured routing is never inferred from numerical zeros. Manifold-safe
  spline integration is still deferred.

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
A custom autograd wrapper owns the declared adjoint strategy; that may be a
hand-written kernel or, as in the current FK lane, a Torch-oracle recomputation.
Optional runtime array types never cross the public boundary. The optional
import stays beside the pass that uses it. The fused FK pass satisfies the current CUDA correctness
matrix and remains opt-in because its Torch-recompute backward cost was not
measured and no owner default-on decision was made. Future Warp passes remain
prototypes until they independently satisfy the same requirements. Ordinary
public calls continue down the Torch lane.

## 3 · Performance anti-patterns (forbidden)

The table below lists the checks actually implemented by
`tests/contract/test_hot_path_lint.py`. They run whenever the contract suite is
run, including the manual full workflow job. There is no automatic
pull-request trigger today.

| Checked pattern | Bad | Required response |
|---|---|---|
| `.item()` / `.cpu()` | `loss.item()` inside a watched hot path | Keep the decision tensorized, or move a necessary host decision to a documented eager boundary with `# bench-ok: <reason>`. |
| Proven tensor conversion through `float` / `bool` (and `int` in named-block solver files) | `bool(done_tensor)` | Keep solver state in tensors; reasoned eager/static-boundary exemptions are explicit. |
| Per-call `.new_tensor(...)` | `q.new_tensor([0, 1])` | Reuse an existing tensor or construct/hoist static data outside the hot call. |
| Torch allocation inside a Python loop | `for ...: torch.zeros(...)` | Hoist or preallocate; named-block initialization has its narrow documented exemption. |
| Rank branch through `.dim()` | `if x.dim() == 2: ...` | Follow the leading-batch convention instead of maintaining rank-specific paths. |

The linter watches ``kinematics/``, ``dynamics/``, ``residuals/``, ``lie/``,
and the two named-block solver files listed in the test. It does not claim a
general all-``optim/`` walk or rules for arbitrary tensor conditionals,
`.to()`, or `torch.cat`.

## 4 · Measurement — how we know

### 4.1 Microbenchmarks (`tests/bench/`)

The repository carries advisory pytest microbenchmarks plus standalone,
schema-backed milestone harnesses. A representative intended shape is:

```python
# tests/bench/bench_forward_kinematics.py
def test_panda_fk_cpu_b1(benchmark):      ...  # ≤ 1 ms
def test_panda_fk_cuda_b1(benchmark):     ...  # ≤ 150 µs
def test_panda_fk_cuda_b1024(benchmark):  ...  # ≤ 250 µs
```

The current pytest files do not encode or assert every target from §1. The M6
measurement contract, real RTX 6000 Ada environment header, canonical batch
matrix, and artifact status live in `tests/bench/definitions.md`.

### 4.2 Regression guard — advisory-then-blocking ladder

Benchmarks land advisory and only flip to blocking once they have
collected enough signal:

| Gate | Initial mode | Promotion criterion |
|------|--------------|---------------------|
| Contract bundle (correctness, DAG, hot-path lint, cache invariants, optional imports) | Required local/manual verification | Automatic triggering remains an owner decision |
| CPU bench | Manual advisory; placeholder comparison baseline | Two release cycles of stable runner variance < 5% |
| CUDA bench | Manual host-context evidence | One cycle of stable self-hosted-runner data |
| `mem_watermark` | Test definition present; no scheduled gate | Promoted at v1 release |

No performance row is currently promoted to a blocking comparison. The legacy
`tests/bench/baseline_cpu.json` is explicitly a placeholder; measured
hardware-named artifacts live under `tests/bench/baselines/`. Hard CUDA gates
before runner stability is measured produce flaky CI that gets muted; the
ladder is the discipline that prevents that. The workflow remains
manual-only by owner request.

A baseline replacement is an explicit reviewed change with the complete
measurement environment and raw evidence. The placeholder is not silently
updated by automation.

### 4.3 Profiling (opt-in)

The proposed `BR_PROFILE=1` profiler/NVTX hook is not wired. Use
`torch.profiler` directly.

### 4.4 Memory watermark

`tests/bench/test_mem_watermark.py` provides an explicit measurement, but no
blocking or scheduled CUDA memory gate exists. The budgets in §1.3 remain
targets until reproducible evidence and a promoted gate land.

## 5 · Compile / JIT lifecycle

### 5.1 Cold start

When a caller explicitly wraps a compatible kernel with `torch.compile`,
the first call records shapes and compiles. Cold-start cost depends on the
PyTorch version, compiler toolchain, device, and input shape and is not
currently gated.

### 5.2 Recompile triggers

For caller-compiled kernels, PyTorch may specialize or recompile when shape,
dtype, device, tensor layout, static Python topology, or compile options
change. BetterRobot does not add a package-level shape cache or automatic
dynamic-shape fallback. A frozen `Model` prevents accidental mutation; using a
different topology can still produce a different compiled graph.

### 5.3 Cache location

Set `TORCHINDUCTOR_CACHE_DIR` when a run needs an explicit cache location.
BetterRobot does not install a project-specific default. Canonical cold-start
measurements use a fresh per-case directory; a manually configured persistent
cache can avoid recompiling in non-cold developer or workflow runs.

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
| `dynamics/*.py` | RNEA / ABA / CRBA | Differentiable recursion; `compute_rnea_derivatives`, `compute_aba_derivatives`, and `compute_crba_derivatives` are autograd-derived |
| `residuals/*.py` | Residual evaluation | Analytic blocks where implemented; temporal residuals declare exact knot offsets |
| `optim/blocks/solver_adam.py` | Named-block first-order solve | Tangent objective VJP; no Jacobian assembly |
| `optim/blocks/solver_lm.py` | Named-block LM/GN | Dense/banded/operator routing; fixed-shape update and eager public `run` |
| `optim/solvers/*.py` | Linear solves | Dense Cholesky/LSTSQ, block-banded Cholesky, and preconditioned normal CG |
| `tasks/parameterization.py` | Numerical trajectory bases | B-spline compression utility; robot-manifold integration requires a separate reviewed design |
| `collision/*.py` | Reserved geometry, distance, decomposition, and residual surfaces | Primitive containers only; computation and performance work are not shipped |
| `io/*.py` | One-shot parse | Not hot; readability > speed; `AssetResolver` Protocol |
| `viewer/*.py` | Scene updates | 60 fps budget |

## 7 · Checklist for a "fast" PR

Every PR that claims "perf" answers all of these in the description:

1. What measurement changed? (cite the benchmark file and number)
2. Is the change portable across GPU generations, or RTX-specific?
3. Does it introduce recompilation when batch or dtype cycles?
4. Is the memory watermark unchanged or better?
5. Does the relevant benchmark definition pass, and does its artifact record
   the complete environment? (No blocking comparison baseline exists today.)
6. Is there a new lint rule needed to keep the win?

If any answer is "don't know," the PR is not ready.
