# Contracts & Validation

> **Status:** normative. Describes the library's **input contracts**,
> **numerical guarantees**, and **error-handling policy**.

Other docs describe *what* the library does. This one describes the
**rules that every public function must honour** and **what the user
must give in return**. Without this file, the boundary between "user
bug" and "library bug" is fuzzy — and code slowly accrues defensive
noise.

## 1 · Input contracts

### 1.1 Tensor shapes

All public tensors are shaped `(B..., feature)` where `B...` is zero
or more leading batch axes and `feature` is the semantic last-axis
(e.g. `nq`, `(njoints, 7)`, etc.). See
[10_BATCHING_AND_BACKENDS.md](../design/10_BATCHING_AND_BACKENDS.md) for the
full shape table.

**What we promise:**
- Every return tensor's leading batch shape **equals** the broadcast
  of all input batch shapes.
- We never drop a leading batch axis (`B=1` stays `(1, …)`).
- We never add a leading batch axis the user didn't provide.

**What we expect:**
- Inputs are contiguous (`.contiguous()` if not — cost is caller's).
- `q` has shape `(B..., nq)`; `v`/`a`/`tau` have `(B..., nv)`.
- Broadcast follows PyTorch rules; ambiguous broadcasts raise.

### 1.2 Device & dtype

- `Model.to(device, dtype)` returns a new Model with **all** tensor
  buffers moved.
- `Data` inherits device/dtype from the input `q`. Calling
  `forward_kinematics(model, q)` with `q.device != model.device`
  raises `DeviceMismatchError`.
- fp16 is **not supported** in kinematics or optim. Cast up before the
  call; cast back after. We reject mixed precision at the boundary
  because analytic Jacobians are numerically sensitive to it.
- fp64 is supported everywhere and tested in `tests/lie/` and
  `tests/kinematics/`.

### 1.3 Quaternion / SE(3) inputs

- Quaternion format is **`[qx, qy, qz, qw]`** (scalar last). Every
  function accepting an SE(3) pose reads it as `[tx, ty, tz, qx, qy,
  qz, qw]`.
- Quaternions are **assumed unit-norm on input**. The library
  normalises once on entry to `forward_kinematics` and any top-level
  solver; internal kernels do **not** re-normalise.
- If you pass a non-unit quaternion with norm outside `[0.9, 1.1]`,
  the entry-point raises `QuaternionNormError`. Norms inside
  `[0.9, 1.1]` are renormalised silently (tolerance for float drift).

### 1.4 Joint limits

- `model.lower_pos_limit <= q <= model.upper_pos_limit` is *not*
  automatically enforced. FK accepts any `q`; the solver honours
  limits only when told to (see `IKCostConfig.limit_weight`).
- Continuous revolute joints (type `R{X,Y,Z}_unbounded`) have `±inf`
  limits — wrapping is the user's responsibility if they care.
- Free-flyer: the 4 quaternion components live on the sphere; the
  library treats them as a free manifold (no rectangular bounds). To
  constrain orientation, use an `OrientationResidual` with a target
  pose and a large weight, not a limit.

### 1.5 Model well-formedness

After `build_model`:
- `parents[0] == -1` (joint 0 is the universe).
- `parents[i] < i` for `i > 0` (topologically sorted).
- `sum(nqs) == nq`, `sum(nvs) == nv`.
- `idx_qs[i] + nqs[i] == idx_qs[i+1]` (contiguous slicing).

Violations raise `ModelInconsistencyError` at build time — never at
query time.

## 2 · Error taxonomy

The library raises a small set of typed exceptions. Each has a single
responsible layer and a documented remediation.

| Exception | Raised in | Meaning | Remediation |
|-----------|-----------|---------|-------------|
| `ModelInconsistencyError` | `io.build_model` | Parsed IR violates topology invariants | Fix the URDF/MJCF or the programmatic builder |
| `IRSchemaVersionError` | `io.build_model` | `IRModel.schema_version` does not match the build's expected version | Re-parse the source asset; regenerate cached `.npz` IR |
| `DeviceMismatchError` | `kinematics`, `dynamics`, `optim` | `q.device != model.device` | Call `model.to(q.device)` or vice versa |
| `DtypeMismatchError` | as above | `q.dtype` incompatible with `model.dtype` | Cast one side; see §1.2 |
| `QuaternionNormError` | `lie`, `kinematics` entry | Input quaternion norm outside `[0.9, 1.1]` | Normalise before passing |
| `ShapeError` | every public entry | Wrong trailing-axis size | Match the published shape |
| `StaleCacheError` | `kinematics`, `dynamics` | `Data._kinematics_level` below required level | Call `forward_kinematics(model, data)` first; or `data.invalidate(NONE)` then re-run |
| `ConvergenceError` | `optim.solve` (optional) | Solver did **not** converge within `max_iter` | Inspect the returned `SolverState`; not always a bug |
| `BackendNotAvailableError` | `backends.set_backend`, parsers, viewer | Optional backend or parser dep not importable | Install the relevant extra — e.g. `pip install better-robot[urdf]`, `[warp]`, `[viewer]` |
| `UnsupportedJointError` | `io.build_model` | URDF/MJCF joint kind without a built-in `JointModel` | Add a custom joint via [15_EXTENSION.md §2](15_EXTENSION.md) |
| `SingularityWarning` *(warning, not error)* | `kinematics`, `optim` | Jacobian condition number > 1e12 | Change initial configuration or relax weights |

**Rule of thumb:**
- Library-internal invariants (e.g. `oMi` computed wrong) should never
  raise — they should assert and fail loudly. Use `assert` for those.
- User-facing input violations raise one of the exceptions above, with
  a message naming the offending tensor and what was expected.

## 3 · Numerical guarantees

### 3.1 Determinism

- `forward_kinematics`, `compute_joint_jacobians`, `rnea`, `aba`,
  `crba` are **deterministic bit-for-bit** on a pinned PyTorch /
  CUDA version for a given (device, dtype, input) triple.
- `solve_ik` is deterministic on a pinned seed and pinned solver
  config, modulo floating-point non-associativity across CUDA
  streams. Tests use `torch.use_deterministic_algorithms(True)` on
  the reference path.

### 3.2 Accuracy

| Routine | Guaranteed accuracy |
|---------|---------------------|
| `se3.exp(log(T))` | `‖Δ‖ < 1e-6` (fp32), `< 1e-12` (fp64) |
| Analytic FK Jacobian vs. `jacrev` | `‖ΔJ‖_F < 1e-4` (fp32), `< 1e-10` (fp64) |
| `solve_ik` pose residual at reported `converged=True` | `‖r‖ < tol` where `tol=1e-4` by default |
| Long-chain FK (30 joints) | 1 ulp of a well-conditioned product of SE(3)s |

Numerical accuracy tests live in `tests/kinematics/` and
`tests/regression/`. See [16_TESTING.md](16_TESTING.md) §4.4.

### 3.3 Singularities

- Near SO(3) singularities (rotation angle ≥ `π - 1e-6`), `log`
  falls back to a Taylor expansion for stable gradients.
- Rank-deficient spatial Jacobians are a **user concern**; the
  analytic inversion in LM handles them via Levenberg damping, not by
  pseudo-inversion. If the user opts into Gauss-Newton and `JᵀJ` is
  singular, the linear solver raises `torch._C._LinAlgError` and the
  outer loop catches it into `ConvergenceError`.

### 3.4 Batched broadcasting

Where two inputs have different leading batch shapes, broadcast
rules are PyTorch's. A broadcast that would create an implicit
copy-larger-than-input (e.g. `(1,7) × (1024, nq)` making a `(1024,
njoints, 7)`) is allowed but logged under
`BR_WARN_BROADCAST=1`. Silent in default runs.

## 4 · Mutability rules

| Object | Mutable? | Notes |
|--------|----------|-------|
| `Model` | **No** | `@dataclass(frozen=True)`. `model.to(device)` returns a new instance. |
| `Data` | Yes | Mutated by kinematics/dynamics. Thread-local — do not share across threads without copying. |
| `IKResult` | No | Dataclass; `.q` is a view into the solver's tensor but treated as read-only. |
| `Trajectory` | Limited | `slice`, `resample` return new instances. Direct tensor access is read-only unless you know what you're doing. |
| `CostStack` | Controlled | `.add(...)`, `.set_weight(...)`, `.set_active(...)` are the *only* mutation points. |
| `LeastSquaresProblem` | No | Re-build if you want different bounds or initial guesses. |

Rule: if it's shared across threads / workers, it's frozen. If it's
per-query, it may be mutable.

## 5 · Autograd rules

- Every public hot-path function participates in autograd: gradients
  flow from `q` → FK output → residual → loss without special
  handling.
- `residual_jacobian(..., strategy=AUTODIFF)` uses `torch.func.jacrev`.
  `strategy=ANALYTIC` uses the residual's `.jacobian()` method.
  `strategy=AUTO` prefers analytic, falls back to autodiff, falls
  back to central finite differences (only as a last resort).
- **Forbidden**: in-place mutation of a tensor currently on the
  autograd tape. The library uses functional-style ops throughout;
  contributions must too.

## 6 · Threading & concurrency

- `Model` is read-only ⇒ freely shareable across threads and
  processes.
- `Data` is mutable ⇒ one `Data` per thread. Use `data.clone()` for
  fork points.
- `CostStack` is mutable; one per optimisation problem. If you're
  parallelising over problems, build a fresh stack per thread.
- The library does **not** call `torch.set_num_threads` internally.
  Inherit whatever the user set.

## 7 · Backwards compatibility policy

### 7.1 SemVer scope

BetterRobot follows SemVer. A **major bump** is required to change:
- The frozen `EXPECTED` public-API set (`better_robot.__all__`,
  currently 26 symbols — see
  [01_ARCHITECTURE.md §Public API contract](../design/01_ARCHITECTURE.md)).
- The SE(3) quaternion layout (`[tx, ty, tz, qx, qy, qz, qw]`).
- The `Model` / `Data` dataclass fields (additive is allowed in
  minor; rename is major unless part of a documented migration
  window).
- The DAG (a new edge in `01_ARCHITECTURE.md`).
- `IRModel.schema_version` increments require an entry in
  `CHANGELOG.md` and may force a major bump if the change is breaking
  to user-cached `.npz` IRs.

The complete release/deprecation discipline lives in
[20_PACKAGING.md](20_PACKAGING.md). This file pins the contract; that
file pins the operational mechanism (extras, version source, release
process, CI gating).

A **minor bump** may:
- Add a new public symbol, joint kind, residual, solver.
- Rename a storage field with one release of deprecation shim (see
  [13_NAMING.md §5](13_NAMING.md)).
- Tighten a numerical tolerance (never loosen without a major bump).

A **patch bump** is:
- Bug fixes; performance improvements that do not change numerical
  output beyond tolerance.

### 7.2 Deprecation mechanism

```python
import warnings
warnings.warn(
    "Data.oMi is deprecated; use Data.joint_pose_world. "
    "Will be removed in v1.1.",
    DeprecationWarning,
    stacklevel=2,
)
```

Deprecation warnings are **on by default** under `pytest` and silent
in production. `BR_STRICT=1` promotes them to errors (used in CI).

### 7.3 Stability tier per symbol

Coarse module tiering, augmented with per-symbol annotations:

| Tier | Meaning | Examples |
|------|---------|----------|
| Stable | SemVer-bound; major bump to remove/rename | `Model`, `Data`, `forward_kinematics`, `solve_ik`, `SE3`, `ModelBuilder`, `LeastSquaresProblem`, `Trajectory` |
| Stable (Protocol) | Extending the protocol (adding methods) is a major bump; using existing methods is stable | `JointModel`, `Residual`, `Optimizer`, `LinearSolver`, `RobustKernel`, `DampingStrategy`, `TrajectoryParameterization`, `AssetResolver`, `Backend` |
| Experimental | May change in minor releases with a deprecation warning | `solve_trajopt`, `retarget`, `compute_centroidal_map`, `BSplineTrajectory`, `MultiStageOptimizer` |

| Module | Stability |
|--------|-----------|
| `lie/`, `spatial/` | Stable from v1. Changes require major bump. |
| `data_model/` | Stable from v1. Field renames follow §7.1 deprecation. |
| `kinematics/`, `dynamics/` (once implemented) | Stable from v1. |
| `residuals/`, `costs/`, `optim/` | Stable from v1 — Protocol signatures are frozen. |
| `tasks/` | Stable from v1 for IK; `solve_trajopt` and `retarget` are **experimental** until phase `T`. `TrajectoryParameterization` Protocol is stable. |
| `collision/` | **Experimental** until phase `C3`. |
| `viewer/` | Experimental. The `RendererBackend` protocol is stable; the concrete modes may iterate freely. |
| `backends/torch_native/` | Stable from v1. |
| `backends/warp/` | **Experimental** until a separate announcement. |
| `io/` (URDF/MJCF) | Stable. Parser edge cases may iterate in patch releases. `IRModel.schema_version` is the controlled change vector. `AssetResolver` Protocol stable. |

Experimental means: no SemVer guarantee, but the signatures will not
wander without a release note.

## 8 · Logging

`better_robot.logger` is a `logging.Logger` named `"better_robot"`.

| Level | When |
|-------|------|
| DEBUG | First-call compile events, registry registrations |
| INFO | One-time Model load summary (nq, nv, njoints) |
| WARNING | `SingularityWarning`, `BroadcastWarning`, deprecated names |
| ERROR | Recoverable failures (convergence, fallback paths engaged) |

No `print` calls in the library. Ever.

## 9 · Assumptions summary (the one-page contract)

If you obey **all** of these, the library obeys its numerical
guarantees:

1. `q.shape == (B..., nq)`; `v/a/tau.shape == (B..., nv)`.
2. Quaternions scalar-last, unit-norm on entry (tolerance 10%).
3. `q.device == model.device`, `q.dtype in {fp32, fp64}`.
4. `Model` is built once; do not mutate.
5. Self-limits (`q ∈ [lo, hi]`) are the user's responsibility unless
   `limit_weight > 0`.
6. `Data` is per-thread.
7. `CostStack`, `LeastSquaresProblem` are per-optimization.

Break any of these and the library will do something, but we make no
promise about what.
