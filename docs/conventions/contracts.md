# Contracts & Validation

> **Status:** normative. Describes the library's input contracts,
> numerical evidence boundaries, and error-handling policy.

Other docs describe *what* the library does. This one describes the
**rules every public function must honour** and **what the user must
give in return**. Contracts pin the responsibility boundary: with one,
"my IK is wrong" splits cleanly into "you passed me a non-unit
quaternion" and "the library has a bug." Without one, every public
function eventually grows ambiguous defensive checks because nobody is
sure who is responsible for which input shape — and the call site
slowly accrues noise that exists to defend against scenarios that
should not happen.

We picked a small set of explicit rules instead. The target policy validates
inputs **at the boundary** with named exceptions and trusts internal callers.
Where the current code does not yet enforce a rule, this document names the
gap instead of claiming the exception already fires. Cache levels are tracked
with `KinematicsLevel`; the SE(3) layout is a stable convention.

The benefit shows up everywhere downstream. The hot-path lint can
forbid `.item()` because contracts already pin shapes. The solver can
skip "is `q` the right shape?" because the entry point already
checked. Documentation can read like prose because the rules live
here, not scattered through every function.

## 1 · Input contracts

### 1.1 Tensor shapes

Supported batched tensor routines use `(B..., feature)`, where `B...` is zero
or more leading batch axes and the trailing axes carry the documented event
(for example ``nq`` or ``(njoints, 7)``). APIs with multiple tensor inputs
state which leading batches participate in broadcasting. See
{doc}`/concepts/batching_and_backends` for the full shape table.

**What we promise:**

- A documented batched return preserves the resolved execution batch of the
  inputs that participate in that operation.
- We never drop a leading batch axis (`B=1` stays `(1, …)`).
- We do not add a leading batch axis unless that surface explicitly documents
  a normalized representation. The current unbatched `solve_trajopt` result
  retains its historical `(1, T, nq)` `Trajectory` shape, and
  `Trajectory.with_batch_dims` adds axes by explicit request.

**What we expect:**

- `q` has shape `(B..., nq)`; `v` / `a` / `tau` have `(B..., nv)`.
- Broadcast follows the boundary's documented right-aligned PyTorch rules;
  incompatible shapes raise. Contiguity is not a universal public
  requirement: direct Torch paths accept supported strided tensors, while an
  opt-in kernel may reject eligibility and fall back.

### 1.2 Device & dtype

- `Model.to(device, dtype)` returns a new `Model` with all tensor
  buffers moved.
- `Data` inherits device / dtype from the input `q`. Calling
  `forward_kinematics(model, q)` with `q.device != model.device`
  raises `DeviceMismatchError`.
- fp32 is primary and fp64 is supported. Supported paths preserve the
  working input dtype; see {doc}`engineering` for accumulation, TF32,
  and tolerance policy.
- fp16 and bf16 are outside the supported contract. Cast up before the
  call. FK and dynamics passes reject them with `DtypeMismatchError`.
- A model/query dtype mismatch raises `DtypeMismatchError` before FK or a
  dynamics pass begins; cast the query or move the model explicitly.

### 1.3 Quaternion / SE(3) inputs

- Quaternion format is **`[qx, qy, qz, qw]`** (scalar last). Every
  function accepting an SE(3) pose reads it as `[tx, ty, tz, qx, qy,
  qz, qw]`.
- Quaternions are assumed unit-norm on input; FK does not normalize them.
  `forward_kinematics(..., check_quaternion_norm=True)` enables an opt-in
  boundary check for free-flyer configurations. The check synchronizes
  accelerator tensors and is therefore disabled on the default hot path.
- With that diagnostic enabled, a norm outside `[0.9, 1.1]` raises
  `QuaternionNormError`. Normalize configurations before calling FK.

### 1.4 Joint limits

- FK accepts configurations outside ``model.lower_pos_limit`` /
  ``model.upper_pos_limit``. The ``solve_ik`` facade always supplies hard
  position bounds to its named-block variable and projects within them;
  ``IKCostConfig.limit_weight`` controls only the additional soft limit
  residual. Direct solver/problem callers own whatever bounds they declare.
- Continuous revolute joints (kind `revolute_unbounded`) have `±inf`
  limits — wrapping is the user's responsibility if they care.
- Free-flyer: the four quaternion components live on the sphere; the
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
| `ModelInconsistencyError` | `io.build_model` | Parsed IR violates topology invariants | Fix the URDF / MJCF or the programmatic builder |
| `DeviceMismatchError` | `kinematics`, `dynamics`, `optim` | `q.device != model.device` | Call `model.to(q.device)` or vice versa |
| `DtypeMismatchError` | as above | `q.dtype` incompatible with `model.dtype` | Cast one side; see §1.2 |
| `QuaternionNormError` | `kinematics` opt-in debug check | Free-flyer quaternion norm outside `[0.9, 1.1]` | Normalise before passing or enable the check only while debugging |
| `ShapeError` | Validated tensor/model boundaries | Wrong trailing event or incompatible batch shape | Match that API's published shape |
| `StaleCacheError` | `kinematics`, `dynamics` | `Data._kinematics_level` below the requested cached result | Run `forward_kinematics(model, q, ...)`, retain the returned `Data`, and advance the required cache before reading it |
| `ConvergenceError` | Defined for caller policy; not raised by current task solvers | A caller chooses to promote a non-converged returned status | Inspect the result/state before optionally raising |
| `UnsupportedJointError` | Defined compatibility type; no current production raise site | Reserved for an unsupported joint boundary | Current parsers/builders normally raise their direct validation error |
| `SingularityWarning` *(warning, not error)* | Defined warning type; not emitted automatically | Reserved for an explicitly diagnosed singularity | Inspect solver/factorization status instead of expecting a warning |

**Rule of thumb:**

- Library-internal invariants should never raise — they should
  `assert` and fail loudly. Use `assert` for those.
- Validated user-facing boundaries use the most specific BetterRobot exception
  already defined for that condition, or a direct ``TypeError``/``ValueError``
  where no typed exception is wired. The table does not claim every public
  function validates every target-policy rule yet.

## 3 · Numerical behavior and evidence

### 3.1 Determinism

- BetterRobot does not enable ``torch.use_deterministic_algorithms`` or change
  process-global backend flags. Pure eager calls have no internal RNG, but
  bitwise reproducibility across devices, CUDA streams, compiler choices, or
  dependency versions is not promised.
- Benchmark and regression evidence records the environment, inputs, and
  tolerances it actually tested. A caller requiring PyTorch deterministic mode
  must configure and validate it for that workload.

### 3.2 Accuracy

There is no single library-wide accuracy budget. Lie round trips, analytic
Jacobians, Pinocchio parity, frozen FK output, and solver convergence each use
routine- and dtype-specific tolerances in their named tests. The committed
fp64 FK oracle is checked at ``atol=rtol=1e-10``; that does not imply a
universal ULP guarantee for arbitrary long chains. Numerical tests live under
``tests/lie/``, ``tests/kinematics/``, ``tests/test_pinocchio/``, and the
relevant residual/solver directories.

### 3.3 Singularities

- SO(3)/SE(3) formulas use safe Taylor branches near zero. At the absolute-pi
  principal-log cut, the rotation remains valid but the chosen tangent's sign
  and derivative are not continuous.
- Dense Cholesky uses ``torch.linalg.cholesky_ex`` rather than relying on a
  thrown ``LinAlgError``. It is strict: an unhealthy factorization returns a
  zero step and per-element failure status to LM, which may increase damping.
  There is no rank-deficient least-squares fallback hidden inside the solver.

### 3.4 Batched broadcasting

Where two inputs have different leading batch shapes, supported boundaries use
documented right-aligned PyTorch broadcasting and raise on incompatible event
or batch shapes. BetterRobot has no ``BR_WARN_BROADCAST`` hook and emits no
automatic broadcast-size warning.

## 4 · Mutability rules

| Object | Mutable? | Notes |
|--------|----------|-------|
| `Model` | Shallowly frozen | `@dataclass(frozen=True)` prevents field reassignment, but contained tensors/dicts remain mutable. Treat them as read-only; deep immutability remains an engineering gap. |
| `Data` | Yes | Mutated by kinematics / dynamics. Thread-local — do not share across threads without copying. |
| `IKResult` | Yes | Plain dataclass containing tensors. Treat it as caller-owned result state; no immutability or view guarantee is promised. |
| `Trajectory` | Yes | Plain dataclass; `slice` and `resample` return new instances, but fields and contained tensors are not frozen. |

Do not infer deep immutability from a dataclass wrapper. Share model state only
under the read-only discipline in {doc}`engineering`; keep mutable data,
solver state, results, and trajectories evaluation-local unless the caller
provides its own synchronization.

## 5 · Autograd rules

- Autograd guarantees are path- and input-specific. Eager Torch Lie/FK and
  named dynamics paths have gradient tests; that does not make every public
  function differentiable. The complete matrix is in {doc}`engineering`.
- The current `solve_ik` detaches its initial iterate and has no
  differentiable-solve guarantee.
- `residual_jacobian(..., strategy=ANALYTIC)` uses the residual's
  `.jacobian()` method. `strategy=AUTO` prefers analytic and falls back to
  unbatched central finite differences at a cost of `2·nv + 1` residual
  evaluations. The named-block `Problem` API separately supports analytic,
  `jacrev`, `jacfwd`, and finite-difference strategies.
- **Forbidden**: in-place mutation of a tensor currently on the
  autograd tape. The library uses functional-style ops throughout;
  contributions must too.

## 6 · Threading & concurrency

- `Model` may be shared only while every contained tensor and dictionary
  is treated as read-only. The current frozen dataclass is not deeply
  immutable.
- `Data` is mutable ⇒ one `Data` per thread. Use `data.clone()` for
  fork points.
- The library does not call `torch.set_num_threads` internally; it
  inherits the user's setting.

CUDA-stream and multiprocessing rules live in {doc}`engineering`.

## 7 · Backwards compatibility policy

### 7.1 SemVer scope

Before 1.0, minor releases may change the public surface. Contract tests pin
a required core and reject duplicate or unresolvable exports without freezing
the current symbol count. Once 1.0 is released, a **major bump** is required
to change:

- A stable public symbol named in this section.
- The SE(3) quaternion layout (`[tx, ty, tz, qx, qy, qz, qw]`).
- The `Model` / `Data` dataclass fields (additive is allowed in
  minor; rename is major unless part of a documented migration
  window).
- The DAG (a new edge in {doc}`/concepts/architecture`).

The complete release / deprecation discipline lives in {doc}`packaging`.
This file pins the contract; that file pins the operational mechanism
(extras, version source, release process, CI gating).

A **minor bump** may:

- Add a new public symbol, joint kind, residual, solver.
- Rename a storage field with one release of deprecation shim (see
  {doc}`naming`).
- Tighten a numerical tolerance (never loosen without a major bump).

A **patch bump** is bug fixes; performance improvements that do not
change numerical output beyond tolerance.

### 7.2 Deprecation mechanism

```python
import warnings
warnings.warn(
    "The 'old_name' API is deprecated; use 'new_name'. "
    "Will be removed in vX.Y.",
    DeprecationWarning,
    stacklevel=2,
)
```

Deprecation warnings follow Python/pytest's standard filter configuration.
BetterRobot does not install a ``BR_STRICT`` warning promotion hook, and the
manual CI workflow does not promote deprecations to errors. A change that adds
a shim should add a focused warning test and removal version.

### 7.3 Stability tier per symbol

| Tier | Meaning | Examples |
|------|---------|----------|
| Stable | SemVer-bound; major bump to remove or rename | `Model`, `Data`, `forward_kinematics`, `solve_ik`, `SE3`, `ModelBuilder`, `Trajectory` |
| Stable (Protocol) | Extending the protocol (adding methods) is a major bump; using existing methods is stable | `JointModel`, `Residual`, `LinearSolver`, `RobustKernel`, `TrajectoryParameterization`, `AssetResolver` |
| Experimental | May change in minor releases with a deprecation warning | `solve_trajopt`, `compute_centroidal_map`, `BSplineTrajectory` |

| Module | Stability |
|--------|-----------|
| `lie/`, `spatial/` | Stable from v1. Changes require major bump. |
| `data_model/` | Stable from v1. Field renames follow §7.1 deprecation. |
| `kinematics/`, `dynamics/` | Stable from v1. |
| `residuals/`, `optim/` | Stable from v1 — Protocol signatures are frozen. |
| `tasks/` | Stable from v1 for IK; `solve_trajopt` is experimental. `TrajectoryParameterization` Protocol is stable. |
| `collision/` | Experimental. |
| `viewer/` | Experimental. The `RendererBackend` protocol refers only to scene rendering and is stable; concrete modes may iterate. |
| `io/` (URDF / MJCF) | Stable. Parser edge cases may iterate in patch releases. `AssetResolver` Protocol stable. |

Experimental means: no SemVer guarantee, but the signatures will not
wander without a release note.

## 8 · Logging

No package-level ``better_robot.logger`` or automatic model-load,
compile-registry, or fallback logging surface ships today.
``SingularityWarning`` is the current dedicated warning class; there is no
``BroadcastWarning``. Library code should avoid unsolicited ``print`` calls.
Any new diagnostics need a scoped standard-library logger or explicit return
field plus tests and documentation before callers rely on them.

## 9 · Assumptions summary (the one-page contract)

These rules define the supported behavior described above:

1. `q.shape == (B..., nq)`; `v / a / tau.shape == (B..., nv)`.
2. Quaternions scalar-last, unit-norm on entry (tolerance 10%).
3. `q.device == model.device`, `q.dtype in {fp32, fp64}`.
4. `Model` is built once; do not mutate.
5. FK does not enforce position limits. ``solve_ik`` always supplies hard
   bounds; ``limit_weight`` only adds/removes its soft limit residual.
6. `Data` is per-thread.
Break any of these and the library will do something, but we make no
promise about what.
