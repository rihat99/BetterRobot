# Changelog

## Unreleased — Legacy-reference cleanup 2026-04-25

The strategic-review fold-in below shifted the doc set into a
post-skeleton stance, but several specs still framed completed work as
future ("the skeleton lands", "tests_v2/", "what gets deleted from the
current codebase"). This pass scrubs that framing without changing the
end-state contract. Affected files:

- **00_VISION** — principle 5 ("delete fearlessly") rewritten in past
  tense (the `_solve_floating_*` twins are gone, not "must go"). G7
  re-cast: the skeleton is in place; forward work is feature
  implementation per `UPDATE_PHASES.md`. Backend principle (#8) clarifies
  that PyPose powers `lie/` *today* and pure-PyTorch is the P10 target.
- **01_ARCHITECTURE** — directory layout now distinguishes shipped
  files from `(planned, P{N})` ones; `__init__.py` line clarifies "25
  today → 26". Optim and tasks subtrees mark `multi_stage.py`,
  `linear_solvers/`, `damping.py`, `parameterization.py`, `assets.py`,
  and the named-helper version of `programmatic.py` as planned phases.
- **02_DATA_MODEL** — the §10 "what gets deleted" section is reframed
  as "already retired (reference)"; the §11 deprecated-name shims
  section corrected to describe the shims that actually exist on
  `Data._DEPRECATED_ALIASES` (oMi / oMf / liMi / nle / Ag).
- **03_LIE_AND_SPATIAL** — directory layout flips the comment on
  `_torch_native_backend.py` (planned default after L-C) and
  `_pypose_backend.py` (current default). §10 retitled with explicit
  L-A → L-D phase tags. §6 "private bridges" updated likewise.
- **04_PARSERS** — §8 "what disappears" reframed as "already retired
  (reference)".
- **05_KINEMATICS** — pseudocode example renamed `_fk_impl` →
  `forward_kinematics_raw` to match `kinematics/forward.py`.
  "Why this replaces the current FK" → "Properties of this FK (vs. the
  original prototype)". §3 "the problem with the current code"
  rewritten as "why one strategy".
- **07_RESIDUALS_COSTS_SOLVERS** — opening rewritten: the
  `CostTerm`/`Problem`/`SOLVERS` trio it replaced is gone, not "the
  current trio".
- **08_TASKS** — §5 "what gets removed" reframed as "already retired
  (reference)".
- **09_COLLISION_GEOMETRY** — §10 "what gets migrated" reframed
  similarly.
- **10_BATCHING_AND_BACKENDS** — §6 GPU-support discussion clarifies
  that PyPose is the live backend today; capability matrix re-orders
  rows to put `pypose` first as the current default and labels
  `torch_native` "planned (P10/L-A)".
- **11_SKELETON_AND_MIGRATION** — **complete rewrite**. Was a
  pre-skeleton phase plan referencing `src/better_robot_v2/`,
  `tests_v2/`, "Phase 5 cutover" — none of which ever happened. Now a
  forensic record of the landing: §1 pre-skeleton tree, §2 what was
  deleted, §3 what survived (with new home), §4 naming-rename, §5 v1
  release acceptance criteria with `[done]` markers on completed
  items, §6 unchanged out-of-scope list, §7 "how to use this document"
  pointing readers at `UPDATE_PHASES.md` for forward work.
- **13_NAMING** — §2 rename-table preface reworded ("left column = the
  Pinocchio cryptic name the prototype used", not "what currently
  exists in src/"). §5 retitled "Migration status" (the rename
  landed); replaces the migration-plan steps with the enforcement
  contract.
- **18_ROADMAP** — phase letters (`Phase B`, `Phase L`, `Phase D`,
  `Phase Q3`) updated to the `P{N}` numbering used in
  `UPDATE_PHASES.md`. `L-A`–`L-E` sub-phase letters retained inside
  P10.
- **PYPOSE_ISSUES** — same re-numbering: "Phase L" → "Phase P10
  (sub-phases L-A through L-E)".
- **UPDATE_PHASES** — Phase L acceptance criterion mentions full test
  suite under both `BR_LIE_BACKEND` settings instead of a stale
  "297-test suite" count. P1 file table corrected: `forward.py`'s topo
  walk lives in `forward_kinematics_raw`, not a non-existent
  `_fk_impl`.
- **20_PACKAGING** — `pypose` dep comment and module-import discipline
  both refer to P10/L-D (the deletion sub-phase) rather than the
  unspecified "Phase L".
- **README (docs/)** — entry 11 retitled "Historical record of the
  skeleton landing"; §"Rule of thumb" rewritten to point forward work
  at `UPDATE_PHASES.md` rather than at 11.
- **claude_plan/README** — proposal 03's landing pointer now reads
  "UPDATE_PHASES P10".

No design contracts changed in this pass — only tense, framing, and
phase-letter consistency. The specs continue to describe the same
end state.

---

## Unreleased — Strategic review fold-in 2026-04-25

The 17 strategic proposals from `docs/claude_plan/` were accepted and
folded into the canonical specs. This is a **doc-only batch** — no code
changes — but the docs now describe a different end state for the v1
release. Implementation phases are sequenced in
[`docs/UPDATE_PHASES.md`](UPDATE_PHASES.md); the proposal bodies are
preserved at [`claude_plan/accepted/`](claude_plan/accepted/).

### New cross-cutting specs

| # | File | Purpose |
|---|------|---------|
| 19 | [STYLE.md](conventions/19_STYLE.md) | Single normative coding-style guide. NumPy-style docstrings, scalar-last quaternions (`[qx, qy, qz, qw]`), no unit suffixes, jaxtyping shape annotations. Supersedes `docs/style/style_by_*.md`. |
| 20 | [PACKAGING.md](conventions/20_PACKAGING.md) | Operational packaging spec. Pyproject extras taxonomy (`[urdf]`, `[mjcf]`, `[viewer]`, `[geometry]`, `[examples]`, `[warp]`, `[docs]`, `[bench]`, `[test]`, `[dev]`, `[all]`); SemVer pre/post 1.0; deprecation mechanism; release process. |

### New top-level guides

| File | Purpose |
|------|---------|
| [UPDATE_PHASES.md](UPDATE_PHASES.md) | Implementation phases (P0 → P13) for the code-change agent — the operational sequencing of all 17 proposals into independently mergeable PRs with acceptance criteria. |

### Updated specs

- **00_VISION** — added G9 (cache invariants enforced) and G10
  (matrix-free trajopt) to the goals table; non-goals updated to call
  out the human-body extension lane (sibling package, phased) and
  forbid `torch.Tensor` subclassing (`SE3`/`SO3` are dataclasses *around*
  tensors); guiding principles refined to mention typed Lie value
  classes and explicit `Backend` objects.
- **01_ARCHITECTURE** — directory layout adds `lie/types.py`,
  `backends/protocol.py`, `backends/torch_native/{lie_ops,kinematics_ops,dynamics_ops}.py`,
  `optim/state.py`, `optim/linear_solvers/`, `optim/damping.py`,
  `optim/optimizers/multi_stage.py`, `tasks/parameterization.py`,
  `io/build_model.py`, `io/assets.py`. Public API contract is now **26
  symbols** (only `SE3` and `ModelBuilder` newly promoted) governed by a
  frozen `EXPECTED` set in `tests/contract/test_public_api.py` and a
  submodule-reachability contract test. New contract tests listed.
- **02_DATA_MODEL** — `_kinematics_level: KinematicsLevel`; new
  `Data.require()`, `Data.invalidate()`, `Data.joint_pose()`,
  `Data.frame_pose()` methods; `__setattr__` invalidates downstream
  caches on `q`/`v`/`a` reassignment. Added §3.1 documenting the
  in-place mutation limitation explicitly. Added §3.2 noting the
  future `RobotState`/`AlgorithmCache` split as out-of-scope for v1.
- **03_LIE_AND_SPATIAL** — added `lie/types.py` (typed `SE3`/`SO3`/`Pose`
  dataclasses with `.tensor` field; not `torch.Tensor` subclasses);
  pure-PyTorch backend (`_torch_native_backend.py`) becomes the default
  in Phase L; PyPose path retained under `BR_LIE_BACKEND=pypose` for
  one release. Corrected the SE3 exp formula sketch (`V = I + b·W +
  c·W²`, `c` Taylor `1/6 − θ²/120`). Added §7.X documenting the
  Force × Motion duality and explaining why `Force.cross_motion`
  deliberately raises `NotImplementedError`. Kept `Symmetric3` reachable
  via `spatial/__init__`.
- **04_PARSERS** — added `IRModel.schema_version: int` and the matching
  `IRSchemaVersionError`. Replaced `ModelBuilder` stringly-typed
  `kind="..."` with named helpers (`add_revolute_z`, `add_prismatic_x`,
  `add_free_flyer_root`, …); `add_joint(kind=)` accepts a `JointModel`
  *instance*, never a string. Added §11 specifying the `AssetResolver`
  Protocol with filesystem / package / composite / cached-download
  implementations.
- **05_KINEMATICS** — `data.require(KinematicsLevel.PLACEMENTS)`
  enforced at every entry point; `ReferenceFrame` enum replaces
  `reference="..."` strings; `JacobianStrategy.FINITE_DIFF` becomes the
  opt-in fallback once Lie autograd is correct.
- **06_DYNAMICS** — added §5.1 specifying the `JointModel` dynamics
  hooks (`joint_bias_acceleration`,
  `joint_motion_subspace_derivative`) with default zeros — required
  before D2 RNEA. D1 close-out runs **in parallel** with D2 (was
  serialised in an earlier draft).
- **07_RESIDUALS_COSTS_SOLVERS** — wired `OptimizerConfig` knobs
  (`linear_solver`, `kernel`, `damping`); added `ResidualSpec` full
  shape (`output_dim`, `tangent_dim`, `structure`, `time_coupling`,
  `affected_knots`, `dynamic_dim`, …); added matrix-free
  `LeastSquaresProblem.gradient(x)` via per-residual
  `apply_jac_transpose`; clarified the three independent concepts
  (active flag vs weight vs robust kernel — `weight=0` ≠ inactive);
  added §9 `MultiStageOptimizer` generalising `LMThenLBFGS`; added §10
  pinning the stable collision residual `dim` contract.
- **08_TASKS** — `Trajectory` accepts both `(T, nq)` and `(*B, T, nq)`
  shapes (no forced `B=1`); added `with_batch_dims(ndim)`. Added §3.1
  introducing the `TrajectoryParameterization` Protocol with
  `KnotTrajectory` (identity) and `BSplineTrajectory` (cuRobo-default).
  Updated tasks/parameterization.py reference.
- **09_COLLISION_GEOMETRY** — `SelfCollisionResidual` declares
  `dim = number_of_candidate_pairs` (stable across iterations);
  inactive pairs contribute zero rows. `ResidualSpec` declares
  `dynamic_dim=True`. Added §9.1 reading
  `model.meta["asset_resolver"]` for collision-mesh URIs.
- **10_BATCHING_AND_BACKENDS** — replaced process-global
  `current()`/`set_backend()` as the architectural core with **explicit
  `Backend` objects** passed via `backend=` kwargs. The `Backend` /
  `LieOps` / `KinematicsOps` / `DynamicsOps` Protocols are now §7.1;
  added §7.2 capability matrix; §7.3 backend-boundary contract test.
  PyPose retire timeline pinned in 03 §10.
- **11_SKELETON_AND_MIGRATION** — phase plan reflects 26-symbol API,
  PyPose retirement (Phase L), `lie/types.py`, named `ModelBuilder`
  helpers, AssetResolver, `JointModel` dynamics hooks. Acceptance
  criteria expanded: optional-imports test, FK regression oracle,
  matrix-free gradient path verified.
- **12_VIEWER** — `URDFMeshMode` reads `model.meta["asset_resolver"]`
  rather than reimplementing URDF path logic.
- **13_NAMING** — added §2.8 enum table (`ReferenceFrame`,
  `KinematicsLevel`, `JacobianStrategy`); added §2.9 listing the
  `_typing.py` shape aliases (`SE3Tensor`, `JointPoseStack`,
  `ConfigTensor`, …); §2.7 lists new entries (`ResidualSpec`,
  `MultiStageOptimizer`, `TrajectoryParameterization`).
- **14_PERFORMANCE** — bench gates land on the advisory-then-blocking
  ladder; matrix-free trajopt memory wins documented; per-module
  ownership table extended with `backends/`, `optim/linear_solvers/`,
  `tasks/parameterization.py`.
- **15_EXTENSION** — added new seams: §12 `TrajectoryParameterization`,
  §13 `AssetResolver`, §14 backend extension, §15 actuators (Muscle
  Protocol). `JointModel` extension in §2 now mentions the new dynamics
  hooks.
- **16_TESTING** — added §4.6 gate-promotion ladder; rewrote §4.5 oracle
  metadata (no SHA — uses `oracle_version + generation_seed + dep
  versions`); added the new contract test files; rewrote §5.2 to
  describe the frozen-`EXPECTED`-set discipline and the
  submodule-reachability test.
- **17_CONTRACTS** — added `IRSchemaVersionError`, `StaleCacheError` to
  the exception taxonomy; `BackendNotAvailableError` extended to cover
  optional parsers/viewer; SemVer §7.1 references the 26-symbol set
  and the `IRModel.schema_version` increment policy; per-symbol
  stability tier table introduced.

### Updated status

- **18_ROADMAP.md** rewritten to reflect the new shape: lie types and
  PyPose retirement (§1), backend abstraction (§2), dynamics hooks
  (§3), trajectory parameterisations (§4), optim wiring gaps (§5), IO
  ergonomics (§9), public-API + typing (§10), user docs (§12), style +
  packaging (§13). Each entry links to a `UPDATE_PHASES.md` phase.
- **PYPOSE_ISSUES.md** absorbs the retirement plan — Phase L-A through
  L-E. Calls out that analytic Jacobians and `apply_jac_transpose` stay
  first-class even after the swap (matrix-free trajopt depends on them).

### Reorganisation

- `docs/claude_plan/` proposals 01–17 moved to
  `docs/claude_plan/accepted/`. `RECONCILIATION.md` and `00_principles.md`
  remain at the `claude_plan/` root as the audit trail.
- `docs/style/style_by_*.md` carry an "ARCHIVED — see 19_STYLE" banner;
  the drafts disagree with the codebase on quaternion ordering and
  tensor library choice.
- `docs/README.md` refreshed to reflect the new layout.

### Rationale

The 2026-04 strategic review surfaced architectural decisions that are
expensive once the project has external users: typed Lie value classes,
backend abstraction as explicit objects, cache-invariant enforcement,
matrix-free trajopt, named-helper `ModelBuilder`, IRModel versioning,
asset-resolution protocol, advisory-then-blocking CI ladder. The
proposals were reviewed against the gpt_plan colleague review (see
`claude_plan/RECONCILIATION.md`) and accepted with several corrections
to formula errors, layering inversions, and over-aggressive promotion
decisions. The canonical specs above now describe the v1 end state;
implementation phases live in `UPDATE_PHASES.md`.

## Unreleased — Docs refinement 2026-04-19

Comprehensive doc-layer refinement ahead of the next implementation
sprint. **No code changes** in this batch — all work is in `docs/`.

### New cross-cutting specs

| # | File | Purpose |
|---|------|---------|
| 13 | [NAMING.md](conventions/13_NAMING.md) | Readable rename of pinocchio cryptic storage names (`oMi` → `joint_pose_world`, `nle` → `bias_forces`, `Ag` → `centroidal_momentum_matrix`, …) with a one-release deprecation shim on `Data`. |
| 14 | [PERFORMANCE.md](conventions/14_PERFORMANCE.md) | Latency / memory budgets, kernel fusion, adaptive dispatch (cuRobo), CUDA graph capture, anti-pattern lint. |
| 15 | [EXTENSION.md](conventions/15_EXTENSION.md) | `Protocol`-shaped seams for residuals, joints, optimisers, kernels, strategies, linear solvers, collision primitives, render modes, parsers, backends. |
| 16 | [TESTING.md](conventions/16_TESTING.md) | Unit / integration / contract / regression / benchmark strategy; coverage budgets per layer. |
| 17 | [CONTRACTS.md](conventions/17_CONTRACTS.md) | Input contracts, error taxonomy, numerical guarantees, autograd rules, SemVer policy. |

### Updated specs

- **00_VISION** — added performance targets section; clarified the name
  stance ("steal algorithms, not jargon"); added reference to
  [13_NAMING.md](conventions/13_NAMING.md).
- **01_ARCHITECTURE** — added an **Extension seams** section pointing
  to [15_EXTENSION.md](conventions/15_EXTENSION.md); replaced the ad-hoc testing
  story with pointers to [16_TESTING.md](conventions/16_TESTING.md) and
  [14_PERFORMANCE.md](conventions/14_PERFORMANCE.md).
- **02_DATA_MODEL** — renamed every `Data` field to its readable form
  (`joint_pose_world`, `frame_pose_world`, `joint_pose_local`,
  `joint_velocity_world`, `mass_matrix`, `bias_forces`,
  `centroidal_momentum_matrix`, `com_position`, `joint_jacobians`, …).
  Added a §11 describing the migration window and the deprecated-name
  shims.
- **03_LIE_AND_SPATIAL** — added pointers to
  [13_NAMING.md](conventions/13_NAMING.md) (keep `Jr` / `hat` / `vee` as Lie
  notation, use verbose function names at the boundary) and
  [17_CONTRACTS.md](conventions/17_CONTRACTS.md) for numerical stability.
- **05_KINEMATICS** — declared as the single source of truth for
  `JacobianStrategy` and the `Residual` protocol; updated every
  `oMi`/`oMf`/`liMi` reference to `joint_pose_world` /
  `frame_pose_world` / `joint_pose_local` in the example code.
- **06_DYNAMICS** — renamed `nle` → `bias_forces`, updated `Data`
  storage references (`data.M` → `data.mass_matrix`, `data.Ag` →
  `data.centroidal_momentum_matrix`, …). Added explicit note that
  RNEA/ABA/CRBA own their **analytic backward kernels**, not autograd.
- **07_RESIDUALS_COSTS_SOLVERS** — added `@runtime_checkable` to the
  `Optimizer`/`LinearSolver`/`RobustKernel`/`DampingStrategy` protocols;
  introduced the shared `SolverState` dataclass (cuRobo pattern); added
  a sparsity-aware assembly note for self-collision residuals.
- **08_TASKS** — added the **two-stage LM-then-LBFGS** solver
  (`optimizer="lm_then_lbfgs"`) for high-DoF humanoids; added
  **B-spline** as the default trajectory parameterisation, with
  implicit smoothness and fewer optimisation variables; fixed `oMf`
  reference.
- **10_BATCHING_AND_BACKENDS** — added the **`WarpBridge` pattern**
  (mjlab) for torch↔warp tensor shims with per-shape caching; added
  **adaptive kernel dispatch** (cuRobo) for collision SDF kernels;
  renamed `oMi` / `oMf` in example code.
- **11_SKELETON_AND_MIGRATION** — corrected the public-API count from
  23 to 25; added a **rename sprint** tied to
  [13_NAMING.md](conventions/13_NAMING.md); rewired the acceptance criteria to
  reference the new contract and benchmark suites.
- **README** — refreshed the doc index into *Core specs* (00–12) and
  *Cross-cutting specs* (13–17) with a reading-order guide.

### Rationale

The previous doc set locked down *structure* but left four dimensions
implicit: **naming**, **performance**, **extensibility**, **testing /
contract discipline**. The new cross-cutting docs make each of these
normative so that the next implementation sprint can regress against
specific bars (perf budgets, contract tests, naming lint) rather than
informal agreement.

No core architectural decisions were reversed — the DAG, the frozen
`Model` / mutable `Data` split, the `JacobianStrategy` flag, the 25-symbol
public API, the `JointFreeFlyer`-as-root floating-base story all stand
exactly as specified in v0.2.

## v0.2.0 — Phase 5 cutover (2026-04-11)

Complete rewrite of `better_robot`. The old `src/better_robot/` tree is
replaced by the pinocchio-inspired v2 architecture. Old tests that bound
to removed internals are dropped; all 227 new tests pass.

### Breaking changes

The public API is entirely new. Nothing from `better_robot` v0.1 is
forward-compatible.

| Old | New |
|-----|-----|
| `br.load_urdf(yourdfpy_obj)` | `br.load(path_or_urdf_obj)` |
| `model.joints.num_actuated_joints` | `model.nq` |
| `model.links.num_links` | `model.nbodies` |
| `model.q_default` | `model.q_neutral` |
| `br.IKConfig(...)` | `IKCostConfig(...) + OptimizerConfig(...)` |
| `br.solve_ik(...) → tensor` | `br.solve_ik(...) → IKResult` |
| `result.q` (was returned directly) | `result.q` (on `IKResult`) |
| `br.compute_jacobian(model, q, link_idx, ...)` | `br.get_frame_jacobian(model, data, frame_id)` |
| `better_robot.algorithms.geometry.RobotCollision` | `better_robot.collision.RobotCollision` |
| `better_robot.math.se3_*` | `better_robot.lie.se3.*` |
| Fixed-base vs floating-base split | Single code path — use `load(..., free_flyer=True)` |

### New features

- **Pinocchio-style Model/Data separation** — `Model` is frozen; `Data` is the per-query workspace.
- **Universal joint system** — revolute (R{X,Y,Z}, unaligned, unbounded), prismatic, spherical, free-flyer, fixed, helical, planar, mimic, composite.
- **Batched FK** — `forward_kinematics(model, q: (B, nq))` runs over any batch shape with no Python loops.
- **Analytic Jacobians** — all built-in residuals have a `.jacobian()` method; `JacobianStrategy.AUTO` picks analytic, falls back to finite differences.
- **Single IK code path** — fixed and floating-base IK through the same `solve_ik` facade; the difference is the model, not the solver.
- **URDF + MJCF parsers** — `br.load(path)` dispatches by suffix; `free_flyer=True` adds a free-flyer root.
- **25-symbol public API** — see `better_robot.__all__`.

### Architecture

```
src/better_robot/
  lie/          SE3/SO3/tangents/_pypose_backend
  spatial/      Motion, Force, Inertia
  data_model/   Model, Data, Frame, Body, Joint, joint_models/
  kinematics/   forward_kinematics, compute_joint_jacobians, get_frame_jacobian
  dynamics/     rnea, aba, crba stubs (raise NotImplementedError)
  residuals/    PoseResidual, PositionResidual, OrientationResidual, limits, regularization
  costs/        CostStack
  optim/        LeastSquaresProblem, LevenbergMarquardt, GaussNewton, Adam, LBFGS
  tasks/        solve_ik, IKCostConfig, OptimizerConfig, IKResult
  collision/    Sphere, Capsule, Box, HalfSpace, RobotCollision
  io/           load, build_model, IRModel, parsers/
  viewer/       Visualizer (stub — pending port)
  utils/        batching, logging
```

### Known limitations (Phase 6+)

- `viewer.Visualizer` raises `NotImplementedError` — viser port pending.
- `rnea`, `aba`, `crba`, `center_of_mass`, `compute_centroidal_map` raise
  `NotImplementedError` — dynamics bodies land in Phase 6.
- No trajectory optimization or retargeting — stubs only.
