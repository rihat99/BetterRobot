# Redesign plan — BetterRobot and BetterHuman (draft for review, 2026-10-06)

Scope: a plan, no code changes. Commits: BetterRobot `dev` 7e2008f, BetterHuman `dev` 793d716.
Evidence lives in `subagents/` (scans, studies, prototypes) and `artifacts/` (raw logs).
Abbreviations: BR = BetterRobot, BH = BetterHuman, BVR = BetterVideoReconstruction,
FK = forward kinematics, LM = Levenberg-Marquardt, GN = Gauss-Newton.

## 1. Findings (what is true today)

F1. Size. BR `src` is 24,070 lines. tasks + optim + residuals = 8,509 lines (35%).
    BH `src` is 7,345 lines; `viz/studio` is about 3,160 of them (43%).
F2. The optim stack is not used by its main customer. BVR never imports `better_robot.optim`;
    it runs its own 193-line engine (`tools/optim.py`) on `torch.optim` with named leaves, scalar
    terms, lazily shared values and phases with per-phase weights and gradient masks
    (`subagents/scan_bvr_optim.md`). Reasons visible in the code: `optim` exports about 43 names;
    every residual needs a fixed `dim`; a scalar loss must be wrapped as a `sqrt(2f)` row; there
    is no public interface to write an optimizer (in-tree optimizers use 18 private `Problem`
    members); `LevenbergMarquardt` is a 530-line class with 16 constructor parameters and robot
    topology inside; `solve_ik` has no hook for an extra cost (`subagents/scan_br_optim.md`).
F3. The Warp lane is a second implementation that buys little. Only FK and RNEA have one
    (2,206 lines). Its backward pass re-runs the torch implementation
    (`kinematics/_warp_bridge.py:38`), so Warp accelerates the forward pass only. FK math exists
    three times, joint-type dispatch three times, each kernel twice (f32, f64). Stored numbers
    (older commit, 25-joint tree, forward only): Warp 0.54-0.63 ms vs torch.compile 0.71-1.42 ms,
    both flat from batch 1 to 4096 (`subagents/scan_backend.md`).
F4. Speed is bound by the number of small operations, not by compute. Measured today, CPU,
    1 thread: Panda FK 0.65 ms at batch 1 and 1.09 ms at batch 64; RNEA 2.1 ms; ABA 6.2 ms
    (`subagents/bench_baseline.md`). FK forward on a 25-joint tree launches about 220 GPU
    kernels, forward + backward about 800; on a 223-joint tree 452 and 2,297
    (`subagents/proto_level_fk.md`). FK, RNEA, CRBA, ABA and centroidal loop over joints in
    Python; only the Jacobian pass is batched across joints (52 launches for any joint count).
    Batched IK with defaults: Panda, batch 1024, CPU: 4.2 s; 67.5% of elements reached the 1e-6
    tolerance in float32 within 100 iterations (all targets reachable). The final error
    distribution was not measured, so this may be a tolerance artifact rather than failed solves.
F5. Three viewers, about 8,000 lines, no shared code: BR 1,872 (one pose, most interface for
    least function, dead code), BH 3,263 (editing studio, no time), BVR 2,871 (playback, forces,
    cameras, multiple people; no layer abstraction). Missing in all: collision geometry, video
    export, headless render, plots (`subagents/scan_viewers.md`).
F6. Foundation layer debts (`subagents/scan_br_core.md`): `Model` has 57 forwarding properties;
    `with_values` covers 3 of 15 value fields (BVR sets gravity with a nested
    `dataclasses.replace`); `Data` is mutable with 26 optional cache fields and a documented
    stale-cache hole; the `spatial` classes are unused by kinematics and dynamics; `collision`
    is all `NotImplementedError` stubs; joints are represented twice (classes and kind codes),
    a new joint type touches 6 files; the API downstream really uses is about twice the declared one.
F7. BH core idea is sound: bake shape once into differentiable `ModelValues`, then stream poses.
    Debts: stored numbers show the torch path 1.1-2.5x slower than upstream smplx; SMPL-X
    expression is a shape group, so per-frame expression forces a re-bake; BH imports BR
    internals; 3 families registered; no retargeting, markers, priors, limits
    (`subagents/scan_better_human.md`).
F8. Bugs found on the way. `lie/tangents.py:187-204` truncates the SE3 right-Jacobian series after
    the 9th power and claims machine precision for |xi| < 6; the 10th term is about 7e-3 at
    |xi| = 3.5. `PoseResidual.jacobian` is therefore wrong far from the target (measured 5e-3 to
    4e-2 against finite differences), accurate near it. Also: `random_configuration` returns
    float32 on a float64 model and inf for unlimited revolute joints; FK with mismatched dtypes
    crashes inside `index_copy` instead of a validation error. The pinocchio parity tests
    (`tests/test_pinocchio/`, including one for this Jacobian) use `importorskip` and pinocchio is
    not installed in the dev environment, so they skip silently.
F9. Benchmarking is thin: 4 scripts in BR, 1 in BH (CUDA hard-coded), no stored history, no
    gate, no pinocchio in the BR env, Panda cases of `benchmarks/baseline.py` report UNSUPPORTED
    without an optional package. No GPU was quiet during this task, so there are no fresh GPU
    numbers; that is step 0 of the roadmap.

## 2. Decisions proposed

### D1. Backend: one torch implementation with an operation budget; Warp removed

- Every algorithm is written once, in torch. Users pass and receive torch tensors.
- The design rule that replaces "use Warp for speed": each pass has an **operation budget**,
  checked in CI (exact, no timing noise). The budget is a constant for passes that can be written
  as sums over index tables (Jacobians today: 52), and grows with tree depth, not joint count,
  for passes that must go level by level (FK chain, ABA). For a serial arm depth is close to the
  joint count (Panda: depth 11 of 14 joints), so arms gain only from (b) below.
- Memory budget: ancestor-descendant tables cost memory (about 4,000 pairs for 200 joints; one
  6x6 float32 tensor per pair at batch 4096 is about 2.4 GB). Each pass states its peak memory
  at batch 4096 and 200 joints; passes that exceed it stay level-wise.
- How: (a) compose the world chain per tree level (prototype: 223-joint forward + backward
  2,297 → 913 kernels, 52 → 13 ms on a shared GPU, bit-identical results); (b) remove the
  constant part that dominates small robots (about 200 kernels in input preparation, per-kind
  local transforms, 4x4 assembly, matrix to quaternion): keep poses as rotation + translation,
  one batched formula per joint family, no quaternion round trip inside passes; (c) the same
  for RNEA, CRBA, centroidal (sums over ancestor-descendant index tensors), ABA per level.
- Optional accelerator: one wrapper that compiles the same functions (`torch.compile`,
  CUDA graphs). It never contains a second implementation.
- Warp is deleted (2,206 lines, `use_warp` flags). The argument: its backward re-runs torch,
  so for anything that needs gradients Warp should be no faster than torch alone. That was
  never measured. Experiment E1 measures it: Warp vs torch, forward and forward + backward,
  current commit, idle GPU, float32, Panda / 25-joint / 200-joint, batch 1 / 256 / 4096, plus a
  CPU leg and peak memory. Rule: if Warp forward + backward is within 10% of torch, delete Warp
  now. What is then lost is forward-only speed (stored: 0.54-0.63 ms vs 2.3-2.6 ms for today's
  torch on a 25-joint tree, different GPUs) until (b) lands; whether that is acceptable is
  owner question Q1. (b) has no prototype yet: it is the first prototype of P0.
- If torch cannot get close after (b) and the accelerator, the fallback is one hand-written
  kernel with a hand-written backward that replaces a torch pass. It costs as much to maintain
  as a Warp lane; the difference is only that there is still one implementation.
- Also in E1/E2: CPU batch-1 latency target (Panda FK 0.25 ms or less; 0.65 ms today), float32
  accuracy far from the world origin (robot at 50 m), double backward under the accelerator.
- Analytic derivatives of RNEA (Carpentier-Mansard) become their own algorithm later; autograd
  remains for everything else and for gradients with respect to model parameters (human shape).
- Not verified: no measurement yet shows the op-budget rewrite reaches Warp's forward time;
  `torch.compile` gave no gain on the 223-joint forward in the prototype.

### D2. Model layer: smaller, functional, one joint registry

- `Model` = structure (topology, index tensors) + values (differentiable, batched). One
  `model.replace(gravity=..., joint_placements=..., ...)` for every value field.
- Replace mutable `Data` and its cache invalidation by immutable results: `state =
  kinematics.forward(model, q, v=None)`; Jacobian and dynamics functions take `state`. `state`
  holds everything later passes need (world rotations and translations, world joint axes /
  motion subspaces, velocities when `v` is given), so no pass recomputes FK.
- These signatures (`forward -> state`, `frame_jacobian(model, state, frame)`) are frozen in P0
  as a thin shim over today's code, so P1 and P3 are written against them and not ported twice.
- One joint registry keyed by kind, one batched formula per joint family; adding a joint type
  touches one file.
- `geometry` becomes a typed object on the model (visual and collision shapes), not `meta["ir"]`.
- Public API equals used API: `build_tree`, `JointKind`, `Model.expand_q`, FK returning rotation
  matrices; BH and BVR import nothing private; a contract test in BH enforces it.
- `spatial`: keep raw functions used by the passes, delete unused wrapper classes.
  `collision`: replace the stubs by a minimal real one in P2 (sphere and capsule pairs,
  sphere-to-SDF), because robot trajectory optimization is judged with obstacles.

### D3. Optim: three thin layers, terms are plain functions

Layer 0, functions: `jacobian(fn, x, manifold)`, robust kernels, linear solvers
(dense, block-banded), `check_jacobian(term)` against finite differences.
Layer 1, `Problem`: variables are a dict of named tensors with leading batch axes; optional
manifold and box bounds per variable. `P.residual(name, fn, jac=None, weight=, kernel=,
stencil=)` and `P.scalar(name, fn)`; `P.cached(fn)` for values shared by terms, computed at
most once per evaluation. `P.loss(x)` is an ordinary differentiable scalar, so any
`torch.optim` loop works on it with no adapter.
Layer 2, solvers and phases: a solver is `init(problem, x) -> State` and
`step(problem, state) -> State` over batch-first tensors (per-problem damping, per-problem
`done`, no host sync inside `step`). Shipped: LM, GN, batched L-BFGS, an adapter for any
`torch.optim` class. `Phase(weights, solver, iters, free=mask, on_start=hook)` and `run`.
`solve_layer(..., mode="implicit" | "unroll")` for use inside NN training.

- Variables declare their axes: independent batch axes, an optional time axis, or shared across
  time (body shape). Solvers keep batch elements independent and couple along time.
- Cached values are pure functions of the variables and are evaluated *inside* the
  differentiated call. Linearization is one traced evaluation per problem: all autodiff terms are
  concatenated and differentiated together, so a shared FK runs once and its dependence on the
  variables is not lost. (Differentiating each term separately would either recompute FK per
  term or, with a cache filled outside the trace, return zero Jacobians.) Analytic terms have
  the signature `jac(values, cache) -> {variable: block}` and read the same cached state.
  Not yet prototyped with several terms sharing one FK: first prototype of P1.
- Jacobians: per term, a dict of blocks in tangent coordinates (dense `(B, m, nv)` or a
  temporal stencil block). Source order: the term's analytic `jac`, else reverse-mode autodiff
  of `delta -> fn(retract(x, delta))` at 0. Measured: all 7 autodiff variants work through the
  library FK today and agree to 1e-10; reverse mode is 2-5x slower than analytic on GPU and up
  to 11x on CPU at batch 4096 (`subagents/proto_autodiff_jacobian.md`). Therefore built-in
  kinematic terms ship analytic Jacobians; autodiff is the default for user terms.
- Normal equations are accumulated block by block (never a full J): dense for IK-size
  problems, block-banded for trajectories. Float32 policy: column scaling, damping floor,
  optional float64 linear solve (a float32 Cholesky failed on a rank-6 `JᵀJ` in the prototype).
- Mixed problems: GN/LM use residual terms for the Hessian and add the gradient of scalar terms.
  A free variable that no residual term reads is an error under GN/LM (its Hessian block would
  be zero); use a first-order solver for that phase.
- Batched L-BFGS uses a fixed-budget backtracking line search per problem. Strong-Wolfe on one
  coupled problem stays available through `torch.optim.LBFGS` (what BVR uses).
- Honest scope for BVR: its fit is first-order today (Adam, L-BFGS, robust scalar losses). The
  new layer first replaces its private engine with the same behaviour plus manifolds, axis
  roles, logging and hooks. LM on its dynamics stage is an experiment, not a promise, and LM
  over body shape shared across time needs the arrowhead solve that is deferred.
- `solve_layer(mode="implicit")` is only valid at a converged, unconstrained solution. It
  checks stationarity and falls back to `unroll`; variables on an active bound are masked; the
  Gauss-Newton Hessian is an approximation when residuals at the optimum are large.
- Manifolds: second-order solvers step in tangent space and retract; `torch.optim` optimizes
  the ambient tensor and projects after each step by default.
- Constraints v1: box bounds by projection; augmented Lagrangian as a tested recipe with
  multiplier updates and a convergence check (needed for collision and contact in trajopt).
- Deliberately left out: variable objects, graph harvesting, `Node`, general sparse solvers,
  linearization-mode analysis, Riemannian first-order methods, QP/SQP/DDP, implicit
  differentiation for scalar terms, arrowhead (global-variable) Schur solve until prototyped.
- Size: the design study estimates about 1,500 lines; the reviewer estimates 2,500-3,500 once
  all solvers, the banded solve and the float32 policy are in. Target: under half of today's
  4,019 lines in `optim`, with the public surface cut from about 43 names to about 15.
- Acceptance tests: the four use cases as runnable examples; BVR's person fit ported to the new
  API reproduces its current losses; batched Panda IK at 1024 reaches 1e-4 m and 1e-3 rad on every reachable target
  in float32; trajectory optimization of an arm around sphere obstacles.
- `tasks` become open recipes: `ik_problem(model, targets)` returns a `Problem` the user can
  add terms to; `solve_ik` is three lines on top.

### D4. Viewer: one core in BetterRobot, on viser

- Stay on viser (installed 1.0.30 has skinned meshes, batched meshes with per-instance color,
  arrows, frustums, plots, scene serializer); use it directly, drop the backend protocol.
- Core: `Viewer` (server, groups, one atomic push per frame), `Timeline` (play, pause, scrub,
  speed; a single pose is T = 1), `Layer` (`build`, `set_frame(k)`, `set_visible`, own GUI)
  fed by preloaded `(T, ...)` arrays. Built-in layers: rigid meshes, skinned mesh, skeleton,
  arrows, frames, points, camera, ground, plot panel, collision geometry.
- Adapters return core layers: `robot_layers(model, q, tau=...)` in BR (FK once per clip),
  `human_layers(body, ...)` in BH, `clip_layers(ClipData)` in BVR. The core imports none of them.
- Tools (IK gizmo, picking, BH editors) sit on top of the viewer; the viewer no longer imports `tasks`.
- Export: `.viser` scene file (plays in a browser with no Python, through viser's static
  client), MP4 through a connected browser (one render round trip per frame: slow, not headless).
- About 1,600 lines in BR, about 400 in BH, replacing about 8,000 across three viewers
  (BH studio editors stay as a BH app, about 1,200 lines).
- Not measured: websocket throughput for several characters at 30 fps; bone limit of skinned
  meshes for MHR. Both are first-week checks.

### D5. BetterHuman: families as data, streams per frame, public BR API only

- `FamilySpec` (pure data: template, bases, regressor, top-K skin weights, corrective kind) +
  `Body.bake(**identity) -> Identity` + `Body.forward(identity, q, stream={...}, outputs=...)`.
  Per-family behaviour is enums plus arrays, not stored callables. SMPL-H and MANO become a
  table and a short loader.
- Per-frame groups (expression) do not re-bake: precompute their effect on joint positions at
  bake time; needs BR FK to accept placements batched over time (a P2 kinematics change).
- `outputs`: joints, markers (skin only the needed vertices), vertices on request.
- Human terms on the new optim (pose prior, limits, markers, reprojection, ground contact),
  inverse dynamics with a contact-force split, retargeting in `better_human.retarget` built
  from BR public tasks and optim.
- Viewer: BH layers on the BR viewer; studio shrinks to its editors.
- Benchmark suite with upstream smplx as the baseline, forward and backward, JSON output.

### D6. Benchmarking and profiling as a product feature

- One runner per repo: algorithm x model x batch x device x forward/backward, JSON per commit.
- CI gate on operation count per call (exact for a pinned torch version; the version is stored
  with each baseline); wall-time gate only on the bench machine (median above 1.10x baseline
  and above 3 MAD).
- Baselines: pinocchio (CPU, parity and speed), pyroki, cuRobo, mujoco_warp, upstream smplx.
- A small profiling helper: launches per call, top operations, peak memory, compile time.
- Needs one reserved or reliably idle GPU for timing runs.

## 3. Roadmap

P0 Measure and fix. Install pinocchio and Warp in the dev environment so parity tests run;
   benchmark runner + clean baselines; fix F8 bugs; freeze the kinematic `state` signatures
   (D2) as a shim; experiments E1 (Warp vs torch with gradients), E2 (compile feature matrix),
   prototype of D1(b) on a 25-joint tree. Record golden outputs of IK, trajopt and a BVR clip
   before anything is deleted. Exit: baseline JSON committed; E1/E2 verdicts.
P1 Optim. Build the new package beside the old one; port IK, trajopt, BVR fit; delete the old.
   Exit: acceptance tests of D3.
P2 Model and kin/dyn. D2 cleanup, op-budget passes, delete Warp (after the E1 gate), accelerator.
   Exit: op counts within budget (constant or depth-bound); parity with pinocchio; targets of
   the benchmark table; minimal collision in place.
P3 Viewer. Core + robot adapter, then BH layers, then BVR port; delete the three old viewers.
   Exit: the four viewer scenarios run; BVR clip plays with the same content as today.
P4 BetterHuman. FamilySpec, streams, public API only, terms, retargeting, benchmarks.
   Exit: SMPL-X forward at batch 256 at least as fast as upstream smplx (target; mechanism:
   fewer FK operations from P2 and skinning only requested vertices); BH imports no BR internals.
P5 Later. Mesh collision, analytic dynamics derivatives, first-class constraints, optional
   rerun/offline export, more families (SKEL, STAR, SUPR).
Order: P0 first. P1 and P3 depend on P2 only through the `state` signatures frozen in P0, so
they can run in parallel with it. P4 needs P1, P2 and P3.
Migration: each phase ends with a tag in both repos; BVR pins a tag and moves one phase at a
time; old and new optim coexist until the BVR fit is ported; docs and the `CLAUDE.md` guides
are updated in the phase that changes their contract.

## 4. Decisions that belong to the owner

Q1. Delete Warp once E1 confirms it gives nothing with gradients, accepting slower forward-only
    FK/RNEA at small batch until the rewrite lands? (Recommended: yes.)
Q2. Replace mutable `Data` by immutable result objects? (Recommended: yes.)
Q3. Terms read variables by parameter name (short, but fragile with `partial`, `**kwargs` and
    renames), or always list `reads=` explicitly? (Recommended: by name, validated loudly at
    registration.)
Q4. Is batch ≥ 65k (sampling, RL) a real use case? It is the only regime where a hand kernel pays.
Q5. Retargeting inside BetterHuman (recommended) or a third package?
Q6. Which BH studio editors are still used (segment, contact frames, posing)?
Q7. Is headless video (no browser) or a rerun exporter wanted?
Q8. Expression exact (recommended) or approximate; next family: SKEL or STAR/SUPR?
Q9. Phase order after P0: optim first (recommended), or speed first?

## 5. Review record

Adversarial review by a clean-context agent on 2026-10-06 (17 findings, 2 critical). Accepted
and applied above: operation budget restated as constant or depth-bound with a memory budget;
E1 redefined to compare with gradients; cache and autodiff interaction specified; axis roles;
honest scope for BVR; limits of implicit differentiation; scalar-only variables under LM;
minimal collision moved to P2; `state` signatures frozen in P0; CPU, far-from-origin accuracy
and double-backward checks; IK convergence claim softened; line estimate widened; pinocchio
tests found to skip. Kept despite the objection: name-based injection as the recommended
default (owner question Q3).

## 6. Owner decisions (2026-10-06, after reading the pages)

Q1. Remove Warp. Guaranteed differentiability of every algorithm is central; a lane that
    cannot differentiate itself is not wanted. E1 becomes a record of what is lost on
    forward-only speed, not a gate.
Q2. Replace mutable `Data` by an immutable result of `forward_kinematics`, under a new name.
    The owner rejected `kin` / `Kinematics` as confusing. Working name chosen by the agent:
    class `KinematicState`, variable `state` (small, reversible; owner may rename).
Q3. Terms read variables by argument name, validated loudly at registration; `reads=` stays.
Q4. Main batch size is about 4096. Larger batches may be tested but are not a design driver;
    the hand-kernel experiment E5 is dropped from the near-term plan.
Q5. Retargeting is NOT part of BetterHuman. It becomes a separate package later, outside this
    update cycle. The optim must still be able to express it (kept as a use-case example).
Q6. Keep all BetterHuman studio functionality (segment painting, contact frames, posing,
    picking, save/load); make it more convenient, on top of the new viewer.
Q7. Recording without a browser is required: the viewer plan gains an offline renderer that
    reads the same layer arrays.
Q8. Expression: exact streaming. Full support for SMPL-X and FLAME in this cycle, with an
    option to switch SMPL-X expression off when not needed. SKEL, STAR, SUPR later.
Q9. Order: P0, then optim first.
