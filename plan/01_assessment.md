# 01 — Current-State Assessment

This document describes BetterRobot as it is today: what is broken, what
is structurally wrong, what is genuinely good, and what we are not sure
about yet.

Where the evidence comes from: five audit reports in `research/` (each
with file:line evidence and small benchmarks), cross-checked by hand.
Every claim carries a tag:

- **[verified]** — reproduced directly in this session; you can re-run it.
- **[audit]** — comes from an audit report with cited evidence, but was
  not independently re-run.
- **[suspect]** — needs verification before anyone acts on it.

One important limitation: all measurements are CPU-only (Xeon 8570). CUDA
does not work on this machine — the driver is too old for the installed
torch build. So every claim about GPU synchronization below comes from
reading the code, not from measuring it.

Baseline facts: about 16.1k lines of code in `src/` and 8.7k in `tests/`;
897 tests pass in ~52 seconds; there is no CI (no `.github/` exists).

---

## 1. Critical defects (silent wrong results or crashes in normal use)

### 1.1 NaN gradients at the Lie singularity **[verified]**

**What happens.** `so3_exp`, `se3_exp`, `so3_log`, `se3_log` return
correct *values* at zero rotation (θ = 0), but their *gradients* are NaN
exactly there. Reproduced directly:
`so3.exp(zeros(3, requires_grad=True))` gives gradient `[nan, nan, nan]`.

**Why it happens.** The code computes
`theta = theta2.clamp(min=0).sqrt()` (the pre-M1
`lie/_torch_native_backend.py:148`, now `lie/_impl.py`),
and the square root has an infinite slope at 0. Terms like
`qw = torch.cos(half)` sit *outside* the `torch.where` that switches to a
Taylor approximation near zero. In the backward pass this produces
`0 · inf = NaN`, and the NaN leaks through `torch.where`'s gradient.

**How bad it is.** The blast radius is precise (the codex re-check
narrowed it — it is not "everything breaks"):

- Any autograd path that goes *through* exp/log at the singularity turns
  to NaN: a consumer running `torch.optim` over tangent increments, the
  planned tangent-space `f(x ⊕ δ)` gradients, autodiff residual Jacobians,
  and anything second-order. On an SMPL-like body, the core audit measured
  96 of 99 gradient components NaN when backpropagating the rest residual
  at the rest pose.
- BR's *current* Adam/LBFGS wrappers dodge the bug by accident: they build
  the residual Jacobian explicitly and apply the retraction outside of
  differentiation. So today's shipped optimizers do not necessarily blow
  up on step 1 — but every design this plan moves toward does.
- This bug is the *stated reason* BetterVideoReconstruction abandoned the
  exp-retraction pattern and hand-rolled its own quaternion math
  (`tools/human_optim/qmath.py:3-8`). A direct adoption killer.
- `lie/CLAUDE.md` claims the functions are "smooth and differentiable
  across the singularity". That is false at 0. The fp64 gradient checks
  test *near* zero, never *at* zero.

**The fix** is standard and small: the jaxlie "safe where" idiom.
Substitute a harmless dummy value under the sqrt/division *before* it is
computed, then select the correct branch afterwards. High value for little
work. The same pattern is needed in the right-Jacobian coefficients of
`lie/tangents.py` and in `JointSpherical.difference`. Note (2026-07-17):
`lie/tangents.py:88,130` carries its *own* dtype-independent cutoff
(`theta2 > 1e-10`), separate from `_TAYLOR_THETA2 = 1e-8` — the
dtype-dependent-cutoff fix must cover both constants.

### 1.2 Batched solving does not exist **[verified]**

"Batched by default" is the library's headline claim, and
`docs/concepts/tasks.md` shows `solve_ik` being called with a `(128, nq)`
batch. In reality, batched `solve_ik` crashes at `optim/state.py:95`: the
expression `r0 @ r0` assumes the residual is a 1-D vector. Reproduced with
a `(4, nq)` initial q.

The problem goes deeper than one line. All four optimizers use scalar
Python logic for damping, cost comparison, and step acceptance. There is
no per-batch-element convergence or damping anywhere. And no batched-IK
test exists — which is how a false headline survived 897 passing tests.

### 1.3 Mimic joints are silently ignored **[verified]**

A mimic joint in URDF is a joint whose motion is defined as a linear
function of another joint — gripper fingers are the classic case.
`data_model/CLAUDE.md` claims BR handles them via gather tensors with
nq=0.

In reality, a URDF mimic joint gets its own independent degree of freedom.
The machinery is half-wired (re-verified 2026-07-17): the URDF parser
reads mimic tags (`io/parsers/urdf.py:246-257`), `build_model` resolves
them into `mimic_source/multiplier/offset` arrays (`build_model.py:486-504`),
and a zero-DOF `JointMimic` placeholder class exists
(`joint_models/mimic.py`) — but no kinematics or dynamics code reads any
of it. A Panda loads with nq=9 and two independently movable fingers.
Any gripper or linkage URDF the user believes is constrained silently
produces wrong kinematics. The rule must be: enforce the constraint or
reject the file — never stay silent. One consequence for the M0 reject:
the Panda itself carries a finger mimic (joint 13 → 12, multiplier 1.0),
so reject-all would refuse the flagship robot the parity suite loads —
the rejection scope is an owner decision (see the roadmap M0 row).

Two calibrations from the codex review: (a) pinocchio's *default*
`buildModelFromUrdf` also yields nq=9 on the Panda, so "matches pinocchio"
is not a valid acceptance test for mimic enforcement — the test needs an
explicitly constrained reference, or an assertion on the reduced
coordinate map itself. (b) Real enforcement is not a small FK change:
Jacobians, limits, torque accumulation, RNEA/ABA/CRBA, and all nq/nv
indexing need the same reduced-coordinate treatment. So "reject at build
time" is the only honest quick fix, and true enforcement belongs to the
Model structure redesign.

### 1.4 Bounded LM stalls at the bounds, with no honest terminal status **[verified; mechanism corrected by codex review]**

With box bounds active — which is the `solve_ik` default — the LM solver
degrades badly. Three independent probes agree on the outcome: without
bounds the solver converges in ~25 iterations to ~1e-14 cost; with bounds
it runs to the iteration limit with real error left. (Audit probe: 3.4e-2
cost / 0.31 m position error on one target. My probe: 1.5 mm on another.
Codex probe: 0.259 m with three joints pinned at their bounds on an
unreachable target.) The size of the error depends on the target; the
failure mode does not.

The original explanation ("the step is clamped after acceptance, which
wrecks the acceptance test") was wrong. The codex review corrected it
against `optim/optimizers/levenberg_marquardt.py:90-117`. What actually
happens: the trial point is projected onto the bounds **before** the trial
residual is evaluated; acceptance is a bare `cost_new < cost` comparison;
the gain ratio is computed but the default adaptive damping ignores it;
and there is no notion of an active set, projected gradient, or KKT
conditions anywhere. (KKT conditions are the standard mathematical test
for "this point is as good as the constraints allow".) So the projected
step direction is repeatedly poor at active bounds, progress stalls, and
the run exits as `maxiter`. Not a false `converged` — but there is also no
distinct "stalled at bounds / KKT satisfied" status: `stalled` exists in
the enum and is never set by LM or GN (correction 2026-07-17: LBFGS *does*
set it, at `lbfgs.py:144` — the gap is LM/GN-specific).

The remedy stands: a real bounded least-squares algorithm, not a status
patch (see 03 §4).

### 1.5 `JacobianStrategy.AUTODIFF` / `FUNCTIONAL` silently run finite differences **[verified]**

`residual_jacobian` checks only for `ANALYTIC` and `AUTO`, then
unconditionally falls back to finite differences (FD — approximating each
derivative by nudging an input and re-evaluating the function).
`torch.func.jacrev` and `jacfwd` appear nowhere in the module. In a
library whose identity is "PyTorch autograd throughout", the autodiff
Jacobian path is unreachable, and the enum lies. The FD fallback is also
unbatched (it crashes on batched states, per the quality audit) and costs
`2·nv + 1` full FK passes per Jacobian — codex counted 19 residual
evaluations at nv=9.

Cost calibration (codex re-measured on a controlled Panda pose residual):
analytic ~1.6 ms; a **real** `torch.func.jacrev` ~14 ms; central FD
~62 ms. So analytic is ~39× faster than FD, but only ~9× faster than
actual reverse-mode autodiff. The audit's famous "42×" number compared
analytic against FD, not against autodiff — because real autodiff does not
exist in the code. An attempt to run `jacfwd` failed on a float/double
mismatch inside the closure, so the code is not `jacfwd`-clean either.

### 1.6 collision/ — every query path is unimplemented **[audit, wording corrected]**

Every collision *query and optimization* path raises
`NotImplementedError`: closest points, SDF distance, pair distance, the
robot-collision methods, and the collision residuals. The geometry
dataclasses and the pair registry do construct without raising, so the
audit's "100 % stubs" was slightly too strong — the usable schema pieces
are worth keeping if the port ever happens. Meanwhile,
`docs/reference/roadmap.md` states the rule "if a symbol is not on this
page, it is implemented and tested" — and collision is not on the page.
`solve_ik` even accepts `collision_margin` / `collision_weight` config
options that are never read.

---

## 2. Structural problems (architecture-level findings)

### 2.1 The backends layer — deleted in M1

**Resolution (2026-07-17).** M1 removed the registry and package, renamed the
direct Lie implementation to `lie/_impl.py`, and introduced
`ModelStructure` / `ModelValues` as the whole-pass seam. The bullets below
record the pre-M1 evidence that motivated the inversion; they are historical,
not a description of the current call graph.

The owner's instinct ("this layer is wrong") is correct, and the evidence
is stronger than "ugly":

- **Too many hops per operation.** One `se3.compose` call travels: facade
  → `default_backend()` registry → `TorchNativeLieOps.se3_compose` → a
  lazy re-import of `lie/_torch_native_backend` → the actual math. Four
  Python stack frames plus a registry lookup, per op. One performance
  correction from the codex review: the core audit's "~19 µs / 27 % per
  op" figure is refuted — the quality audit and codex's repeated
  measurements both found ~1–2 µs (~2–3 %) end-to-end on FK. The original
  synthesis had picked the larger of two conflicting numbers. So the case
  for deletion is **structural, not performance**.
- **The dependencies point the wrong way.**
  `backends/torch_native/kinematics_ops.py` and `dynamics_ops.py` lazily
  import from `kinematics/` and `dynamics/` — the "bottom" layer is
  implemented by the layers above it. The lazy imports exist only to dodge
  the layer-contract test. This is not a backend; it is a re-export table.
- **Nothing consumes it.** `DynamicsOps` has zero call sites.
  `KinematicsOps` has two, both bouncing back into the same package. A
  third of the Lie facade (`identity`, `from_axis_angle`, `apply_base`, …)
  bypasses dispatch entirely — so a second backend would silently get
  torch-native behavior for part of the surface. And
  `forward_kinematics_raw` itself calls default-backend `se3.compose`
  internally, so passing `backend=` never fully routes anyway.
- **Wrong granularity for the Warp future.** Warp pays off when it fuses
  whole algorithm passes (the FK sweep, RNEA, residual+Jacobian) — not
  per-op quaternion multiplies. See `references/design/PyposeWarp.md`:
  putting 24 Lie ops into Warp took 13.3k lines of code plus 11.9k lines
  of tests (about 12× the size of BR's entire pure-torch `lie/`), with
  hand-written backward kernels and zero committed benchmark numbers. The
  whole-pass seam BR needed was *located at* the `*_raw` functions. M1
  re-signed FK and RNEA as pure functions over `ModelStructure`,
  `ModelValues`, and query tensors. The Jacobian raw helper still consumes
  `Model` / `Data` and is not yet part of that migrated seam. One
  correction from the warp-platform audit: mujoco_warp and newton run pure
  warp with no torch bridge in their cores; the
  autograd.Function/custom-op boundary pattern comes from warp's
  documented interop guide and from cuRobo. Either way: no Protocol
  registry anywhere.

Verdict (implemented in M1): delete `backends/` (~530 lines), have
`lie/se3.py` / `so3.py` call `_impl.py` directly, and use the migrated raw
passes as the seam. A second implementation is selected only by a plain
`if` at a whole-pass integration point.

### 2.2 Single-flat-variable optimization — the consumer blocker

`LeastSquaresProblem` holds one flat `x0`, one `retract`, one pair of
bounds. `ResidualState` mandates `(model, data, variables)` — and every
built-in residual simply reads `variables` as `q`, by fiat. The
consequences, verified across both consumer repos (consumer-gaps audit):

- Every real fit is multi-block: `q` + forces;
  `transl` + `pose` + `betas` + `scale`; camera `dw` + `t` + `log_s`. This
  is why **all 24 hand-rolled `torch.optim` call sites across 10 files**
  exist — BR simply cannot express those problems.
- BVR wrote a complete 188-line optimization engine (`tools/optim.py`:
  Problem/Phase/run_phases with lazy shared state, per-phase weight
  columns, per-DOF gradient masks). That file is, in effect, the
  requirements spec for BR's redesign.
- BR's own Adam/LBFGS wrappers build the **full dense Jacobian on every
  iteration** (`optim/optimizers/adam.py:79`, `lbfgs.py:81`) —
  contradicting their own docstrings, and making them unusable at
  trajectory scale (T×211 variables).

### 2.3 Model is single-instance and non-parametric — the better_human blocker

Good news first **[audit, experimentally backed]**: autograd already flows
from FK/RNEA outputs back into `joint_placements` and `body_inertias` when
they are swapped in via `dataclasses.replace`. So differentiable body
shape is an *indexing* problem, not an algorithm rewrite. The actual
blockers are narrow:

- FK and dynamics index model tensors on dim 0 (`joint_placements[j]`,
  `body_inertias[i]`), so per-batch `(B, njoints, 7)` placements crash.
  Reproduced in the better_human audit at `forward.py:112`, `rnea.py:167`,
  `crba.py:60`, `aba.py:121`, `centroidal.py:41,152`. Codex calibration:
  switching to `[..., j, :]` indexing is *necessary but not sufficient* —
  `Data` allocation derives its batch shape from `q` alone, dynamics
  allocate from that same shape, and frames are Python records. Batched
  model values therefore also need a defined q-batch × value-batch
  broadcast contract (see 03 §5).
- `Frame` placements are frozen per-object `(7,)` structs. Shape-dependent
  markers (SMPL landmarks) cannot ride the residual stack, and
  `Model.to()` does not move frames — on GPU models this costs a
  host-to-device copy per frame per call.
- `build_model` forcibly reorders joints depth-first; both consumers wrote
  ~180-line name-keyed q-remap shims to undo it.
- Spherical-joint "limits" are inert `[-1, 1]` quaternion boxes; human
  models need swing/twist limits and per-joint rotation priors.
- On the plus side, BR already has correct spherical and free-flyer
  manifold math (roundtrip error 1e-8) and a battle-tested programmatic
  tree builder — the foundation is genuinely close.

### 2.4 The performance posture contradicts the project's own rules

The stated rules (no `.item()` in hot paths, no host syncs, stay
compile-friendly) are violated systematically, and the hot-path lint is
too narrow to notice. A "host sync" means the CPU stops and waits for the
GPU so it can read one value — it stalls the pipeline and breaks
`torch.compile` and CUDA-graph capture.

- Free-flyer FK executes a `bool((...).any())` quaternion-norm check on
  **every call** — a forced host sync and a verified
  `torch.compile(fullgraph=True)` breaker. Fixed-base FK compiles fine,
  with a measured 5× speedup; free-flyer fails on exactly this line.
- Every optimizer performs 3–5 `float()`/`bool()` host syncs per
  iteration, plus builds a fresh `torch.eye` every LM iteration.
- Constants are constructed per call (source-inspected, **not
  GPU-measured** — CUDA is broken on this box; note that `.to()` is a
  no-op when device and dtype already match): `r.new_tensor([...])` in
  `pose.py:65,102` (an allocation per call — the same pattern as the
  just-fixed `so3_inverse` bug); module-level CPU axis tensors `.to()`-ed
  per joint per FK call in revolute/prismatic/helical (a real
  host-to-device copy for CUDA models); 6×6 spatial inertias rebuilt from
  the packed 10-vector on every dynamics call. Hoisting these is right
  regardless; their measured GPU impact is an open item for the M6
  profiling pass.
- FK's per-joint Python loop carries ~4 ms of fixed overhead (B=1 costs
  the same as B=256 on CPU). `update_frame_placements` is a per-frame
  Python loop with a per-frame `.to()` — one third of FK+frames time.
  `Model.integrate/difference` also loop per joint in Python. Measured
  slowdown versus a vectorized twin: 2.4× (audit, 24 joints) and 3.66×
  (codex re-probe, 24-joint SMPL-like model, T=200). BVR's "4–10× at 52
  joints" is that consumer's own claim — plausible, but not independently
  verified. Either way, BVR already re-implemented the vectorized version
  for itself.
- Ceremony without payoff elsewhere: `solve_ik` runs 11 FK calls in a
  5-iteration solve (no state caching between residual and Jacobian
  evaluation) — 108 ms against the project's own budget of 50 ms for 30
  iterations.

### 2.5 Speculative ceremony and dead code

For an unreleased library, BR carries a lot of machinery that serves
no one: 18 deprecated `Data` alias shims plus a no-legacy-strings contract
test; an IR schema-version handshake that nothing ever pickles; `utils/`
100 % unimported; a residual registry (`get_residual`) with zero callers;
`ResidualSpec` + `jacobian_blocks` with no consumer; `dynamics/action/`
(a Crocoddyl-style three-layer skeleton, ~290 lines) with no DDP/iLQR
solver and unusably slow autograd Jacobians per knot; the now-removed
`graph_capture` no-op; `OptimizerConfig` options (`"cg"`, `"trust_region"`) that crash
when selected; a `retarget` stub frozen into the 26-symbol public API
contract; a `rich` dependency that is never imported; and
`JointComposite`/`JointHelical`, which are constructible but unreachable
from any parser (ModelBuilder emits kind strings that `build_model`
rejects — verified `IRError` for `add_helical` and
`add_joint(kind=JointRX())`).

**Migration caveat (codex review; revised 2026-07-17):** not everything
that looks dead is dead to the *consumers*. BetterHumanForce imports
`better_robot.costs.stack.CostStack` (`scripts/motion/optimize_motion.py:210`)
and uses the deprecated `Data.oMi` surface — including *assignments*
(`tools/robot_motion/playback.py:115,180,190`, `motion.py:201`); the full
legacy surface also includes `GaussNewton.minimize`
(`tools/geometry/icp.py:58,330-332`), the Huber/Cauchy kernels
(`tools/object_align/sdf_fit.py:40-41`), `Trajectory`, trajectory
residuals, and `ResidualState`. Owner decision: the redesign lands on a
dedicated branch and the consumers stay on the pre-redesign branch until
the M4 migration — so shims are **not** required; the legacy surface is
recorded in the M4 migration table instead (see the roadmap's branch
strategy rule).

### 2.6 The docs cannot be trusted as a stub inventory

6 of 10 sampled documentation claims were wrong (quality audit): the
front-page IK snippet raises `KeyError` (frames are named
`body_panda_hand`; the docs — and CLAUDE.md — say `panda_hand`; partial
correction 2026-07-17: the repo `README.md` now uses `body_panda_hand`,
but `docs/index.md:36,39` and the root `CLAUDE.md` are still broken);
`lower_pos_limit` is documented as `(nv,)` but is `(nq,)`; the roadmap
rule "not on this page ⇒ implemented and tested" is false for ~10 symbols;
CLAUDE.md uses the deprecated `data.J`; the bench README cites CI that
does not exist (plus a doc path, `docs/claude_plan/accepted/12_...md`,
that does not exist, and `_status: PLACEHOLDER` baselines). Further rows
found 2026-07-17: `docs/concepts/solver_stack.md` documents a
`"matrix_free"` structure value, a block-Cholesky solver that "reads
jacobian_blocks", a `SparseCholesky` "default for trajopt", and a
`"sparse_cholesky"` config literal — none of which exist; and
`residuals/CLAUDE.md` advertises banded structure for
`Velocity/AccelerationResidual`, which never implemented `.spec`. Docs
drift is a systemic failure mode of how this code was
produced. The fix is executable snippets and a generated stub inventory —
not one-off corrections.

---

## 3. What is genuinely good (keep, and build on)

- **The Model/Data split, and the free-flyer as joint 1.** A single code
  path for fixed and floating base genuinely holds through FK, Jacobians,
  and dynamics. This matches the pinocchio/pyroki/brax consensus and is
  the right skeleton.
- **The Lie math core.** The *values* are correct: Shepperd's 4-branch
  quaternion conversion, Taylor-stitched exp/log, the scalar-last
  convention enforced everywhere. Dropping PyPose was the right call. Only
  the gradient-at-singularity handling is broken (§1.1).
- **The pinocchio-parity test suite** (`tests/test_pinocchio/`, absolute
  tolerance 2e-6, plus fp64 gradient checks). The most valuable QA asset
  in the repo — it is what makes aggressive refactoring safe.
- **The raw-pass boundary.** `forward_kinematics_raw` now consumes
  `ModelStructure`, `ModelValues`, and `q`; `rnea_raw` follows the same
  pattern. The fixed-base Torch FK was verified to compile at a 5× speedup.
  `_compute_joint_jacobians_raw` still consumes `Model` / `Data`, so it must
  not be described as part of the pure seam yet.
- **`JointModel` per-kind stateless dispatch** — the right shape; it just
  needs the axis-packing fix.
- **The IR → `build_model` → frozen Model pipeline** — right-sized, not
  over-abstracted. Its defects are specific bugs (kind mapping, mimic,
  forced DFS ordering), not the architecture.
- **The Featherstone implementations read correctly** (RNEA/ABA/CRBA
  checked against RBDA and pinocchio conventions; the LWA frame-Jacobian
  handling and its documented adjoint pitfall are right).
- **What consumers already love** (real adoption evidence): the
  `se3`/`so3` facades, `build_kinematic_tree_model` with per-body inertia,
  `rnea(fext=...)` with autograd through `fext`, batched FK,
  `CostStack.gradient`, and the standalone Huber/Cauchy kernels.
- **Contract tests as an idea** — enforcing the layer DAG by parsing the
  AST is real and verified. The lint just needs wider nets, and the
  frozen-API test should stay unfrozen until 1.0.

---

## 4. Suspect — verify before acting

- All GPU-sync magnitudes (no working CUDA here) — profile on a real GPU
  box before sizing any performance work.
- Deep autograd graphs from sequential FK composition over long trajectory
  horizons — a plausible slow-backward risk, unmeasured.
- `JointPlanar` sign handling at θ = π; `se3_log` gradient quality near
  θ = π.
- frax-style ancestor-mask RNEA/CRBA (see `references/design/frax.md`) is
  O(N²)–O(N³) dense — needs profiling at 50+ DOF before adopting it for
  SMPL-scale humans.
- ~~The breadth of pinocchio-parity coverage across joint kinds and batch
  shapes~~ — **resolved 2026-07-17, the suspicion was right:** FK parity
  is unbatched-only; RNEA/CRBA/ABA have single-axis batched-q tests only;
  no batched-values parity exists anywhere; the suite is effectively
  fp64-only; planar/translation/helical/revolute_unaligned/mimic kinds
  are uncovered. M3 task 1 closes these gaps before the indexing sweep.
