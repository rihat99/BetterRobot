# M2a — Variables, Evaluation Protocol, and the Vertical Slice: Agent Execution Instructions

> **Implementation log (2026-07-17):** Complete on `dev`. The owner confirmed
> scalar `ObjectiveItem` support with a strict GN/LM fence and the recommended
> differentiation contract; CI is intentionally stopped. See `m2a_results.md`.

> Read `plan/for_agents/README.md` first. It carries the standing rules
> (deletion ordering, honesty rules, kernel requirements, test commands).

## Mission

Replace the single-flat-variable optimization core with **named variable
blocks**: manifolds with feasible retraction, a `VarSpec`/`Values`/`Problem`
draft where residuals declare what they read, a provider DAG with
evaluation-local caching, and a per-block-dimension AD strategy. This is the
single biggest redesign of the plan — it is what unblocks both consumer
repos (their ~24 hand-rolled `torch.optim` call sites across 10 first-party
files are all multi-block problems BR cannot express today) and better_human
(betas become just another block). The milestone's own centerpiece rule:
**nothing built here is public API until one real consumer problem — the
vertical slice — runs end-to-end against the draft.** The roadmap's
done-when, verbatim: *"the slice runs end-to-end on the draft API, and only
then is the API declared public; a custom-residual author guide exists and
was followed verbatim by the slice."*

## Prerequisites

- **M0 — the θ=0 gradient fix** (`m0_truth_and_correctness.md`). The
  tangent-space autograd helper (T2a.2) differentiates `f(values ⊕ δ)` at
  δ = 0, which routes every gradient through `so3/se3 exp` at exactly the
  singularity. Verify it landed by running:

  ```bash
  uv run python -c "
  import torch
  from better_robot.lie import so3
  x = torch.zeros(3, requires_grad=True)
  g = torch.autograd.grad(so3.exp(x).sum(), x)[0]
  assert torch.isfinite(g).all(), f'theta=0 NaN gradient — M0 not landed: {g}'
  print('OK', g)"
  ```

  As of 2026-07-17 this prints `[nan, nan, nan]` — the fix has **not**
  landed (`lie/_torch_native_backend.py:148` still computes
  `theta = theta2.clamp(min=0.0).sqrt()` and `qw = torch.cos(half)` at
  line 159 sits outside the Taylor `where`; same pattern at
  `lie/tangents.py:86`). Do not start T2a.2 before this passes, including
  M0's second-order and `JointSpherical.difference` coverage.
- **M0 — enum-lie deletion.** `grep -n "AUTODIFF" src/better_robot/kinematics/jacobian_strategy.py`
  must come back empty (today it does not — the values are still at
  `jacobian_strategy.py:21-23`). T2a.6 re-introduces real autodiff; it must
  not land while the lying enum values still exist.
- **M1 — the seam and hygiene** (`m1_two_lane_seam_and_hygiene.md`).
  Checks: `src/better_robot/backends/` no longer exists;
  `grep -rn "class ModelStructure" src/better_robot/` hits;
  the 26-symbol API freeze in `tests/contract/test_public_api.py` is
  unfrozen (M1 item 8); the cross-cutting engineering contract document
  from M1 item 5 exists (T2a.10 extends it). If the owner green-lights
  overlap, only T2a.1, T2a.2 and T2a.10 may start before M1's checklist
  passes — they depend on M0 alone. T2a.3–T2a.8 build on the M1 tree.
- Consumer repos readable at
  `/data3/rikhat.akizhanov/better/BetterVideoReconstruction` (BVR) and
  `/data3/rikhat.akizhanov/better/BetterHumanForce` (BHF). They are
  **read-only evidence** — never modify them in this milestone.

## Sizing & parallelism

Roadmap label: **L** (multi-week), the largest single milestone.

Dependency order (explicit):

```
Stage 1 (foundations, ordered):   T2a.1 → T2a.2
Stage 2 (the draft):              T2a.3 first (defines the types),
                                  then T2a.4, T2a.5, T2a.6, T2a.7 in parallel
Stage 3 (the gate, ordered):      T2a.9 (guide draft) → T2a.8 (the slice)
                                  → T2a.9 revision → freeze
In parallel from day one:         T2a.10 (must land before the freeze)
```

T2a.4–T2a.7 can go to parallel agents once T2a.3's type definitions are
committed (they all consume `VarSpec`/`Values`). T2a.8 is a single-agent
task — the slice is where API friction is discovered, and that feedback
must flow back into T2a.3–T2a.7 from one head.

**Freeze discipline (binding):** every new module lands *unexported* — not
in `better_robot/__init__.py`, not in the API-contract test, no docs page
declaring it public. The exports, the contract-test update, and the docs
land as the final commit of the milestone, after T2a.8 passes and the
T2a.5/T2a.10 owner reviews are done.

## Tasks

### T2a.1 — Manifold protocol + Euclidean/SO3/SE3/RobotConfig + state-space bounds  [M]

**Goal / done-when:** A `Manifold` protocol owning
`retract(x, dv)`, `difference(x0, x1)`, `tangent_dim(shape)`; four
implementations: `Euclidean`, `SO3`, `SE3`, `RobotConfig(model)`.
State-space bounds (`Bounds`, nq-shaped for `RobotConfig`) are enforced by
**feasible retraction** — a retraction result is always feasible.
Constructing an `SO3`/`SE3` block with bounds raises. All tested.

**Current state:** the only manifold hook is one `retract` callable on
`LeastSquaresProblem` (`optim/problem.py:34-36`) plus one `(lower, upper)`
pair clamped after the step; `Model.integrate`/`difference` are per-joint
Python loops (`data_model/model.py:160-190`) dispatching to
`JointModel.integrate/difference` (e.g.
`data_model/joint_models/spherical.py:52-62`). There is no `Manifold`,
`VarSpec`, or `Bounds` type anywhere in `src/` (grep-verified).

**Implementation plan:**

1. Create the draft package — suggested `src/better_robot/optim/blocks/`
   with `manifolds.py` and `variables.py` (invoke the `file-naming` skill
   before fixing names; the package stays unexported per the freeze
   discipline). Layer DAG: `optim` already sits above `data_model`, so
   `RobotConfig(model)` importing `Model` is legal.
2. `Manifold` protocol (runtime-checkable, matching the repo's existing
   Protocol style). `Euclidean`: `x + dv`. `SO3`/`SE3`: **right/local
   perturbation, matching the repo convention** — `retract(q, ω) =
   normalize(compose(q, exp(ω)))`, `difference(q0, q1) =
   log(compose(inverse(q0), q1))` — exactly what
   `spherical.py:52-62` and `free_flyer.py` do today. `RobotConfig(model)`
   wraps `model.integrate`/`model.difference`; `tangent_dim` returns
   `model.nv` (this is where nq ≠ nv lives — nowhere else).
3. `Bounds`: state-space lower/upper (shape = the block's *state* event
   shape; `(nq,)` for `RobotConfig`). Feasible retraction: retract, then
   project in configuration space. For `RobotConfig`, projection touches
   only box-boundable coordinates (revolute/prismatic/helical slices);
   quaternion slices (spherical, free-flyer orientation) are never
   clamped — validate at `Bounds` construction that quaternion coordinate
   slices are ±inf, else raise.
4. `SO3()`/`SE3()` with `bounds is not None` raises at `VarSpec`
   construction. Draft wording:

   > `ValueError: SO3/SE3 variable blocks have no meaningful global box
   > bound — neither in state space nor in tangent space. Express rotation
   > limits as residuals (rotation prior / swing-twist, roadmap M3), or use
   > RobotConfig with joint limits. Got bounds=... on VarSpec '...'.`

5. Decide and document initial-feasibility behavior: recommend
   **validate-and-raise** on infeasible initial values (mirrors the
   documented LM rule "initial x0 is NOT clamped — caller provides a
   feasible start", repo `CLAUDE.md`), with the Panda joint-4 case
   (`q_neutral` outside `[-3.07, -0.07]`) called out in the error text.

**Constraint 1 from the review (binding, restate in the module docstring):
two kinds of bounds, never conflated.** State feasibility lives in
configuration space and belongs to the manifold (this task). Per-step
bounds (trust regions, step clamps) live in tangent space and belong to
the **solver** — that is M2b (`m2b_batched_second_order_solvers.md`)
territory. Nothing in `Manifold`/`Bounds` may take a tangent-space box.

**What to test** (`tests/optim/test_manifolds.py`; float32 only, per the
workspace `write-tests` skill — invoke it first):

- Retract/difference roundtrip: `difference(x, retract(x, dv)) ≈ dv` for
  small `dv`, each manifold, batch shapes `()`, `(4,)`, `(2, 3)`.
- `RobotConfig` on the Panda (nq = nv = 9) and on a free-flyer model
  (nq = nv + 1): `tangent_dim`, roundtrip, and parity with
  `model.integrate`/`difference` outputs (exact — it's a wrapper).
- Feasible retraction: a step pushing a bounded coordinate outside its box
  lands exactly on the bound; quaternion coordinates unchanged and unit.
- `SO3()`+bounds and quaternion-slice bounds raise, exact messages
  asserted.
- Infeasible initial values raise (Panda `q_neutral` un-clamped as the
  fixture).

**Pitfalls / do-not-forget:**
- Do not vectorize `Model.integrate/difference` here — that is M3 item 6.
  Wrap what exists.
- The perturbation side matters: the analytic residual Jacobians
  (`residuals/pose.py`) are derived for the local/right convention. A
  left-perturbation SO3 manifold would pass roundtrip tests and silently
  break Jacobian parity later.
- Keep the old `LeastSquaresProblem` and its `retract` untouched — it dies
  in M2c, with the `solve_ik` re-base, not here (deletion ordering rule).

### T2a.2 — The tangent-space autograd helper  [S–M]

**Goal / done-when:** a helper that differentiates `f(values ⊕ δ)` at
δ = 0 (jaxlie's `manifold.grad`/`manifold.rplus` pattern —
`references/design/jaxlie.md`), returning per-block tangent gradients as a
`Values`-shaped dict. Gradcheck passes **at** singular points. This is what
makes first-order methods manifold-correct with zero optimizer changes
(quaternions never see a Euclidean step).

**Current state:** nothing like it exists; `torch.func` appears nowhere in
`src/` (grep-verified — the only hits are the lying docstrings in
`jacobian_strategy.py`). The FD fallback (`kinematics/jacobian.py:246-284`)
hardwires `model.integrate` and `model.nv` and is the only "autodiff"
today.

**Implementation plan:**

1. `optim/blocks/autograd.py` (name via `file-naming`): given
   `f: Values -> Tensor`, `specs`, and `values`, build zero tangents
   `δ = {name: zeros(B..., tangent_dim)}` with `requires_grad=True`,
   evaluate `f({name: spec.manifold.retract(values[name], δ[name])})`, and
   return `torch.autograd.grad` results per block. Masked (fixed)
   coordinates get their gradient entries dropped/gathered per T2a.3's
   elimination semantics.
2. Support `create_graph=True` pass-through (T2a.10 decides the guarantee
   level; the helper must not structurally preclude it).
3. This helper is also the substrate for T2a.6's `jacrev`/`jacfwd` paths —
   keep the "perturb-then-evaluate" closure factored out so all three share
   it.

**What to test** (`tests/optim/test_tangent_autograd.py`, float32,
`gradcheck(..., eps=1e-3, atol=1e-2)` per the `write-tests` skill):

- Gradcheck of `δ ↦ f(values ⊕ δ)` at δ = 0 for each manifold, with
  `values` at singular points: identity rotation (θ = 0), two identical
  quaternions differenced, θ near π. This is the test that requires M0.
- A no-NaN assertion on an SMPL-like rest-pose gradient (the
  96-of-99-NaN case from `plan/01_assessment.md §1.1`), through
  `RobotConfig` on a spherical-joint tree from
  `build_kinematic_tree_model`.
- Gradient parity against the existing analytic path on a Panda pose
  residual (tolerance ~1e-3 fp32).
- A `create_graph=True` smoke test (second-order numbers checked in
  T2a.6's tests).

**Pitfalls:** the double-cover convention — `so3_log` flips sign on
`qw < 0`; gradcheck fixtures must not straddle the hemisphere boundary or
FD will disagree for reasons that are not bugs. Sample near-identity and
use the M1 engineering contract's quaternion-continuity policy.

### T2a.3 — Draft `VarSpec` / `Values` / `Problem`  [L]

**Goal / done-when:** the block-world types exist as an unexported draft:
residuals declare `reads`; `jacobian_blocks` is keyed
`(residual_name, var_name)`; masks **eliminate** fixed coordinates;
per-coordinate `scale` is plumbed through. The 03 §3 sketch, reproduced
here **as direction, not final signatures** (small deviations are fine
when the code demands them — document each):

```python
@dataclass(frozen=True)
class VarSpec:
    name: str
    shape: tuple[int, ...]          # event shape; batch axes are separate and explicit
    manifold: Manifold = Euclidean()  # Euclidean | SO3 | SE3 | RobotConfig(model)
    bounds: Bounds | None = None    # STATE-space feasibility (nq-shaped for RobotConfig),
                                    # enforced by feasible retraction/projection —
                                    # distinct from any per-step trust-region bound
    scale: Tensor | None = None     # per-coordinate scaling for damping/preconditioning
    mask: Tensor | None = None      # frozen DOFs (per-phase overridable; fixed
                                    # coordinates are ELIMINATED from the system,
                                    # not zero-columned into a singular one)

# A Manifold owns: retract(x, dv), difference(x0, x1), tangent_dim(shape).
# RobotConfig(model) wraps model.integrate/difference — nq != nv handled here.

Values = dict[str, Tensor]          # name -> (B..., *shape)

class Problem:
    vars: tuple[VarSpec, ...]
    residuals: tuple[ResidualItem, ...]   # residual + weight + kernel + name

    def residual(self, values: Values) -> Tensor          # (B..., dim_total)
    def gradient(self, values: Values) -> Values          # tangent-space, per block
    def jacobian_blocks(self, values: Values) -> dict[tuple[str, str], Tensor]
        # (residual_name, var_name) -> (B..., dim_i, tangent_dim_j); absent = zero
```

**Current state:** `LeastSquaresProblem` (`optim/problem.py:23-36`) holds
one flat `x0`, one `retract`, one bounds pair, and a private `_nv` that
BHF duck-types (`BHF tools/geometry/_icp_problem.py:99,142`).
`ResidualState` (`residuals/base.py:29-45`) mandates `(model, data,
variables)`, and built-in residuals read `variables` as q by fiat. This is
the verified consumer blocker (`plan/01_assessment.md §2.2`). BVR's
`tools/optim.py` (188 lines: `Problem`/`Phase`/`run_phases`, lazy `State`,
per-phase weight columns, per-leaf grad masks) is the de-facto
requirements spec — read it before designing.

**Implementation plan:**

1. `optim/blocks/variables.py` + `optim/blocks/problem.py`. `VarSpec`
   frozen; validate `scale`/`mask` shapes against
   `manifold.tangent_dim(shape)` at construction (both are
   **tangent-dim-shaped** — a mask freezes DOFs, not q coordinates; for
   `RobotConfig` that means nv-shaped, while `bounds` is nq-shaped.
   Spell this asymmetry out in the docstrings; it is constraint 1 again).
2. **Mask semantics = elimination.** Precompute per-block free-index
   tensors (`torch.nonzero` at spec/problem build, not per evaluation).
   `gradient()` and `jacobian_blocks()` return **reduced** tangent
   dimensions (`tangent_dim_j` = number of free coordinates); `Values`
   always stay full-state. Provide the expand/gather mapping as a small
   public-on-freeze utility so solvers can scatter steps back. A zeroed
   column must never reach a normal system (codex B2.2: it makes JᵀJ
   singular).
3. **`scale`** is carried, validated, and exposed on the assembled blocks;
   its consumer is M2b's Madsen–Nielsen damping on scaled blocks
   (`m2b_batched_second_order_solvers.md`) — heterogeneous blocks
   (radians, meters, pixels, newtons) must not share one unscaled damping
   value. M2a only guarantees it is *there* and correctly indexed under
   masking.
4. **Residual protocol stays structural** (duck-typed — today's protocol
   style, which the optim audit §4 endorses): a new-style residual carries
   `name`, `reads: tuple[str, ...]`, `__call__(ctx) -> (B..., dim)`,
   optional `jacobian_blocks(ctx) -> dict[str, Tensor]` (keyed by var
   name), where `ctx` is the T2a.4 evaluation context (a plain mapping).
   **No imports from `optim/` are needed to author one** — this is what
   keeps the layer DAG intact when built-in residuals migrate in M2c.
5. `ResidualItem` = residual + weight + kernel + name; **per-item robust
   kernels** (03 §6 change 3) — record the kernel here; IRLS application
   stays solver-side (M2b), but the item is where the kernel lives.
6. Dense `dim_total` bookkeeping: residual dims must be static per
   evaluation (no `self.dim` mutation — the audit §2.10 pathology).
7. Do not touch `LeastSquaresProblem`, `CostStack`, or `solve_ik` — the
   old stack keeps working beside the draft until M2c.

**What to test** (`tests/optim/test_problem_blocks.py`):

- A toy two-block problem (e.g. Euclidean 3-vector + SO3): `residual`
  shape `(B..., dim_total)`; `gradient` returns per-block tangent shapes;
  `jacobian_blocks` keys — a residual that reads one block yields no key
  for the other (absent = structurally zero, asserted).
- Masks: freeze two coordinates; assert reduced column counts, that the
  assembled normal system (T2a.7) is nonsingular where the zero-column
  variant would be singular, and that expand/gather roundtrips.
- `scale`/`mask` shape validation errors (nq-vs-nv confusion must raise
  with a message naming both shapes).
- Batched evaluation: `(4,)`-batched values produce per-element residuals
  matching 4 sequential evaluations (stated tolerance, per standing
  rule 3).

**Pitfalls:**
- `Values` is `name -> (B..., *shape)`: batch axes are *separate and
  explicit*, never folded into `shape`. A trajectory is one block with
  event shape `(T, nq)` — T is coupled (smoothness reads across it), so it
  is event, not batch. Write this distinction into the `VarSpec`
  docstring; it was codex B2.2's ambiguity complaint.
- Reproduce the sketch's comment lines (bounds/scale/mask semantics) in
  the real docstrings — they are the review constraints in compressed
  form.
- `jacobian_blocks` on `Problem` is keyed `(residual_name, var_name)`; on
  a *residual* it is keyed by var name only. Keep the two namespaces
  straight.

### T2a.4 — Provider DAG + evaluation-local context  [M]

**Goal / done-when:** review constraints 2 and 3. Providers declare
`inputs` (variable names and/or other providers' outputs) and `outputs`;
the problem toposorts them and rejects cycles. A
`RobotStateProvider(model, var="q")` computes `Data` (FK) **once per
evaluation**; every kinematic residual reads it from the context. One
expensive shared pass (the consumers' nearest-neighbour pass) feeds
several residuals, computed at most once and only if an active residual
asks. **No persistent caches keyed on mutable values versions. Nothing
outlives the iteration except detached accepted-state artifacts.**

**Current state:** every `state_factory` call re-runs FK — `solve_ik` runs
11 FK calls in a 5-iteration solve (`plan/01_assessment.md §2.4`); M1
item 11 added a per-iterate stopgap cache. The proven consumer pattern is
BVR's lazy `State` (`BVR tools/optim.py:27-50`: `put`/`put_lazy`, maker
called at most once, only when an active term reads it) and its shared
scene pass: `scene_signed_distance` (`BVR
tools/human_optim/losses.py:190` — docstring: *"One detached
nearest-neighbour pass, shared by the three scene losses"* —
penetration/attraction/clearance differ only in the penalty applied to
its output). BHF's equivalent utility is `SpatialHashNN`
(`BHF tools/geometry/nn_search.py:48`). **Note:** 03 §3 constraint 3 calls
this "one neural-network pass"; in the consumer code "NN" is
*nearest-neighbour*. The requirement is unchanged either way: one
expensive shared pass, declared once, feeding ≥3 residuals.

**Implementation plan:**

1. `optim/blocks/providers.py`: a structural `Provider` protocol —
   `name`, `inputs: tuple[str, ...]`, `outputs: tuple[str, ...]`,
   `__call__(ctx) -> dict[str, Any]`. Namespace variables and provider
   outputs together (a residual's `reads` may name either); reject
   collisions at problem build.
2. Toposort at `Problem` construction (static — the DAG never changes per
   evaluation). Cycle error, draft wording:

   > `ValueError: provider dependency cycle: nn_pass -> sdf -> nn_pass.
   > Providers must form a DAG over declared inputs/outputs.`

3. The evaluation context: created per `residual`/`gradient`/
   `jacobian_blocks` call, seeded with `values`, **lazy** per provider
   (BVR's `put_lazy` semantics — a provider whose outputs no active
   residual reads never runs). Immutable from the residual side (a
   read-only mapping view is enough; don't over-engineer).
4. `RobotStateProvider(model, var="q")`: runs
   `forward_kinematics(model, q, compute_frames=True)` once, exposes
   `data` (and convenience keys as needed by the slice). This replaces
   today's `state_factory` concept in the block world.
5. **Lifetime rules, enforced by design not comments:** the context is a
   local of the evaluation call — no field on `Problem` or the residuals
   holds it. Accepted-state artifacts a solver wants to keep (best-so-far
   values, diagnostics) are `.detach()`ed by the solver, never cached
   inside providers. No "values version" integers, no id()-keyed dicts —
   the review verdict is explicit that mutation-invisible tensors make
   such caches unsafe (`codex_plan_review.md §B2.6`).

**What to test** (`tests/optim/test_providers.py`):

- A counting provider read by three residuals runs exactly once per
  `residual()` call and once per `gradient()` call; a second evaluation
  recomputes (no cross-evaluation reuse).
- An inactive-consumer provider (zero-weight residual / not in the active
  set) never runs.
- Cycle rejection, exact message. Unknown `reads` name rejection at build.
- `RobotStateProvider`: one FK per evaluation, verified by
  monkeypatch-counting `forward_kinematics`; a Panda pose-residual solve
  built on the draft does ≤ 2 FK per iteration (residual + Jacobian
  evaluation sharing is the point).
- Context does not leak: after an evaluation, no reference to the context
  survives on the problem/residuals (weakref liveness test), and a kept
  "accepted artifact" has `grad_fn is None`.

**Pitfalls:**
- Do NOT let the context cache across the residual-then-Jacobian pair by
  silently persisting on `self` — share by *passing the same context*
  within one logical evaluation where the caller (solver) chooses to, and
  document that AD transforms (T2a.6) may need fresh contexts per
  perturbation. Graph-carrying cached tensors are how second backward
  breaks (constraint 2's whole point).
- Detached provider outputs are legitimate and common (the NN pass is
  detached by design — gradients flow through the residual's use of `v`,
  not through the argmin). The author guide (T2a.9) must cover declaring
  detachedness.

### T2a.5 — Scalar objective terms: decide via the slice  [M]

**Goal / done-when:** review constraint 4 is resolved **by evidence, not
assumption**. Either (A) an `ObjectiveTerm` protocol participates in
first-order solves and is rejected by GN/LM with a clear error, or (B) the
docs state a least-squares fence and the consumer keeps those terms
outside. The slice (T2a.8) provides the evidence; the owner signs off.

**Current state:** BR is strictly least-squares (`Residual` requires a
static `dim`, `residuals/base.py`); the consumers are not. **Every one of
BVR's ~15 kinematic-stage terms is a scalar** (`TermFn` returns a scalar
loss, `BVR tools/optim.py:53-55`; GM reprojection means, masked chamfer
averages, hinge penalties — `tools/human_optim/optimizer.py:483-560`), and
BHF's silhouette term is a scalar MSE (`tools/object_align/sdf_fit.py`,
`L_mask`). Several return auxiliary diagnostics
(`(loss, {"cm": ..., "n": ...})`).

**Implementation plan:**

1. Implement the slice with option (A) as the working hypothesis — the
   consumer evidence is heavily on its side: `ObjectiveTerm` = `name`,
   `reads`, `__call__(ctx) -> (B...,)` scalar (plus optional detached
   diagnostics dict, the BVR pattern). `Problem.gradient` adds its tangent
   gradient (via T2a.2); `Problem.residual`/`jacobian_blocks` ignore it.
2. GN/LM rejection (wire into the *existing* solvers' entry path only if
   trivially possible; otherwise this lands as a check in the M2b solver
   spec — record which). Draft wording:

   > `ValueError: problem contains scalar objective term(s) ['sil_mask'] —
   > Gauss-Newton/Levenberg-Marquardt minimize sums of squared residual
   > vectors and cannot consume scalar terms. Run these terms in a
   > first-order phase (Adam/LBFGS), or reformulate them as residual
   > vectors.`

3. After the slice runs, write up what option (A) actually cost
   (gradient-assembly complexity, mask/phase interaction, any weighting
   ambiguity) versus what option (B) would have left in consumer land —
   and **STOP for owner review** before the freeze (standing rule 8).
   Do not silently declare the default.

**What to test:** a scalar term's contribution to `gradient()` matches
autograd of the weighted sum (fp32 tolerance); the GN/LM rejection raises
the exact message; diagnostics pass through without forcing a host sync
(floats only at log time — the BVR rule, `tools/optim.py:80-82`).

### T2a.6 — AD strategy per block dimension  [M]

**Goal / done-when:** review constraint 6. A written decision table
choosing analytic vs `jacrev` vs `jacfwd` vs VJP/JVP per
(residual dim × tangent dim), implemented as the `jacobian_blocks`
fallback; graph lifetime and `create_graph` support specified **and
tested**; the evaluation path is `jacfwd`-clean.

**Current state:** there is no autodiff Jacobian at all — every
non-analytic path is central FD costing `2·nv+1` evaluations
(`kinematics/jacobian.py:246-284`), and the code is **not even
jacfwd-clean**: the codex probe hit a float/double dtype mismatch inside
the residual closure (`plan/01_assessment.md §1.5`,
`codex_plan_review.md §A5`). Measured calibration (same sources): analytic
~1.6 ms, real `jacrev` ~14 ms, FD ~62 ms on a Panda pose residual.

**Implementation plan:**

1. Decision rule (write it as a table in the module docstring, then follow
   it): analytic block when the residual provides one; otherwise
   reverse-mode (`torch.func.jacrev` over the T2a.2 perturbation closure)
   when residual dim ≤ tangent dim; forward-mode (`jacfwd`) when
   tangent dim < residual dim (e.g. a fat point-cloud residual over a
   6-DOF pose block); plain VJP (no materialized J) is the `gradient()`
   path always. Record measured crossovers from the slice problem as a
   committed micro-benchmark note (not a CI gate).
2. Make the closures dtype-clean: no Python-float literals promoted to
   float64 inside residual math, no `.double()` remnants — add a
   `jacfwd` smoke test that would have caught the current mismatch.
3. Graph lifetime: Jacobian computation must not retain graphs beyond the
   call unless `create_graph=True` is requested; specify (and test) that
   `create_graph=True` yields differentiable blocks (needed by T2a.10's
   contract and M6's implicit diff).
4. FD survives only as an explicit debug strategy (M0 already deleted the
   lying enum values — keep it that way; the new code's strategy knob
   only advertises what works).

**What to test** (`tests/optim/test_ad_strategies.py`, float32):

- Parity analytic vs jacrev vs jacfwd on the Panda pose residual through
  the block API (atol ~1e-3).
- jacrev/jacfwd on a masked block return reduced columns consistent with
  T2a.3's elimination.
- `create_graph=True`: gradcheck of a function *of* a Jacobian block
  (fp32, loose tolerances per the `write-tests` skill).
- A batched-values Jacobian matches per-element Jacobians (stated
  tolerance).

**Pitfalls:** `torch.func` transforms want pure functions — this is
exactly why T2a.4 forbids hidden mutable caches; if a provider must run
inside the differentiated closure (gradients flow through FK), it runs
inside; if its output is detached (NN pass), hoist it outside the closure
and document the split in the author guide.

### T2a.7 — Dense assembly; sparsity is a stated non-goal  [S–M]

**Goal / done-when:** review constraint 5. Dense assembly of the full `J`
(and/or `JᵀJ` blocks) from `jacobian_blocks` for IK-sized problems, with a
written assembly order (var blocks → column offsets, residual items → row
offsets). The banded/temporal structure information in today's
`ResidualSpec` is preserved as **design notes** for M5 — not as shipped
dead code.

**Current state:** `ResidualSpec`
(`optim/jacobian_spec.py`: `structure: dense|diagonal|block|banded`,
`time_coupling: single|5-point|custom`, `affected_knots/joints/frames`)
has zero consumers (audit §2.7) and is slated for deletion (03 §10) —
check whether M1's de-bloat already removed it. Codex B2.5's warning
stands: block structure is an API property, not a performance result — a
dict of blocks can be *slower* than today's one dense Jacobian if
assembled naively.

**Implementation plan:**

1. Assembly in `optim/blocks/problem.py`: deterministic ordering (vars in
   `Problem.vars` order, residuals in `residuals` order), preallocated
   `(B..., dim_total, tangent_total_free)` fill — no per-block Python
   `torch.cat` chains in the hot path.
2. State the non-goal in the module docstring: *dense assembly is v1;
   a trajectory remains one big dense q-block; symbolic sparsity, banded
   solvers, and Schur elimination are M5*
   (`m5_sparse_trajectory_structure.md`).
3. Preserve the `ResidualSpec` knowledge: copy its field semantics
   (structure kinds, time coupling, affected-knots indexing) into a short
   design-notes section — put it where
   `m5_sparse_trajectory_structure.md` says its design notes live; if that
   file predates yours, add `plan/design_notes/residual_sparsity.md` and
   tell the M5 executor via a one-line pointer in your PR description.
   Only after the notes exist may `ResidualSpec` be deleted (if M1 hasn't
   already — in that case verify the notes exist before building on them).
4. Commit a benchmark definition (hardware, dtype, shapes, warmup,
   statistics — standing rule 4) comparing draft-API assembly against the
   current `LeastSquaresProblem.jacobian` on Panda IK. This is the
   number M2b's acceptance will reference; a regression is a finding to
   report, not silently accept.

**What to test:** assembled dense `J` equals the autodiff Jacobian of the
concatenated residual on the toy two-block problem (fp32 tolerance);
masked columns absent; row/column offset bookkeeping asserted against
hand-computed indices; the benchmark script runs.

### T2a.8 — THE VERTICAL SLICE  [L] — the gate

**Goal / done-when:** review constraint 7, and the milestone's acceptance:
one real consumer problem implemented against the draft API **before**
freezing it, containing all seven ingredients: **two variable blocks, one
shared provider (the nearest-neighbour pass), one custom residual, one
scalar term, masks, batching, and a phase transition.** Only after the
slice runs end-to-end is the API declared public. This is the plan's own
two-caller rule applied to its centerpiece.

**Current state — read the consumer code and pick the problem.** Both
candidate directories, with what each offers (verified 2026-07-17):

- **BVR human_optim** —
  `/data3/rikhat.akizhanov/better/BetterVideoReconstruction/tools/human_optim/`:
  - `optimizer.py:398-599` (`optimize_person`): leaf `q (T, nq)` with
    per-phase DOF grad masks, ~15 scalar terms, the lazy `"sig"` NN pass
    shared by penetration/attraction/clearance, phases
    root→full→refine→lbfgs (`stages.py:363-395` builds the `PhaseSpec`s).
  - `prefit.py:201-202`: a natural **two-block** problem — leaves `q` and
    `log_s` — with GM reprojection, chamfer, priors, and a grad mask.
  - `losses.py:190-244` (`scene_signed_distance`): the shared detached
    nearest-neighbour pass.
  - The engine itself: `tools/optim.py` (State/Problem/Phase/run_phases).
- **BHF object-align** —
  `/data3/rikhat.akizhanov/better/BetterHumanForce/tools/object_align/`:
  - `sdf_fit.py`: three leaves (`delta_omega` — note the 1e-6-not-0 init
    at `sdf_fit.py:358`, the θ=0 NaN tell — `t_param`, `log_s`), BR
    `Cauchy`/`Huber` kernels, kernel-`c` annealing (a phase-like
    schedule), a scalar silhouette mask term, a UDF `grid_sample` data
    pass.
  - `polish.py`, `refine.py`: the stage sequencing around it.
  - `../geometry/nn_search.py:48` (`SpatialHashNN`): the batched NN
    utility.

Recommended composition (executor's call after reading — record the
choice and why): the **BVR shape** covers every ingredient most directly —
blocks `q` + `log_s` (from `prefit.py`), the scene NN provider feeding
penetration/attraction/clearance-style residuals (from
`optimizer.py`/`losses.py`), scalar smoothness/prior terms, per-phase DOF
masks, and a real phase transition. A BHF-flavored slice (pose blocks
`δω`/`t`/`log_s` + robust-kernel data term + scalar mask term) is equally
valid if it demonstrably ticks all seven boxes.

**Implementation plan:**

1. The slice lives in BR as a permanent test:
   `tests/optim/test_vertical_slice.py` (plus a small
   `tests/optim/slice_support.py` if needed). **Synthetic data only** —
   replicate the problem *structure* (shapes, term types, masking, phases)
   at toy scale (T ≤ 10, a few hundred points), runnable on CPU in
   seconds. Do not import from the consumer repos.
2. Drive it with a plain hand-rolled `torch.optim.Adam` loop over
   `Problem.gradient` — **M2a ships no solver**. The slice exercises the
   evaluation protocol ("projects keep their loop; BR owns the step");
   batched/second-order solving is M2b, the phase *engine* is M2c.
3. "Batching" means **batched evaluation**: run the same slice with a
   leading batch axis (B = 3) on the values and assert per-element
   residual/gradient parity with sequential evaluation at stated
   tolerances (standing rule 3). Per-element accept/reject does not exist
   yet and is not required here.
4. "Phase transition" means a **manual** one: two run segments with
   different weight columns and a different mask on `q` (BVR's
   root→full pattern), proving weights/masks are per-evaluation inputs or
   cheaply rebuilt — without building `Phase` (M2c).
5. The custom residual is written by following the T2a.9 guide
   **verbatim** — literally execute its steps as written; every point
   where you must deviate or guess is a guide bug: fix the guide (and the
   API if the friction is real).
6. Also assert the efficiency claim that motivates providers: count FK /
   NN-pass invocations per iteration (exactly one each per evaluation).
7. Keep a running friction log (API deviations from the 03 §3 sketch,
   signature changes, missing hooks). Feed each item back into
   T2a.3–T2a.7. This log plus the T2a.5 evidence is the owner-review
   packet.
8. **Only after** the slice passes and the owner reviews T2a.5/T2a.10:
   export the public symbols, update
   `tests/contract/test_public_api.py`, add the docs pages, and update the
   repo `CLAUDE.md` optimization section. That commit is the freeze.

**What to test:** the slice file *is* the test. Its docstring carries the
seven-ingredient checklist with a line each stating where the ingredient
appears. It must: converge (loss decreases to a fixed threshold on the
synthetic data with a fixed seed), demonstrate the phase transition
changed the active set, and pass the batching parity and provider-count
assertions. All existing suites stay green.

**Pitfalls:**
- Scope creep is the failure mode: no LM/GN work (M2b), no `Phase` class
  (M2c), no projection/chamfer/SDF residuals in `residuals/` (M4) — the
  slice's residual and terms are *test-local consumer code*, exactly as a
  downstream author would write them.
- Don't "fix" the slice by special-casing the library to it. If the API
  can't express an ingredient, the API changes.
- The consumers stay untouched; migration is M4.

### T2a.9 — The custom-residual author guide  [S–M]

**Goal / done-when:** a written contract a downstream author can follow
without reading BR internals — the codex complaint was precisely that the
sketch "is not enough for a consumer author to implement one"
(`codex_plan_review.md §B2.3`). Required sections:

1. **Residual shape** — `(B..., dim)`, dim static per evaluation, batch
   axes never folded into dim.
2. **Semantic grouping for robust kernels** — what "one residual item"
   means to a kernel: today rows are reweighted per scalar
   (`optim audit §2.6`), Ceres applies loss per block; state the grouping
   the block world guarantees and how to structure `dim` so a kernel sees
   meaningful groups (e.g. one 3-vector per point).
3. **Optional analytic blocks** — `jacobian_blocks(ctx)` keyed by var
   name, reduced-mask column convention, and the rule that a raising
   analytic Jacobian is an error, never a silent fallback (audit §2.14).
4. **Provider requests** — declaring `reads`, consuming provider outputs,
   the detached-output convention (NN passes), and what may/may not be
   cached (nothing across evaluations).
5. **Per-element failure signaling** — how a residual reports per-element
   invalidity (NaN rows vs a validity mask — align with the 03 §9
   per-element failure-semantics bullet and record the decision), instead
   of raising and killing the batch.

**Implementation plan:** write it as a Diátaxis how-to guide (invoke
`sphinx-docs` + `diataxis-docs` before writing), suggested
`docs/guides/custom_residuals.md`, but keep it out of the published toctree
until the freeze commit. Its worked example must be the *slice's* custom
residual, kept in sync by construction (T2a.8 step 5). The guide's code is
executable — extract it into the slice test or a doctest so CI (M1
item 10) runs it.

**What to test:** the guide's example code runs verbatim (doctest or
imported by the slice test). A checklist item in T2a.8 confirms the slice
residual was written by following the guide, with the friction log as
evidence.

### T2a.10 — Differentiation-contract decisions (decide now, implement M6)  [M — parallel]

**Goal / done-when:** the differentiation-contract questions that
constrain state shape are **decided and written down** in this milestone;
the implementation stays M6 (roadmap M6 item 5 consumes exactly this
contract: "Implicit-diff `solve()` per the contract decided in M2a").
Required decisions (03 §9's differentiation-contract bullet + 03 §4's
implicit-differentiation paragraph):

1. **Which gradients are guaranteed** — w.r.t. optimized variables,
   model values, residual parameters (targets, weights, kernel scales),
   solver hyperparameters — and which are explicitly *not*.
2. **To what order** — first order guaranteed where; second order
   (`create_graph`) supported through which paths (the torch lane is the
   only double-backward path — 03 §2.1).
3. **Unrolled vs implicit** — default behavior of a future `run`
   (detached outputs? unrolled graph opt-in? implicit-diff opt-in?), and
   which the evaluation protocol must not preclude. jaxopt's `root_vjp`
   is the reference (`references/design/jaxopt.md`); the "~100 lines"
   estimate is formally retracted (codex §D) — manifolds, active bounds,
   robust kernels, singular systems, and the optimized-vs-external
   parameter split are the real contract.
4. **Behavior on non-converged solves** — gradients through the last
   iterate with a status flag, or refusal — decide and write it; a wrong
   silent gradient is the failure mode to design against.

**Why now:** these decisions constrain M2a/M2b state shape — solver state
must be a plain tensor pytree (no Python floats), the optimality system's
ingredients (final J blocks, active-bound masks, robust weights) must be
recoverable from state, and `Problem` must be able to enumerate external
(non-optimized) tensor parameters. If M2a freezes an API where residual
parameters are unreachable Python attributes, M6's implicit diff is dead
on arrival.

**Implementation plan:** extend the M1 engineering-contract document
(wherever `m1_two_lane_seam_and_hygiene.md` item 5 put it — likely under
`docs/conventions/`) with a "Differentiation contract" section covering
the four decisions, each with a one-paragraph rationale and the
consequences for state shape spelled out as requirements on M2b
(`m2b_batched_second_order_solvers.md` must cite them). Where genuinely
open, present options with a recommendation — then **stop for owner
sign-off**; the sign-off is a freeze precondition.

**What to test:** nothing executable beyond what T2a.2/T2a.6 already
cover (`create_graph` paths); the deliverable is the written contract plus
a cross-reference check — grep that m2b's instruction file and the new
code's docstrings don't contradict it.

## Milestone acceptance checklist

- [x] M0 θ=0 gradcheck one-liner passes; `AUTODIFF` grep is clean; M1
      checklist passed (or owner-approved early start of T2a.1/2/10 only).
- [x] `Manifold` protocol + `Euclidean`/`SO3`/`SE3`/`RobotConfig(model)`
      implemented and tested; right-perturbation convention verified
      against `spherical.py:52-62`.
- [x] State-space bounds via feasible retraction; nq-shaped for
      `RobotConfig`; `SO3`/`SE3`+bounds raises with the specified message;
      no tangent-space box anywhere in `Manifold`/`Bounds`.
- [x] Tangent-space autograd helper: fp32 gradcheck at δ=0 **at** singular
      points (θ=0, identical quaternions, near π); SMPL-like rest-pose
      gradient NaN-free.
- [x] `VarSpec`/`Values`/`Problem` landed unexported as a draft, then was
      frozen after the slice; residuals declare `reads`; `jacobian_blocks`
      is keyed `(residual_name, var_name)`; absent block = structurally zero.
- [x] Masks eliminate coordinates (reduced columns, nonsingular normal
      system, expand/gather roundtrip); `scale` validated and exposed for
      M2b.
- [x] Provider DAG: toposort, cycle rejection, lazy evaluation-local
      context; FK/NN-pass counted once per evaluation; nothing outlives
      the iteration except detached artifacts (leak test).
- [x] Scalar-term decision made **via the slice**, evidence written up,
      owner reviewed; GN/LM rejection message (or documented fence) in
      place.
- [x] AD decision table written and implemented; jacfwd-clean (smoke
      test); `create_graph` support specified and tested; FD reachable
      only as explicit debug.
- [x] Dense assembly parity-tested; sparsity non-goal stated;
      `ResidualSpec` structure/time-coupling semantics preserved in design
      notes for M5 before any deletion; assembly-vs-old-path benchmark
      definition committed.
- [x] **The vertical slice runs end-to-end** with all seven ingredients
      demonstrably present (two blocks, shared NN provider, custom
      residual, scalar term, masks, batched evaluation parity, manual
      phase transition), as a permanent CPU test on synthetic data.
- [x] Custom-residual author guide exists with all five contract sections;
      the slice's residual followed it verbatim (friction log empty or
      resolved into guide/API fixes).
- [x] Differentiation contract written (four decisions), owner-approved,
      cross-referenced by m2b.
- [x] Only after all of the above: symbols exported, API-contract test
      updated, docs + repo `CLAUDE.md` updated in the same commit.
- [x] Full suite green (1,099 non-Warp passes plus 12 Warp/layout passes);
      pinocchio parity is untouched and green. Consumer repos remained
      read-only; compatibility/import migration is deferred by standing
      rule 1 on the dedicated redesign branch (details in the results).

## Out of scope

- **Batched/second-order solving** — per-element damping, `cholesky_ex`
  statuses, the real bounded algorithm, Madsen–Nielsen numerics,
  capture-ready `update` → `m2b_batched_second_order_solvers.md`. M2a only
  guarantees `scale`, masks-as-elimination, and pytree-able state
  *inputs* for it.
- **Matrix-free Adam, the `Phase` engine, `solve_ik` re-base, retiring
  `costs/` shims and `LeastSquaresProblem`** → `m2c_first_order_phases_tasks.md`.
  The slice's hand-rolled loop and manual phase switch are deliberately
  not library code.
- **Sparse assembly, banded solvers, Schur elimination** →
  `m5_sparse_trajectory_structure.md`. Dense is v1 by decision.
- **Implicit differentiation implementation** → M6
  (`m6_warp_fast_path_and_cuda_graphs.md` item 5). Only the written
  contract lands here (T2a.10).
- **Library vision residuals** (projection, chamfer, SDF trio,
  Geman-McClure kernel) and consumer migration →
  `m4_consumer_packs_and_migration.md`. The slice's residual/terms stay
  test-local.
- **Parametric/batched ModelValues breadth, mimic coordinate map,
  vectorized `integrate`/`difference`** → `m3_parametric_model_breadth.md`.
  `RobotConfig` wraps the per-joint loops as-is.
- **No Warp kernels, no `torch.compile` work** in this milestone; no
  changes to `residuals/` built-ins (their port to `reads`/blocks rides
  M2b/M2c).
- Do not delete anything with a live consumer import; do not modify the
  consumer repos.

## References

- `plan/04_roadmap.md` — the M2a paragraph (source of the verbatim
  done-when) and the standing rules.
- `plan/03_architecture.md §3` — the sketch reproduced above and the seven
  binding review constraints; `§4` — optimizer stack + the
  implicit-differentiation paragraph behind T2a.10; `§9` — the
  differentiation-contract and per-element failure-semantics bullets.
- `plan/01_assessment.md §2.2` — single-flat-variable blocker, the ~24
  call sites, dense-J Adam/LBFGS (verified today at
  `optim/optimizers/adam.py:79`, `lbfgs.py:81`); `§1.5` — FD-as-autodiff +
  the jacfwd dtype mismatch; `§1.1` — the θ=0 NaN this milestone's helper
  depends on.
- `plan/research/codex_plan_review.md §B2` — why the first draft was
  "unsound as specified"; the origin of every constraint above. (Its BVR
  line numbers have drifted: `optimize_person` is now at
  `optimizer.py:398-599`, phases at `stages.py:363-395`.)
- `plan/research/audit_optim_stack.md` — §2.1 (variable hardwiring), §2.6
  (kernel grouping), §2.7 (matrix-free machinery unused, `ResidualSpec`
  dead), §6 Q5 (FK-caching ownership).
- `plan/research/audit_consumer_gaps.md` — §3.1 (BVR engine = requirements
  spec), §4.1 (consumer impact of θ=0), G2/G3 recommendations.
- `references/design/pyroki.md` (jaxls `Var`/`VarValues`),
  `references/design/jaxlie.md` (`manifold.rplus`/`grad` pattern),
  `references/design/jaxopt.md` (state pattern, `root_vjp`),
  `references/design/optax.md` (first-order composition — M2c context).
- Consumer evidence (read-only):
  `BVR tools/optim.py` (the 188-line engine),
  `BVR tools/human_optim/{optimizer,prefit,losses,stages}.py`,
  `BHF tools/object_align/{sdf_fit,polish,refine}.py`,
  `BHF tools/geometry/nn_search.py`.
- Sibling instruction files: `m0_truth_and_correctness.md`,
  `m1_two_lane_seam_and_hygiene.md` (prerequisites);
  `m2b_batched_second_order_solvers.md` (consumes `scale`, masks, state
  pytree, differentiation contract); `m2c_first_order_phases_tasks.md`
  (consumes the whole draft as the phase-engine substrate);
  `m5_sparse_trajectory_structure.md` (consumes the `ResidualSpec` design
  notes); `m6_warp_fast_path_and_cuda_graphs.md` (consumes the
  differentiation contract).
