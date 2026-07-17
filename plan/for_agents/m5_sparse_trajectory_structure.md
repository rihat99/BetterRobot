# M5 — Sparse Trajectory Structure: Agent Execution Instructions

> **Implementation log (2026-07-17):** The structured CPU lane and named-block
> trajopt rebase are complete on `dev`. Focused correctness gates and an isolated
> T=50 dense/structured benchmark pass; the canonical T=50/125/250/500 scaling
> sweep remains explicitly unmeasured. See `m5_results.md`.

> Read `plan/for_agents/README.md` first. It carries the standing rules
> (deletion ordering, honesty rules, benchmark requirements, test commands).

> **Dating note.** This file was written 2026-07-17, against the pre-M0
> working tree. Every `file:line` below was verified on that tree. By the
> time M5 executes, M0–M2 (and possibly M3/M4) have rewritten most of the
> optimization stack — the references serve two purposes: (a) git-history
> archaeology for the structural information M1 deleted, and (b) the
> mathematical content (band structures, matrix-free transposes) that the
> old code got right and the new code must preserve. **Re-verify every
> claim against the tree you actually have before acting on it.**

## Mission

Dense per-(residual, variable) Jacobian assembly is the known scalability
wall for long-horizon trajectory optimization (`plan/04_roadmap.md` M5;
`plan/research/codex_plan_review.md` §B2.5, §C4). Concretely: the
SMPL-scale model (`make_smpl_like_model()`: nq=99, **nv=75** — free-flyer
root + 23 spherical joints; verified by probe 2026-07-17) at T=500 knots
has a tangent space of T·nv = 37,500. A dense `JᵀJ` is 37,500² fp32
≈ **5.6 GB**; a single dense acceleration-residual Jacobian
`(nv·(T−2), T·nv)` is another ≈ 5.6 GB; dense Cholesky is ~N³/3 ≈ 1.8e13
flops per LM iteration. Yet the true structure is banded: a 3-point
smoothness residual couples knots at offsets {−1, 0, +1}, so `JᵀJ` is
block-pentadiagonal with 75×75 blocks — ~56 MB and O(T) factorization
work. M2a's variable-block Problem was deliberately designed so this
structure can be exploited *without another redesign* (03 §3, key
semantics: "The block structure is what later enables sparse trajectory
solvers"). M5 cashes that cheque: a reviewed design first (Phase A), then
symbolic block sparsity + matrix-free operators + a banded solver behind
the M2b linear-solver contract (Phase B), proven by a committed benchmark
showing memory AND time sublinear in the dense baseline at T=500
(Phase C).

## Prerequisites

- **M2a landed** (`m2a_variable_blocks_and_slice.md`): `VarSpec` /
  `Values` / `Problem` with the `reads` dependency declaration, the
  provider DAG, manifolds (`Euclidean/SO3/SE3/RobotConfig`), and
  `jacobian_blocks(values) -> dict[(residual_name, var_name), Tensor]`
  with absent = structurally zero (03 §3). Check: the M2a acceptance
  checklist passes; the vertical-slice test is green; grep for the
  frozen `Problem`/`VarSpec` symbols and read the custom-residual author
  guide M2a produced — M5 *extends* that contract, it does not replace it.
- **M2b landed** (`m2b_batched_second_order_solvers.md`): batched LM/GN on
  `init_state/update/run`, per-element damping/statuses, and the linear
  solver contract **`solve(matvec_or_matrix, b, ridge)`** with damping
  orthogonal to the solver (03 §4 "Linear solvers"). That contract is the
  seam every M5 solver plugs into. Check: the M2b solver-quality tests
  pass; confirm the actual signature in the tree (03's sketch shows
  direction, not the final API).
- **M2c landed or explicitly deferred** (`m2c_first_order_phases_tasks.md`):
  `solve_trajopt` re-based on the new Problem, or temporarily dropped
  (the roadmap allows either). If it was dropped, T5.7 re-instates the
  trajectory task routing as part of this milestone — budget for it.
- **NOT required:** M3 (parametric breadth), M4 (consumer packs), M6
  (GPU). The benchmark model builder `make_smpl_like_model`
  (`src/better_robot/io/builders/smpl_like.py:142`) exists today,
  independent of M3. A GPU is not required for M5 acceptance (CUDA is
  broken on the dev box — standing rule 7); the committed benchmark is
  CPU, with the GPU variant deferred to M6.

## Sizing & parallelism

Roadmap label: **M–L** overall.

| Task | Phase | Effort | Depends on |
|---|---|---|---|
| T5.1 recover structural inputs | A | S | — |
| T5.2 design document + owner review | A | M | T5.1 |
| T5.3 sparsity declaration + structured assembly | B | M | T5.2 approved |
| T5.4 JVP/VJP linear operators | B | M | T5.2 approved (parallel to T5.5) |
| T5.5 banded/block-tridiagonal solver | B | M | T5.2 approved (parallel to T5.4) |
| T5.6 Schur elimination (optional, evidence-gated) | B | S–M | T5.5 |
| T5.7 residual declarations + task routing | B | M | T5.3, T5.4, T5.5 |
| T5.8 committed benchmark | C | S–M | definition written in T5.2; run after T5.7 |

Phase A is strictly sequential and ends at a **hard stop for owner
review** (standing rule 8). Within Phase B, T5.4 and T5.5 are
parallelizable once T5.3's declaration API is committed; T5.6 only
happens if the approved design says so.

---

## Tasks

### T5.1 — Recover the structural-sparsity inputs  [S]

Preserved legacy field semantics and the M2a/M5 boundary live in
`plan/design_notes/residual_sparsity.md`; treat that note as T5.1 input.

**Goal / done-when:** the executor has in hand (a) the banded/temporal
structure semantics of the deleted `ResidualSpec`, (b) the matrix-free
`apply_jac_transpose` math from the old trajectory residuals, and (c) the
consumer-scale evidence — collected into working notes that feed T5.2.

**Current state (pre-M0 tree; will have changed):**
- `ResidualSpec` lived at `src/better_robot/optim/jacobian_spec.py:30-55`:
  fields `dim`, `output_dim`, `tangent_dim`,
  `structure: Literal["dense","diagonal","block","banded"]`,
  `time_coupling: Literal["single","5-point","custom"]`,
  `affected_knots`, `affected_joints`, `affected_frames`, `dynamic_dim`.
  The assessment (`plan/01_assessment.md` §2.5) verified it had **zero
  consumers**; M1's de-bloat deletes it, with 03 §10 requiring "its
  banded/temporal structure information moves into the sparse-milestone
  design notes first".
- Only two `.spec` implementations ever existed:
  `residuals/temporal.py:118-126` (`TimeIndexedResidual`: `structure=
  "block"`, `time_coupling="single"`, `affected_knots=(t_idx,)`) and
  `residuals/collision.py:60-70` (a stub residual). **Plan-vs-code note:**
  the old `residuals/CLAUDE.md` table advertised "banded"/"tridiagonal"
  structure for `VelocityResidual`/`AccelerationResidual`, but neither
  class defined `.spec` — the banded information lived only in docstrings
  and in the analytic Jacobian layout itself.
- The mathematical content worth reviving, per file (pre-M0 lines):
  - `residuals/smoothness.py:67-81` / `:127-142` — velocity Jacobian
    blocks `[−I, 0, +I]·w/(2dt)` at offsets {−1,+1}; acceleration blocks
    `[+I, −2I, +I]·w/dt²` at offsets {−1,0,+1} (both built dense by a
    Python loop over T — the anti-pattern).
  - `residuals/smoothness.py:83-94` / `:144-161` — the O(T·nv)
    matrix-free `Jᵀr` as three aligned slice-accumulations. This is the
    VJP the new linear operators generalize.
  - `residuals/temporal.py:93-95` — single-knot column scatter;
    `:97-115` — its sparse `Jᵀ vec`.
  - `residuals/regularization.py:126-154` — block-diagonal
    reference-trajectory Jacobian with per-frame scaling.
  - `residuals/contact.py:94-134` — block-bidiagonal (offsets {0,+1})
    per contact frame; note its `float(w_pair[t, k])` per block was a
    host sync per pair per step (audit `audit_optim_stack.md` §2.10).
- Recovery route if the design notes are missing or thin:
  `git log --follow --oneline -- src/better_robot/optim/jacobian_spec.py`
  (introduced in `ae6a422` "trajopt: … sparse J^T r", extended in
  `3b39197`); same for `residuals/smoothness.py`, `temporal.py`,
  `contact.py`.
- Consumer-scale evidence: `plan/01_assessment.md:236` (trajectory scale
  is "T×211 variables" for the BVR problem), `plan/research/
  audit_optim_stack.md` §2.10 (dense `(nv·(T−2), T·nv)` allocations per
  residual per LM iteration; for a 30-DOF humanoid at T=240 that is
  7k×7k per residual, several residuals, per iteration).

**Implementation plan:**
1. Locate the design notes M1 T-de-bloat wrote when deleting
   `ResidualSpec` (check `plan/` and `docs/`; cross-ref
   `m1_two_lane_seam_and_hygiene.md` for the path it chose). If absent,
   recover the semantics from git history at the commits above.
2. Read the M2a custom-residual author guide and the frozen `reads`
   contract — write down exactly what a residual can declare today.
3. Read `references/design/jaxopt.md` §5.4/§6 and
   `references/design/pyroki.md` §4 (details under T5.2 below).
4. Produce a short working-notes section (feeds T5.2; may live as an
   appendix of the T5.2 design doc — do not create a separate orphan
   document).

**What to test:** nothing executable; the deliverable is input to T5.2.

**Pitfalls / do-not-forget:**
- Do **not** resurrect the `ResidualSpec` dataclass verbatim. It carried
  deprecated aliases (`input_indices`, `is_diagonal`) and a stateful
  `dim` convention tied to the dead `ResidualState` world. Revive the
  *information* (which blocks, which time offsets, banded vs diagonal vs
  dense), expressed in M2a's vocabulary.
- The old docs were wrong in both directions: `docs/concepts/
  solver_stack.md:85-90` (pre-M0) described a `"matrix_free"` structure
  value that never existed in the Literal, and `:168-172` claimed
  `SparseCholesky` "is the default for trajopt" while
  `optim/solvers/sparse_cholesky.py:15` raised `NotImplementedError`.
  Treat every old doc claim as hostile input (standing README rule).

### T5.2 — The design document (Phase A deliverable; STOP for review)  [M]

**Goal / done-when:** a written design exists in `plan/` (suggested path:
`plan/design/m5_sparse_trajectory_structure_design.md`; `docs/` is
acceptable if the owner prefers user-facing placement), it answers every
question listed below, it contains the Phase C benchmark definition, and
**the owner has reviewed and approved it before any Phase B code is
written** (standing rule 8 — do not silently pick defaults and proceed).

**The design MUST answer, explicitly:**

1. **How sparsity is declared by residuals — extending M2a's `reads`
   contract without breaking it.** Today (post-M2a) a residual declares
   `reads: tuple[str, ...]` — which variable blocks it touches; absent
   (residual, variable) blocks are structurally zero (03 §3). M5 adds
   *intra-block* structure along a trajectory's time axis. The design
   must specify the declaration surface — e.g. an optional
   `structure(var_name) -> BlockStructure` hook or a declarative field
   carrying `(time_offsets: tuple[int, ...], block: "banded" | "diagonal"
   | "block_column" | "dense")` — such that: residuals that declare
   nothing keep working exactly as in M2a (dense within their blocks);
   solvers that ignore declarations keep working (dense assembly remains
   the default and the oracle); and the declaration is *checkable* (see
   the structure-vs-values parity test in T5.3). Every existing M2a
   residual must be untouched or trivially recompiled — **if you find the
   M2a Problem API cannot host this extension without redesign, STOP and
   report to the owner; do not fork the Problem API** (the block
   structure was designed in 03 §3 precisely to make M5 possible).
2. **What a trajectory variable *is*.** Two candidate encodings, both
   must be priced: (a) one `VarSpec` block with an explicit leading time
   axis (`shape=(T, nq)`, manifold applied per knot) — codex B2.5 warned
   this is "still one big dense q-block" unless the time axis is
   annotated; (b) per-knot variable instances, jaxls-style — pyroki
   builds `traj_vars = robot.joint_var_cls(jnp.arange(timesteps))` and
   lets the solver's analyze step build block-sparse structure from the
   (cost, var-id) incidence (`references/design/pyroki.md` §4), which
   codex B2.5 warned makes the public API and provider DAG unwieldy at
   T=500. Decide with a small prototype measurement of assembly overhead
   (Python-loop cost at T=500 for (b); annotation complexity for (a)),
   present both numbers to the owner.
3. **Symbolic block sparsity.** From the declarations, a symbolic
   analysis (run once per problem, cached on the Problem — never
   per-iteration) computes which (residual, variable, time-offset)
   blocks are structurally nonzero, and derives the sparsity pattern of
   `JᵀJ`: for 3-point smoothness residuals the Gauss-Newton Hessian is
   block-pentadiagonal (residual offsets {−1,0,+1} ⇒ Hessian couples
   knot pairs up to |Δt| ≤ 2). The design gives the general rule
   (Hessian offsets = differences of residual offsets) and the storage
   layout for a banded block matrix (`(T, 2w+1, nv, nv)` or
   lower-triangle-only variant).
4. **JVP/VJP linear operators, matrix-free GN/LM.** So GN/LM can run
   without materializing `J` at all: per-residual structured
   `apply_jac(δ)` (JVP) and `apply_jac_transpose(u)` (VJP) compose into
   problem-level operators `v ↦ Jv`, `u ↦ Jᵀu`, and
   `v ↦ (JᵀJ + ridge·diag)v`, plugged into the M2b
   `solve(matvec_or_matrix, b, ridge)` seam with a CG-on-normal-equations
   solver. Blueprint: jaxopt's matrix-free LM path
   (`references/design/jaxopt.md` §5.4 — jvp/vjp operators + any callable
   `solve(matvec, b, ridge=…)`; damping as a ridge-matvec wrapper, never
   built into the solver; `diag(JᵀJ)` for damping init via structured
   column norms or vmap over basis vectors; warm-starting CG with the
   previous step's delta, §6). The design must state where autograd
   supplies the VJP (residuals without hand-written structured ops) and
   where hand-written slice-accumulation ops are required (the T5.1
   recovered math), and how `create_graph`/double-backward interacts
   (cross-ref the differentiation contract from M1 item 5 / 03 §9).
5. **Banded / block-tridiagonal factorization for the temporal q-block.**
   A direct solver for banded block SPD systems: block-Cholesky
   (block-Thomas) sweep over T with dense `nv×nv` (and `w·nv × nv`)
   inner ops — O(T·w²·nv³) time, O(T·w·nv²) memory, both linear in T.
   Batched over B (all inner ops are `(B, …)` batched torch ops — the
   sequential-over-T loop is static and compile-friendly). The design
   states: pivoting/regularization policy (`ridge` added to diagonal
   blocks — same semantics as M2b's damping), failure signaling
   per batch element (composing with M2b's `cholesky_ex` info-mask
   policy), and whether this solver is CUDA-graph-capture-eligible
   (fixed shapes, no host syncs — see pitfalls) or documented as
   eager-only in v1 with the reason recorded.
6. **Optional Schur elimination for camera/nuisance blocks.** The
   bundle-adjustment arrow pattern: banded temporal q-block + a small
   dense shared block (camera extrinsics, betas) coupled to many knots.
   Eliminate the banded block with the T5.5 factorization, solve the
   small dense Schur complement, back-substitute. The design specifies
   the detection rule (a shared variable read by residuals that also
   read the trajectory block), the cost model for when it pays, and —
   per standing rule 5 — names the second concrete caller (BVR's
   `q + camera-extrinsics` problem from the M2c acceptance is the
   natural one). If no second in-tree caller exists at execution time,
   the design may recommend deferring T5.6; that recommendation goes to
   the owner, not into silent scope-cutting.
7. **How it composes with per-element batching from M2b.** M2b solves B
   independent problems with per-element damping `mu[..., None, None]`
   and per-element accept/reject/statuses. The structured path must
   preserve exactly that: symbolic structure is batch-independent
   (computed once), numeric blocks carry `(B, …)` leading axes, the
   banded factorization and CG are batched, per-element `ridge` is a
   `(B,)` tensor, statuses stay per element. State explicitly that
   batched-vs-sequential parity follows standing rule 3 (stated
   tolerances + per-element statuses, not exact equality).
8. **What stays dense.** At minimum: the `nv×nv` within-knot blocks
   (75×75 dense is correct and fast); the small shared/nuisance blocks;
   IK-sized problems (T=1 — `solve_ik` never routes through the sparse
   path); any residual without a structure declaration (it contributes a
   dense block-column, and the assembly must handle mixed
   declared/undeclared stacks); and the dense assembly path itself,
   which is retained unmodified as the parity oracle.
9. **The B-spline interaction (from M2c).** A spline basis couples
   knots: with `expand(z) = B @ z` (cubic open-clamped basis, support 4
   control points per sample — pre-M0 code at
   `tasks/parameterization.py:69-118`), a residual with time offsets O
   in *knot* space touches, in *control-point* space, the union of
   supports of the sampled rows — band width grows by the basis support
   (Minkowski-sum rule: control-point offsets = O ⊕ support(B)). For
   C ≪ T the control-point system can be effectively dense — the design
   must give the band-width formula, a fall-back-to-dense threshold
   (e.g. when predicted band width × block size exceeds the dense cost),
   and how the chain rule `J_z = J_q · (∂q/∂z)` is applied in structured
   form (the old code materialized `torch.kron(B, I_nq)` —
   `tasks/trajopt.py:201`, `(T·nq, C·nq)` dense — which at T=500, C=125
   is 49,500×12,375; the design should replace it with a per-knot-window
   contraction).
10. **Manifold/tangent bookkeeping inside the banded system.** Each knot
    is a point on the RobotConfig manifold: bands are `nv×nv` in
    *tangent* space while states are nq-dim (nq=99 ≠ nv=75 on the SMPL
    model); retraction is per knot; `difference`-based residuals'
    Jacobians use the identity-right-Jacobian approximation (documented
    in the old smoothness docstrings — keep the approximation and its
    stated validity regime, or upgrade to exact right Jacobians and say
    why); and quaternion double-cover continuity along the trajectory
    needs a stated policy (hemisphere alignment between consecutive
    knots — codex review §C5) so `difference` doesn't produce 2π jumps
    that poison smoothness terms.
11. **The Phase C benchmark definition** (standing rule 4) — in full:
    hardware (named dev box, CPU, thread count pinned), dtype (fp32),
    model (`make_smpl_like_model()`, nq=99/nv=75), problem (trajectory
    IK: pose targets via time-indexed residuals at a fixed knot subset,
    velocity + acceleration smoothness, rest, limits — exact weights and
    seeds committed), T sweep {50, 125, 250, 500}, fixed iteration
    budget, warmup runs, statistic (median of ≥5 runs + IQR), memory
    metric (peak RSS / `torch.profiler` on CPU;
    `torch.cuda.max_memory_allocated` reserved for the M6 GPU variant),
    and the sublinearity acceptance rule (T5.8).

**Deliverable format:** one markdown design doc; every decision either
carries evidence (a probe number, a git reference, a consumer file:line)
or is explicitly listed in a "decisions needing owner input" section at
the top. End Phase A by presenting that doc to the owner. **Do not start
T5.3 until it is approved.**

**What to test:** prototype probes referenced by the doc must be
reproducible (`uv run python` one-liners or small scripts checked in
next to the design doc), but no library code changes in this task.

**Pitfalls / do-not-forget:**
- API sketches in 03 §3/§4 show direction, not final signatures — the
  M2a/M2b code as landed is the authority. Quote the real signatures in
  the design doc.
- Sparsity remains **opt-in structure, not a new Problem**: 03 §3
  constraint 5 held sparsity out of v1 assembly deliberately; M5 adds a
  second assembly/solve path behind the same Problem, it does not
  version the API.
- No new abstraction without a second concrete caller (standing rule 5):
  the declaration API's callers are the banded direct solver AND the
  matrix-free operators; the declaring residuals must number ≥ 2 real
  ones (velocity, acceleration, time-indexed, reference-trajectory).

### T5.3 — Sparsity declaration + structured assembly  [M]

**Goal / done-when:** residuals can declare intra-block temporal
structure per the approved design; the Problem's symbolic analysis
builds the block-sparsity pattern once; a structured assembly produces
the banded `JᵀJ` blocks and `Jᵀr` directly (never materializing a
`(dim, T·nv)` dense Jacobian); dense assembly is untouched and remains
the default for undeclared problems.

**Implementation plan:**
1. Add the declaration surface exactly as designed (new module under
   `src/better_robot/optim/` — invoke the `file-naming` skill; a name
   like `structure_blocks.py` / `sparsity.py` per its convention).
2. Implement the symbolic pass: input = residual declarations + variable
   blocks; output = a frozen structure object (nonzero (residual, var,
   offset) triples, Hessian band width, block layout) cached on the
   Problem. Must be pure Python over static shapes — no tensor values
   involved (torch.compile-friendly; no data-dependent structure).
   Residuals with `dynamic_dim`-like behavior (the old collision case)
   are rejected from the structured path with an honest error naming
   the dense fallback.
3. Implement structured assembly: per declared residual, evaluate its
   Jacobian *blocks* (offset-indexed `(B, T', nv_out, nv)` tensors —
   vectorized over T, replacing the old Python-loop-over-T builders),
   scatter-accumulate into the banded `JᵀJ` storage and `Jᵀr`. Mixed
   stacks (declared + undeclared residuals on the same variable) either
   assemble the undeclared part densely into the band-plus-dense form
   the design specified, or fall back to fully dense with a logged
   reason — whichever the design chose.
4. Wire robust kernels and per-item weights identically to the dense
   path (IRLS reweighting happens on residual blocks before
   accumulation — semantics per the M2b kernel-grouping decision; do not
   invent a second weighting convention).

**What to test** (`tests/optim/test_sparse_structure.py`, fp32 per the
workspace `write-tests` rules, stated tolerances):
- Symbolic: for a velocity+acceleration+rest+time-indexed stack at
  T ∈ {3, 5, 17}, the computed Hessian band width and nonzero-block set
  match hand-derived expectations (pentadiagonal blocks; single-knot
  columns for time-indexed).
- **Structure-vs-values parity (the load-bearing test):** for every
  declaring residual, densify the structured blocks and compare against
  the dense-path Jacobian at random configurations (free-flyer SMPL-like
  model AND a fixed-base chain) — allclose, fp32 rtol ≤ 1e-5. Also
  assert the *complement* is exactly zero in the dense Jacobian (the
  declaration must not under-claim coupling — an over-narrow declaration
  is a silent wrong-answer bug).
- Assembled banded `JᵀJ` and `Jᵀr` vs dense `Jᵀ J` / `Jᵀ r` at small T:
  allclose fp32 rtol ≤ 1e-5, including batched `(B=4, T, nq)` inputs
  and a mixed declared/undeclared stack.
- The honest-error path: a dynamic-dim residual requesting the
  structured path raises with the drafted message, e.g.
  `ValueError: residual '<name>' has a dynamic output dimension and
  cannot use structured assembly; it will be assembled densely — see
  plan/design/m5_sparse_trajectory_structure_design.md`.
- Existing M2a/M2b test suites stay green untouched.

**Pitfalls / do-not-forget:**
- The old builders' Python-loop-over-T with `torch.eye` per call and
  `float()` per block (contact) are the exact anti-patterns the hot-path
  lint (M1 item 7) now catches — the new assembly must be loop-free over
  T in tensor ops (a static loop over the ≤ 2w+1 offsets is fine).
- Structure is computed from *declarations*, never from probing tensor
  values (no "detect zeros numerically" — that breaks compile and lies
  near coincidental zeros).

### T5.4 — JVP/VJP linear operators (matrix-free GN/LM)  [M]

**Goal / done-when:** GN/LM can run matrix-free: problem-level `Jv`,
`Jᵀu`, and damped normal-equation matvec operators exist, composed from
per-residual structured ops (autograd-derived where no hand-written op
exists); a CG solver satisfying the M2b `solve(matvec_or_matrix, b,
ridge)` contract consumes them; `diag(JᵀJ)` is available for
Madsen–Nielsen damping init without materializing `J`.

**Implementation plan:**
1. Per-residual structured `apply_jac` / `apply_jac_transpose`
   equivalents in the M2a residual vocabulary (port the T5.1 recovered
   slice-accumulation math for velocity/acceleration/time-indexed/
   reference-trajectory); autograd `vjp`/`jvp` fallback for residuals
   without them (this is real autodiff — M0 deleted the lying enum;
   don't reintroduce FD here).
2. Problem-level composition respecting variable blocks and masks;
   damping enters as a ridge wrapper around the matvec (jaxopt
   `_make_ridge_matvec` pattern — `references/design/jaxopt.md` §6),
   with per-element `(B,)` ridge.
3. A batched CG implementation behind the M2b contract (fixed max
   iterations + tolerance-based masking per element — no Python
   branching on tensor values; warm-start `init` from the previous LM
   step where the solver state carries it). Note the M0-era `CG` stub
   class was deleted; this is a fresh implementation against the new
   contract, not a resurrection.
4. Preconditioning: at minimum block-Jacobi (invert the `nv×nv`
   diagonal blocks — already computed by T5.3). Anything fancier needs
   benchmark evidence first.

**What to test** (`tests/optim/test_linear_operators.py`):
- Operator parity: `Jv` and `Jᵀu` vs dense-Jacobian products at small T,
  batched and unbatched, fp32 rtol ≤ 1e-5; `diag(JᵀJ)` vs dense diag.
- Adjoint consistency: `⟨Jv, u⟩ == ⟨v, Jᵀu⟩` to fp32 tolerance for
  random v, u — catches sign/offset bugs the parity test can miss.
- CG solves a small banded SPD system to the same solution as the dense
  Cholesky path (rtol ≤ 1e-4 fp32), per batch element, with per-element
  ridge values differing across the batch.
- An end-to-end matrix-free LM trajectory solve at small T matches the
  dense-assembly LM solve within stated tolerances and per-element
  statuses (standing rule 3).

**Pitfalls / do-not-forget:**
- `update` must stay sync-free (M2b capture-readiness): CG's stopping
  logic inside `update` must be mask-based with a fixed iteration cap,
  not `bool(residual_norm < tol)` host branches.
- Autograd-derived VJPs build graphs per evaluation — respect M2a's
  evaluation-local context rule (03 §3 constraint 2); no persistent
  cached graphs.
- Double backward: if the differentiation contract (03 §9) promises
  gradcheckable solves, the operator path must support `create_graph`;
  test it or document the exclusion honestly in the same change.

### T5.5 — Banded / block-tridiagonal direct solver  [M]

**Goal / done-when:** a batched block-banded Cholesky (block-Thomas)
factor/solve exists behind the M2b `solve` contract, consuming T5.3's
banded storage; it matches the dense Cholesky solution at small T and is
O(T) in time and memory (asserted by the T5.8 benchmark, sanity-checked
by a quick scaling test here).

**Implementation plan:**
1. Implement factor + solve for block-banded SPD matrices in the
   design's storage layout: sequential loop over T (static, batched
   inner ops `(B, nv, nv)`), lower-triangle blocks stored, `ridge`
   (`(B,)`) added to diagonal blocks before factorization.
2. Per-element failure handling composed with M2b policy: use
   `torch.linalg.cholesky_ex` on the diagonal blocks and propagate an
   info mask per batch element; a failed element's output is blended
   out exactly the way M2b's dense fallback does (branch-free).
3. Register it as a selectable linear solver for structured problems
   per the design's routing rule (structured problems default to it;
   dense problems never see it).

**What to test** (`tests/optim/test_banded_solver.py`):
- Factor/solve parity vs `torch.linalg.cholesky` + `cholesky_solve` on
  densified banded systems: T ∈ {3, 8, 32}, nv ∈ {2, 7, 75}, band width
  w ∈ {1, 2}, batched `(B=4,)`, fp32 rtol ≤ 1e-4.
- SPD edge cases: near-singular diagonal block rescued by ridge; an
  indefinite element in a batch flags only that element (info mask) and
  leaves siblings' solutions bit-identical to their solo solves.
- A no-materialization guard: peak memory of factor+solve at T=256,
  nv=75 stays under a stated budget (e.g. < 5× the banded storage size)
  — a cheap tripwire against accidental densification.
- Scaling sanity: time(T=256)/time(T=64) < 6 (i.e. roughly linear, with
  slack for fixed overhead) — the real scaling claim lives in T5.8.

**Pitfalls / do-not-forget:**
- The sequential T loop is inherent to a direct banded solve; it is
  fine for torch.compile (static trip count) but means GPU latency at
  small B — do not "optimize" it into a parallel scan without benchmark
  evidence (that is M6-grade work; note it in the design doc's future
  section instead).
- fp32 conditioning: normal equations square the condition number. The
  ridge floor and the (documented) option to run the factorization in
  fp64 while keeping residual evaluation fp32 must follow the dtype
  policy from the M1 engineering contract — state what you implement.

### T5.6 — Schur elimination for shared/nuisance blocks  [S–M, evidence-gated]

**Goal / done-when:** *only if the approved design said build it now:*
an arrow-structured solve (banded temporal block + small dense shared
block) via Schur complement, matching the dense joint solve at small
sizes; demonstrated on the `q-trajectory + camera-extrinsics` toy
problem from M2c's acceptance.

**Implementation plan:** per the design: eliminate the banded block with
T5.5's factorization, form the dense Schur complement on the shared
block (small: 6–20 dims), solve, back-substitute. Reuse — do not
duplicate — T5.5's factor.

**What to test** (`tests/optim/test_schur.py`): parity vs the dense
joint solve, fp32 rtol ≤ 1e-4, batched; gradient flow to the shared
block intact (a small gradcheck through the solve if the contract
promises it); the toy problem converges to the same solution through
the structured and dense paths.

**Pitfalls / do-not-forget:** if the second in-tree caller is missing,
this task is a design-doc section plus a STOP-and-report, not code
(standing rule 5). Do not build it "because BA does".

### T5.7 — Residual declarations + trajectory task routing  [M]

**Goal / done-when:** the shipped smoothness/velocity/acceleration (and
time-indexed / reference-trajectory, if they survived M2 under those
names) residuals declare their banded structure; `solve_trajopt` (or its
M2c successor) routes declared problems through structured assembly +
the banded solver by default, with an explicit escape hatch to dense;
the dense path remains available and untouched for parity.

**Implementation plan:**
1. Add declarations to each temporal residual per T5.3's API — exact
   offsets from the analytic math ({−1,+1} velocity, {−1,0,+1}
   acceleration, {t} time-indexed, diagonal reference-trajectory,
   {0,+1} contact-consistency if it exists post-M2/M4).
2. Route the trajectory task: structured by default when every active
   residual declares structure and the parameterization is knot-based;
   B-spline problems apply the design's band-width formula and its
   dense-fallback threshold (T5.2 item 9). Log/record which path ran in
   the result object so benchmarks and tests can assert it.
3. If M2c dropped `solve_trajopt`, re-instate it as a thin preset over
   the M2 Problem here (its scope: knot + B-spline parameterizations,
   quaternion-safe, bounds honored — the M2c file's acceptance items
   apply; coordinate rather than duplicate).
4. Update docs the change touches in the same change: the solver-stack
   concepts page (whatever replaced `docs/concepts/solver_stack.md`
   §§7–8 claims about `SparseCholesky`/"block-Cholesky" — those claims
   must finally become true or stay deleted), `optim`/`residuals`
   CLAUDE.md files, and the roadmap/stub inventory.

**What to test** (`tests/tasks/test_trajopt_sparse_parity.py`):
- **Dense-as-oracle parity at small T (the milestone's key regression):**
  the same trajectory IK problem (SMPL-like model, T ∈ {5, 16}, fixed
  seeds) solved via forced-dense and via structured paths reaches the
  same solution within stated tolerances (fp32: q rtol ≤ 1e-3 or final
  cost within 1e-5 — per-element statuses equal), batched and unbatched.
- Free-flyer + spherical manifold correctness: the double-cover
  continuity policy from T5.2 item 10 has a directed test (a trajectory
  crossing the hemisphere boundary keeps smoothness residuals bounded).
- B-spline routing: a C ≪ T problem takes the path the threshold rule
  dictates (assert on the recorded path), and a knot-based T=16 spline
  parity test against dense.
- Bounds still honored through the structured path (the M2b bounded
  algorithm composing with banded solves — an interior-solution bounded
  trajectory converges; cross-check with the M2b bounds tests).
- All pre-existing suites green; the pinocchio-parity suite untouched.

**Pitfalls / do-not-forget:**
- Declarations must match the *implemented* Jacobian, including weight
  scaling and the identity-right-Jacobian approximation — the
  structure-vs-values parity test from T5.3 runs against each newly
  declaring residual as part of this task.
- Do not change residual math while adding declarations (surgical-change
  rule); any math fix you discover is its own reported item.
- BVR/BHF grep before touching any surviving public trajopt surface
  (standing rule 1).

### T5.8 — The committed benchmark (Phase C)  [S–M]

**Goal / done-when (the roadmap's M5 done-when, expanded):** trajectory
IK at T=500 on the SMPL-scale model **runs** through the structured path
(completes its iteration budget without OOM on the named CPU box);
measured memory AND time are **sublinear in the dense-assembly
baseline**; the benchmark definition, script, and numbers are committed
to the repo.

**Implementation plan:**
1. Implement the benchmark exactly as defined in the approved design
   (T5.2 item 11). Suggested placement: `tests/bench/bench_trajopt_sparse.py`
   following the existing `tests/bench/` conventions (session-scoped
   model fixtures, `-m bench --benchmark-only`); numbers in a committed
   JSON next to the existing baselines. Note the pre-M0
   `tests/bench/README.md` cited a CI job and a
   `docs/claude_plan/accepted/12_regression_and_benchmarks.md` that do
   not exist (verified 2026-07-17) — follow whatever M1's CI item
   actually built, and fix the README if it still lies.
2. Measurement protocol: for T ∈ {50, 125, 250, 500}, run both paths
   with a fixed LM iteration budget; record median wall time per
   iteration (≥5 runs after ≥2 warmups) and peak memory. **The dense
   baseline at T=500 may be unrunnable** (a single dense `JᵀJ` is
   ~5.6 GB; several residual Jacobians more): if it OOMs or exceeds a
   stated time cap, record the failure honestly as the baseline value
   ("dense: DNF at T=500, OOM at N GB") and evidence sublinearity from
   the T-sweep slopes instead.
3. Acceptance rule (write it into the committed definition, not the
   chat): log-log slope of time-vs-T and memory-vs-T for the structured
   path strictly below the dense path's slopes over the sweep, AND
   structured T=500 completes. Report raw numbers + slopes; no bare
   ratios (standing rule 4).
4. CPU numbers are the M5 deliverable (standing rule 7). Parameterize
   device so the M6 GPU runner can re-run it; leave the GPU row
   explicitly marked "pending M6", never extrapolated.

**What to test:** the benchmark script itself runs under
`uv run pytest tests/bench/ -m bench --benchmark-only` and a fast smoke
variant (T=50, 2 iterations) runs in the normal suite so the harness
can't rot.

**Pitfalls / do-not-forget:**
- Pin torch thread count (existing bench conventions) — CPU timing noise
  otherwise swamps the slope fit.
- Warmup must include the first `torch.compile` call if the structured
  path compiles anything (cold-start ~31 s was measured on the CPU
  probe; report cold-start separately, don't average it in).
- Commit the exact model-construction call (height/mass defaults,
  dtype, seed) — "SMPL-scale" must be reproducible, not a vibe.

---

## Milestone acceptance checklist

- [ ] Design doc exists in `plan/` (or `docs/`), answers all eleven
      questions of T5.2, and **was approved by the owner before Phase B
      code landed** (commit dates prove ordering).
- [ ] ResidualSpec's banded/temporal semantics are demonstrably the
      design's input (the doc cites the M1 design notes or the git
      archaeology).
- [ ] Sparsity declarations extend the M2a `reads` contract without
      breaking any M2a test; the M2a Problem API is unchanged (or a
      STOP-and-report happened instead).
- [ ] Structure-vs-values parity tests pass for every declaring residual
      (densified blocks match the dense Jacobian; complement exactly
      zero).
- [ ] Matrix-free `Jv`/`Jᵀu`/damped-normal operators pass parity +
      adjoint-consistency tests; CG runs behind the M2b
      `solve(matvec_or_matrix, b, ridge)` contract with per-element
      ridge.
- [ ] Batched block-banded Cholesky passes parity vs dense Cholesky
      across stated (T, nv, w, B) grid, with per-element `cholesky_ex`-
      style failure masks.
- [ ] Schur elimination: implemented-and-tested, or explicitly deferred
      by owner decision recorded in the design doc — no third state.
- [ ] Temporal residuals declare their structure; trajectory tasks route
      through the structured path by default with a recorded path flag
      and a dense escape hatch.
- [ ] Dense-as-oracle parity at small T passes (same solutions, stated
      tolerances, per-element statuses) — batched and unbatched,
      fixed-base and free-flyer.
- [ ] Trajectory IK at T=500 on `make_smpl_like_model()` completes via
      the structured path on the named CPU box.
- [ ] Committed benchmark (definition + script + JSON numbers) shows
      time AND memory sublinear vs the dense baseline over the T sweep;
      dense DNFs recorded honestly if they occur.
- [ ] `update` remains host-sync-free through the structured path (M2b
      capture-readiness checklist not regressed), or the design doc
      records the exact eager-only exclusion.
- [ ] Docs/CLAUDE.md updated in the same changes: no page claims a
      sparse solver that doesn't exist, and the new one is documented
      where the old false claims lived.
- [ ] Full test suite green: `uv run pytest tests/ -v` (pinocchio-parity
      suite untouched).

## Out of scope

- **GPU execution of any of this** — banded-kernel work, capture/replay
  of structured solves, GPU benchmark rows: M6
  (`m6_warp_fast_path_and_cuda_graphs.md`). M5's deliverable is CPU.
- **Warp kernels for residuals or assembly** — M6, boundary table 03
  §2.2.
- **Implicit differentiation through the solve** — contract decided in
  M2a, implementation M6 (03 §4); M5 must merely not preclude it.
- **General sparse-matrix support** (arbitrary CSR/COO `torch.sparse`
  solvers, fill-reducing orderings, supernodal factorizations): M5 ships
  *structured* (banded/arrow) solvers only. Resurrecting a generic
  `SparseCholesky` class is an anti-goal — that ornament was deleted for
  cause (M0/audit §2.8).
- **DDP/iLQR-style trajectory solvers** — the `dynamics/action/`
  skeleton was parked in a branch in M1; do not revive it here.
- **Collision residuals in the structured path** — collision lands (or
  dies) in M4; its dynamic-dim shape is exactly what the honest error in
  T5.3 exists for.
- **Redesigning the M2a Problem/VarSpec API** — hard anti-goal; STOP and
  report instead (see T5.2 item 1).
- **Compacting converged batch elements out of the batch** — known M2b
  non-goal, stays a non-goal here.

## References

- `plan/04_roadmap.md` (M5 paragraph) — the done-when this file expands.
- `plan/03_architecture.md` §3 (variable blocks; constraint 5 = sparsity
  door held open), §4 (optimizer stack + `solve(matvec_or_matrix, b,
  ridge)` contract), §10 (ResidualSpec deletion → design-notes rule).
- `plan/01_assessment.md` §2.2 (dense Jacobians at trajectory scale,
  T×211 variables), §2.5 (ResidualSpec + jacobian_blocks had no
  consumer).
- `plan/research/audit_optim_stack.md` §2.7, §2.10–2.11 (matrix-free
  machinery unused; dense O(T²) trajectory Jacobians; B-spline
  breakages), §6 Q1 (per-problem batching vs one-big-problem — M5 serves
  the latter).
- `plan/research/codex_plan_review.md` §B2.5, §B6, §C4 — why M5 is its
  own milestone; the "block structure ≠ sparsity" argument this design
  answers.
- `plan/research/audit_compute_pass_inventory.md` rows on
  Velocity/Acceleration (line ~61) — the dense-alloc evidence.
- `references/design/jaxopt.md` §5.4 (materialized vs matrix-free LM),
  §6 (solve-as-calling-convention, ridge wrapper, normal-CG, warm
  start), §7 (implicit diff — the seam to not preclude).
- `references/design/pyroki.md` §4 (jaxls: per-timestep vars +
  analyze-then-solve symbolic structure; sparse QR backend; 5-point
  temporal residuals).
- Sibling instruction files: `m2a_variable_blocks_and_slice.md` (the
  `reads` contract M5 extends), `m2b_batched_second_order_solvers.md`
  (solver state pattern, linear-solver contract, capture-readiness
  checklist), `m2c_first_order_phases_tasks.md` (trajopt/B-spline
  rebase M5 routes through), `m1_two_lane_seam_and_hygiene.md` (the
  de-bloat that produced the ResidualSpec design notes),
  `m6_warp_fast_path_and_cuda_graphs.md` (GPU follow-ups).
- Git archaeology for deleted structure code: commits `ae6a422`
  (trajopt + sparse `Jᵀr`), `3b39197` (P0–P13 implementation) —
  `git log --follow -- src/better_robot/optim/jacobian_spec.py
  src/better_robot/residuals/smoothness.py`.
- Consumer repos (read-only): BVR `tools/human_optim/` (camera +
  trajectory nuisance-block shapes for Schur), BHF
  `scripts/motion/optimize_motion.py` (live imports constraining
  deletions — standing rule 1).
