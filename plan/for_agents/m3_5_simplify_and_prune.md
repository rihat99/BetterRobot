> **Execution log — 2026-07-17, complete on `dev`.** T1–T12, T14/T15,
> mimic×optim coverage, and truth fixes landed; B-spline/trajopt and
> Model/ModelValues dedup remain deferred exactly as ordered. Close gate:
> 1,362 passed, 1 skipped, 3 deselected; docs and offline lock checks pass.
> Deviations: T2 had two test callers, T6 had two extra stale monkeypatches,
> and T10 moved active autograd ownership to a wrapper so its VJP could reuse
> the real raw FK seam. No external repository was accessed.

# M3.5 — Simplify and Prune (cleanup milestone, runs before M4)

**Status:** work order, authored 2026-07-17 after a four-reviewer audit of the
committed M0–M2c work and the in-flight M3 tree. Not yet started. **Owner
resolved the three deferral gates 2026-07-17:** `solve_trajopt` and
`BSplineTrajectory` stay until M5, and the `Model`/`ModelValues` state dedup
goes with the M4 migration (see §"Deferred to later milestones"). M3.5 is
therefore purely the do-now safe cleanup: Tiers 1–3, 5, and 6.

**Why this milestone exists.** The owner suspected the redesign added too much
code and overcomplicated things. A read-only review (four Opus agents, disjoint
scopes, plus direct verification of the load-bearing findings) reached a
nuanced verdict that shapes this milestone:

- **No correctness bugs were found.** All four scopes traced their math against
  the code; the full suite is green (1,371 passed, 3 deselected, 110 s) on the
  current tree including the uncommitted M3 work. The pinocchio-parity and
  contract suites are green. **The agent is going in the right direction and is
  not making correctness mistakes.** M3.5 is therefore pure cleanup, run
  against a known-green net — not a rescue.
- **Most of the size is deliberate, not accidental.** The big files
  (`problem.py` 1041, `solver_lm.py` 953, `build_model.py` 956) are dominated
  by plan-mandated machinery (the frozen public/prevalidated shadow API,
  projected active-set LM, robust groups, build-time mimic policy), not by
  copy-paste bloat. "Rewrite it to be half the size" is **not** the goal.
- **The genuinely cleanable-now surface is ~350–600 LOC** of true dead code and
  mechanical redundancy, listed below as Tiers 1–3. Deleting it is low risk.
- **The largest apparent bloat — the ~1,400-LOC legacy optimizer stack — is
  NOT cleanable in M3.5.** It is not new overcomplication; it is the *old*
  stack kept alive by exactly one caller (`solve_trajopt`), and its deletion is
  gated on the M5 trajopt rewrite and the M4 consumer migration. It is
  documented in §"Deferred to later milestones" so the executor does not touch
  it, and so the owner sees the real sequencing.

**Placement — M3.5, before M4.** Cleaning before the M4 consumer migration is
deliberate: migrating BHF/BVR onto surfaces that then churn is double work. The
one caveat is that the legacy-stack deletion travels with M4/M5 regardless, so
M3.5 is scoped to the surfaces that are safe to change *before* consumers move.

## Standing rules (inherit README §"Standing rules" + these)

1. **Green is the safety net; keep it green after every task.**
   `UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q -m "not bench and not cuda"`
   is 1,371 passed today. Every task ends with this suite green. The
   pinocchio-parity suite (`tests/test_pinocchio/`) and contract suite
   (`tests/contract/`) must never go red as a side effect.
2. **Surgical.** Each task is an independent, revertible commit. Do not
   "improve" adjacent code. Do not refactor the frozen M2a public API or the
   named-block solver math — those were reviewed and are correct.
3. **Every removed consumer-facing symbol goes in a new
   `docs/reference/m3_removed_symbols.md` ledger** (there is no successor to
   `m1_removed_symbols.md` yet — create one), so M4's migration table stays
   complete. Private helpers (leading `_`, or names not in any `__all__`) do
   not need a ledger row.
4. **Same-milestone honesty (README rule 2):** when a task makes a doc / comment
   / CLAUDE.md claim false or true-again, fix it in the same commit. Tier 6
   collects the drift already found; new drift you create is yours to fix.
5. **Do not weaken a contract to make a deletion easier.** Some contract tests
   pin symbols that Tiers 1–4 remove (listed per task); update the *assertion's
   expected set*, never delete the check or loosen the AST rule.

---

## Tier 1 — Dead code deletion (all S; zero-to-very-low risk; ~230 LOC)

Verified dead by caller-graph grep. Do these first; they are independent.

| # | Task | Where | LOC | Done when |
|---|------|-------|----:|-----------|
| T1 | Delete `_make_damping_strategy` + the now-unused `Adaptive`/`Constant` imports. `solve_ik` selects damping via `fixed_damping = cfg.damping == "constant"` (`ik.py:388,408`); the factory is never called. **Verified:** only caller is `tests/optim/test_config_wiring.py`. | `tasks/ik.py:34-35,167-172` | ~8 | factory gone; `test_config_wiring.py` damping cases deleted or repointed at the bool selection; suite green |
| T2 | Delete the legacy `optim.solve()` wrapper + its `"solve"` `__all__` entry. **Verified:** zero callers in `src/`, `tests/`, `examples/`; only self-references remain. | `optim/__init__.py:45-80,89` | ~36 | `br.solve`/`optim.solve` grep clean; ledger row added (public-ish symbol); `test_submodule_public_imports` expected-set updated if it names `solve` |
| T3 | Delete `ResidualSpec` and `optim/jacobian_spec.py`, plus the two `.spec` properties that produce it and nothing consumes. **Verified:** no solver or test reads `.spec`. Its sparse/temporal semantics already live in the M5 design note (`plan/design_notes/residual_sparsity.md`). | `optim/jacobian_spec.py` (whole), `residuals/collision.py:60-68`, `residuals/temporal.py:107-115`, exports in `optim/__init__.py:39,88` | ~71 | `ResidualSpec`/`.spec` grep clean; ledger row (public `optim` symbol); suite green |
| T4 | Delete the dead Warp backward scaffolding: the `_warp_fk_backward` custom-op + its fake registration (explicitly "not the active path" per `m1_results.md` dev.3, zero callers) and `require_warp()` (exported, never called, only re-runs `wp.init()`). This is M6 groundwork shipped early — a "no abstraction without a second in-tree caller" violation. The active VJP recompute path and all Warp parity/gradcheck tests are untouched. | `kinematics/_warp_bridge.py:32-35,187-247,448` | ~65 | Warp CPU parity + gradcheck suite still green; `require_warp` grep clean; bridge module docstring updated |
| T5 | Remove the orphaned CRBA buffer: `composite_storage` is allocated then every `Y_c[i]` view is overwritten by `I_i.expand(...).contiguous()` before any read. **Verified.** `centroidal.py:95` does the same job in one line. | `dynamics/crba.py:64-65` | ~2 + a wasted `(*batch,njoints,6,6)` alloc | `Y_c` built directly from `spatial_inertias`; CRBA + centroidal parity green |
| T6 | Delete three zero-caller members: `LMState.iter_num` (`solver_lm.py:83-86`, grep-confirmed 0 refs), `IKResult.q_only()` (`ik.py:112-114`, trivial `return self.q`, 0 callers), `Problem.normal_matrix()` (`problem.py:1009-1018`, test-only per its own docstring — repoint `test_problem_blocks.py:215` at `J.mT@J`). | as noted | ~17 | members gone; the one affected test computes the value inline; suite green |

Tier 1 total: ~200 src LOC + a handful of test edits.

---

## Tier 2 — Mechanical redundancy collapse (S–M; low risk; efficiency + elegance)

| # | Task | Where | Win | Done when |
|---|------|-------|-----|-----------|
| T7 | **Collapse the repeated input revalidation on the FK/dynamics hot path.** `forward_kinematics` calls `_validate_q` at `forward.py:214` **and again identically** at `:224`, then `forward_kinematics_raw:122` validates a third time; `_validate_q` and `execution_batch_shape` *each* call `values.validate(structure)`, so one `forward_kinematics(model,q)` runs `values.validate` 6× (7× with frames). Every dynamics wrapper repeats the pattern (`rnea.py:263` then `:128`, etc.). Validate **once at the public boundary**; let the `*_raw` passes trust pre-validated inputs. Fix the false `forward.py:222` comment ("the backend impl trusts inputs" — currently untrue). | `kinematics/forward.py:214,224,282`; `dynamics/{rnea,aba,crba,centroidal}.py` wrapper/raw pairs; `data_model/model_values.py:95` | removes ~4–6 redundant full validations per pass call; real sync/CPU-work win | one `values.validate` per public call (assert with a counter test); comment corrected; parity + contract suites green |
| T8 | Deduplicate `_batch_shape` and `_blend_values` — **byte-identical** in `solver_lm.py` (`:103,187`) and `solver_adam.py` (`:55,85`). Factor into one shared internal module (e.g. `optim/blocks/_solver_common.py`). | `optim/blocks/solver_{lm,adam}.py` | ~-12 LOC, removes drift risk | one definition; both solvers import it; solver suites green |
| T9 | Collapse `reduce_generalized_force` and `reduce_jacobian` — byte-identical bodies (`x @ v_expansion`) in the new mimic module. Keep both public names as thin aliases only if a call-site reads better; otherwise one function. | `data_model/reduced_coordinates.py:37-56` | ~-10 LOC | one implementation; the 5 existing callers + mimic tests green |
| T10 | Collapse the Warp torch-VJP duplication: `_torch_joint_transform` + `_torch_fk_from_tables` reimplement `joint_dispatch.joint_transform` + `forward_kinematics_raw`. The structure/value seam exists precisely so the recompute VJP can reuse the raw pass. **Medium risk** — VJP correctness — but fully gated by the existing Warp gradcheck-vs-torch parity tests. | `kinematics/_warp_bridge.py:38-129` | ~-90 LOC | recompute path reuses the raw FK pass; Warp gradcheck + parity (fp32/fp64, branched, >16 joints) green |

Tier 2 total: ~-110 LOC plus the hot-path validation win (T7, the highest-value item here — do it carefully with a before/after `values.validate` call-count assertion).

---

## Tier 3 — Abstraction collapse (M; medium value; touches correctness-adjacent code)

Do these only after Tiers 1–2 are committed and green. Each needs its own
verification because it touches logic, not just dead lines.

| # | Task | Where | Win | Guardrail |
|---|------|-------|-----|-----------|
| T11 | Unify the **triplicated per-joint-kind dispatch**. Three switch statements encode overlapping joint semantics: `manifolds._joint_box_mask` (box-projectable coords), `manifolds._joint_unit_ranges` (unit-norm slices), `solver_lm._joint_q_for_v` (tangent→q map + free-flyer "unsafe" set) — 17 `kind ==`/`startswith` sites across two files. Introduce one per-joint-kind descriptor (ideally hung off the joint model) that all three read. The free-flyer "unsafe translation" branch (`solver_lm.py:206-228,287-298`) is defensive-only (finite free-flyer translation bounds are rejected upstream) — simplify, don't extend it. | `optim/blocks/manifolds.py`, `optim/blocks/solver_lm.py` | ~-30–40 LOC; removes 3-way drift risk | the RobotConfig q↔v bound-mapping tests (11 cases: revolute/prismatic/helical/translation/planar/nested/free-flyer-reject) stay green; active-set + KKT bound tests green |
| T12 | Replace the **17-element positional tuple** returned by `_build_mimic_reduction` (`build_model.py:118-314`, unpacked at `:803-822`) with a frozen dataclass / NamedTuple. Build-time only, one caller; makes the boundary auditable. | `io/build_model.py` | ~0 net LOC, big readability win | mimic build + reduced-map tests green |

---

## Tier 4 — (resolved: deferred to M5)

The `BSplineTrajectory` removal previously proposed here is **deferred to M5**
per owner decision 2026-07-17, together with the `solve_trajopt` migration it is
coupled to. Do not touch it in M3.5. Rationale and the exact surface are in
§"Deferred to later milestones" item 2.

---

## Tier 5 — Test-suite hygiene (S; do alongside the code tasks they touch)

| # | Task | Where |
|---|------|-------|
| T14 | Remove tautological / non-asserting tests: `contract/test_shape_annotations.py:28-46` computes a ratio and **never asserts** ("Always passes — advisory"); `contract/test_docstrings.py:44-47` has near-vacuous operator-precedence logic. Either give them a real assertion or delete. | `tests/contract/` |
| T15 | Dedupe the ~180 LOC of hard-duplicate optim tests flagged by the audit: robust-kernel `w=2ρ'` identity (`test_robust_kernels.py:53-68` vs `test_kernel_rho_weight_consistency.py:38-58` — keep the latter); Huber-accept/Tukey-reject (`test_solver_quality.py:195-252` vs `test_solver_lm_damping.py:392-451`); the two near-identical Panda P6 tests (`test_solver_quality.py:256-327` vs `:329-396`). Keep one authoritative copy of each; do not thin genuine batched-vs-sequential parity coverage. | `tests/optim/` |

**Genuine coverage GAP to add (not a deletion) — mimic × optim is untested.**
No `solve_ik`/`Problem` run exists on a reduced-coordinate (mimic) model. Add
one small IK-on-a-mimic-gripper test so the reduced map is exercised through
the solver, not only through the raw passes. (Fixed-base revolute-Z mimic is
the only shape currently covered even in the raw tests.)

---

## Tier 6 — Truth fixes (S; the honesty rule made concrete)

Verified drift to correct in the same commits as the code above, or as one
"docs truth" commit:

- `kinematics/forward.py:222` "the backend impl trusts inputs" — **false**; fixed by T7.
- `m1_results.md:12` "blocking CI boundaries" — CI is `on: workflow_dispatch`
  (manual only, `.github/workflows/ci.yml:4`); nothing gates push/PR. State it
  honestly (or, if the owner wants a real gate, that is a separate infra task —
  do not silently claim one).
- Root `CLAUDE.md`: `test_public_api.py` "adds SE3 and ModelBuilder to the prior
  25" implies 27; the `REQUIRED` frozenset has **24**. Correct the count.
- `m2c_results.md:29` says `solve_ik` installs "position, orientation, limit,
  and rest" residuals; the facade installs only Pose + JointPositionLimit +
  Rest (`ik.py:313,327,343`). Correct the claim.
- `apply_jac_transpose` is advertised as a **live** matrix-free feature in root
  `CLAUDE.md`, `residuals/CLAUDE.md`, and `residuals/base.py:8-12`, but **no
  production code calls `.gradient`** on any legacy problem — only
  `tests/optim/test_matrix_free.py`. Either state it as test-only, or (cleaner)
  fold its removal into the M4/M5 legacy-stack deletion and drop the claim then.
  For M3.5: correct the docs to stop calling it live.

---

## Deferred to later milestones (out of scope for M3.5)

These are real, but not M3.5-safe. The owner resolved each deferral on
2026-07-17. Documented here with its true gate so the executor does not touch it.

1. **The legacy optimizer stack (~1,400 LOC): `optim/optimizers/` (Adam, GN,
   LBFGS, MultiStage, LMThenLBFGS, legacy LM), `optim/problem.py`
   (`LeastSquaresProblem`), `optim/state.py`, `optim/cost_stack.py`,
   `optim/strategies/`, the `costs/` shim.** Held alive by the single production
   caller `tasks/trajopt.py`. Deletion is gated on (a) migrating `solve_trajopt`
   onto a named-block `Problem` — **M5** scope (manifold-safe trajectory work),
   and (b) the **M4** consumer migration (BHF imports `CostStack`,
   `GaussNewton.minimize`, `LeastSquaresProblem`). When it lands, ~1,400 src +
   ~900 test LOC go with it, and these contract assertions need their expected
   sets updated (do not delete the checks): `test_public_api.py:45,47`,
   `test_submodule_public_imports.py`, `test_cost_stack_location.py` (whole
   file), `test_layer_dependencies.py` (`costs/` forwarding assertion),
   `test_solver_state.py`, `test_protocols.py`/`test_pluggable_protocols.py`
   (drop legacy Optimizer/strategy cases), `test_docstrings.py`. **Do not start
   this in M3.5** (owner decision 2026-07-17: legacy-stack deletion rides with
   the M4/M5 work).
   - **NOTE — `optim/kernels/` and `optim/solvers/` are NOT legacy.** The
     new named-block stack (`optim/blocks/solver_lm.py:24-27`) and `tasks/ik.py`
     import them. They stay. (The original review handoff mislabeled them;
     corrected here.)

2. **`BSplineTrajectory` (Cox–de Boor basis, ~97 LOC) → M5.** Unexported,
   rejected by `solve_trajopt`, exercised only by its own unit tests + 3
   rejection tests. Deleting it now would be a clean two-caller-rule win, but it
   is coupled to the `solve_trajopt` migration, and **M5 is chartered to build
   the manifold-safe replacement** — so per owner decision 2026-07-17 it stays
   as-is until M5 rewrites it correctly. Surface, for the M5 executor:
   `tasks/parameterization.py:61-157`, the `parameterization.init/expand`
   wrapper in `trajopt.py:130-167` (which collapses to a no-op once
   `KnotTrajectory` is the only impl), and the tests in `test_trajopt_param.py`.
   Do not touch in M3.5.

3. **`Model` / `ModelValues` state duplication (~15 tensors) → M4.** `Model`
   still carries `joint_placements`, `body_inertias`, and the limit tensors as
   direct fields that `ModelValues.from_model` copies, and several sites still
   read `model.joint_placements` directly for device/dtype (`model.py:292,362`,
   `model_structure.py:231`). This is the largest state redundancy in
   `data_model` — but it is an **M1 artifact on a near-public surface**, not M3
   overcomplication. **Owner decision 2026-07-17: deferred to the M4 migration**
   (safer to move consumers before churning the near-public Model surface; needs
   a ledger entry and moving those reads to `.values.*`).

4. **The public/prevalidated "shadow API" → deferred (revisit at M6 freeze).**
   `VarSpec.{retract,difference,gather_tangent,expand_tangent}` and
   `Problem.{normal_matrix,dense_jacobian}` (the public host-syncing variants)
   have no production caller — production goes through the `_*_prevalidated`
   twins. This roughly doubles the math-method count and is the real driver of
   the file sizes. It is **motivated** (sync-free hot path + frozen M2a public
   contract + custom-residual author guide), not accidental. **Owner decision
   2026-07-17: deferred** — it is stable and correct, so it is not touched now;
   whether to narrow the frozen public surface is revisited when the API is
   re-frozen (M6). (T6's `normal_matrix` removal is the one piece of this that is
   unambiguously test-only and safe to do in M3.5.)

5. **Milestone-evidence benchmark scripts** (`benchmarks/m2a_dense_assembly.py`
   ~470 LOC, `tests/bench/bench_solver_convergence_scipy.py`, the uncommitted
   `bench_integrate_difference.py`) are per-milestone advisory evidence, not in
   CI. Consolidating them under one advisory harness (or pruning post-acceptance)
   is worthwhile but is housekeeping, not correctness — do it opportunistically,
   not as a gating M3.5 task.

---

## Verification protocol (per task and at milestone close)

1. After each task: `uv run pytest <the directly affected test dirs> -q`, then
   the full `-m "not bench and not cuda"` suite. Green or revert.
2. At milestone close: full suite green (target: still 1,371 passed minus only
   the tests intentionally removed in T14/T15, plus the new mimic-IK test);
   `tests/test_pinocchio/` and `tests/contract/` green; `git diff --check`
   clean; scoped Ruff clean on changed files.
3. Report, per README working-style: LOC removed per task (src + test), every
   removed consumer-facing symbol in `docs/reference/m3_removed_symbols.md`, and
   any task where the plan and code disagreed (stop and flag, don't guess).

## Expected size delta

Tiers 1–3 + T14/T15: roughly **−350 to −500 src LOC and −250 test LOC**, all
low risk, against a green suite. This is the honest cleanable-now surface. The
headline multi-thousand-LOC reduction the raw counts suggest lives in the
out-of-scope legacy stack (§1) and is an M4/M5 outcome, not an M3.5 one — the
owner should not expect M3.5 to shrink the tree by thousands of lines.
