# 05 — Rewrite the documentation

**Goal:** documentation a human wrote for humans. A beginner who knows Python
but has never touched a robot can learn from it; an expert finds every design
decision argued, not asserted. Runs **after phases 1–4** — the API it
describes must be settled. Invoke the `sphinx-docs` and `diataxis-docs`
skills before writing.

**Scope:** the ~35 hand-written files under `docs/` (~8.6k lines; the
auto-generated `reference/api/` is regenerated, never hand-edited). Target:
~24 pages. Full audit evidence dated 2026-07-18 below.

**The voice.** Write like the best current pages (`concepts/model_and_data.md`
lines 1–31 is the gold standard: names the alternative, the failure
pressures, in plain sentences). Ban the machine register: no "path-specific
autograd coverage", no "prevalidated boundary", no sentence a reader must
parse twice. Short sentences. Concrete examples. Every "we do X" is followed
by "because Y, instead of Z".

---

## T1 — Restructure to the four quadrants

Target tree (current file → destination):

**getting_started/ (tutorials, ~7):** `index`, `installation` (trim the
mimic-internals smoke test at :55-58), **new** `01_robot_model` ("anatomy of
a robot model": what `load` returns, what `q`/joint/body/frame are — the
missing on-ramp), `02_forward_kinematics` (rewrite; *define FK*),
`03_inverse_kinematics` (rewrite; *define IK*; drop solver-internal leaks),
`04_floating_base`, **new** `05_batched_gpu` (a thousand problems in one
call; content from `batching_and_backends.md:52-63`).

**guides/ (how-to, ~5):** `custom_residual` (rewrite; de-jargon), **new**
`load_a_robot` (URDF/MJCF/builder, from `parsers_and_ir.md`), **new**
`visualize` (from `viewer.md` examples), **new**
`differentiate_through_kinematics` (the autograd recipe — the answer to
"PyTorch-native, so what?"), **new** `own_your_optimization_loop`
(`init_state`/`update`/warm-starts).

**concepts/ (explanation, 16 → ~9):** `why_betterrobot` (absorbs `vision.md`;
rewrite from the old vision's clean register, no consumer names), keep
`design_decisions` (exists; extend per T4), `architecture` (strip dual-stack
prose), `model_and_data` (keep; trim mimic digressions :150-157, :183-189),
`joints_bodies_frames` (keep), `lie_and_spatial` (keep + **beginner primer**:
what SE(3)/a quaternion/a tangent *is*, before the storage decisions; fix the
stale "matched PyPose" justification — the real argument is
SciPy/ROS/Eigen/Pinocchio alignment), `kinematics_and_jacobians` (keep +
fold in T3's Jacobian explainer), `dynamics` (keep — already the model:
defines RNEA/ABA/CRBA in one line each), **merge**
`residuals_and_costs` + `solver_stack` → `residuals_costs_and_solvers`
(~250 lines from ~1,100: the two-stack prose dies with the code; add the
least-squares/LM intuition and the why-own-LM argument), **merge**
`batching_and_backends` + `warp_bridge` → `the_compute_seam` (rename away
from "backends"; import the why-Warp-not-CUDA rationale now in
`design_decisions`), `viewer` and `parsers_and_ir` keep as short chapters;
`collision_and_geometry` demotes to a reference note (it documents stubs).

**reference/ (~9):** `api/` regenerated; `glossary` expanded (T3 terms);
`roadmap` keep-edit (the `NotImplementedError` inventory idea is good; the
stub list shrank in phase 3); `changelog` — rewrite the Unreleased section
without milestone framing; conventions pages (`naming`, `contracts` +
`engineering` merged into it and slimmed, `extension`, `testing`,
`performance`, `style`, `packaging`, `source_and_license`) de-jargoned;
**delete** `named_block_solvers.md` (fold anything unique into the solver
concept + API reference).

Update every toctree; the Sphinx build must be warning-clean except the four
offline-intersphinx warnings.

## T2 — Purge the process jargon

Grep list (docs/ excluding `_build/`, case-sensitive where shown):
`named-block`, `named block`, `legacy`, `prevalidated`, `pre-validated`,
`work order`, `vertical-slice`, `migration ledger`, `\bM[0-9]\b`, `M3.5`,
`milestone`, `roadmap M`, `GraphExecutor`, `owner approved`, `2026-07-1`.
Zero hits when done (exceptions: `source_and_license.md` may keep "ledger"
for its provenance table; `roadmap.md` may say "planned" without milestone
names). The audit's known hit list: `engineering.md:124-127,193,196,271,292,
325`, `testing.md:60,65,93,233`, `performance.md:156,164,253`,
`naming.md:217`, `packaging.md:92`, `solver_stack.md:208,253`,
`batching_and_backends.md:190-191`, `warp_bridge.md:19`,
`vision.md:102-103`, `tasks.md:308`, `custom_residuals.md:172,188`,
`index.md:99`, `CHANGELOG.md:8-47`, plus ~60 "named-block" occurrences that
disappear naturally when the pages are rewritten (there is only one stack
now — the qualifier is meaningless).

## T3 — The beginner explainers (the named gaps)

Write these, each in the page indicated, each in plain words first and math
second:

1. **What a Jacobian is, and the three ways to get one** (in
   `kinematics_and_jacobians`): a Jacobian says how outputs move when inputs
   wiggle; *analytic* = a hand-derived formula, exact and fast; *automatic
   differentiation* = the framework derives it, exact, needs a differentiable
   forward; *finite differences* = wiggle numerically, approximate and slow,
   works on anything — kept only as a debugging cross-check. When BetterRobot
   uses which, and why.
2. **SE(3)/quaternion primer** (top of `lie_and_spatial`): what a rotation
   quaternion is, why 4 numbers for 3 degrees of freedom, what SE(3) means,
   what a tangent/twist is — one screen, with pointers to real references
   (Lynch & Park, Solà's quaternion notes, micro Lie theory paper).
3. **FK, IK, least squares, LM in plain words** (FK/IK in their tutorials +
   glossary; LM in `residuals_costs_and_solvers`): FK: joints in, poses out.
   IK: target pose in, joints out — solved as optimization. Least squares:
   make many small errors simultaneously small. LM: Gauss–Newton with an
   adjustable damping knob between "bold Newton step" and "cautious gradient
   step".
4. **Batching intuition** (tutorial `05_batched_gpu`) — the existing
   explanation is good; move it where beginners look.

Add each term to the glossary. Wiki-style external references are welcome
throughout: Pinocchio and Featherstone for dynamics, Ceres for least squares,
MuJoCo/Drake for what we deliberately don't do — cite, don't paraphrase badly.

## T4 — Decision honesty

`concepts/design_decisions.md` exists (written 2026-07-18) and covers:
PyTorch-vs-bindings/JAX, Model/Data, batching, one fixed/floating path,
whole-pass seam, Warp-not-CUDA, own-LM-not-own-Adam, variable blocks,
scalar-last quaternions (the honest ecosystem argument), LOCAL_WORLD_ALIGNED,
no-simulation, and the costs. Your job: keep it accurate as phases 1–4 land
(e.g. the Adam claim becomes literally true after phase 2), link each chapter
to its relevant decision instead of re-arguing, and fold in anything the
audit found asserted-but-unjustified that the page missed.

## T5 — Executable truth

Every code snippet in the rewritten docs runs: keep/extend the existing
snippet-execution tests (the front page already has one). The "what ships"
lists must match the post-polish surface exactly — when in doubt, grep the
code, not the old docs (they drifted before; that is how this phase became
necessary).

## Acceptance

- Tree matches T1 (±owner-visible improvements); Sphinx warning-clean modulo
  the four offline-intersphinx warnings.
- T2 grep list: zero hits (stated exceptions aside).
- The four T3 explainers exist where named; glossary covers every term a
  tutorial uses.
- Snippet tests green; full gate green; results file with per-page disposition
  (kept/rewritten/merged/deleted) per standing rule 8.
