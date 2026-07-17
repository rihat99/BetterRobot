# BetterRobot — Assessment & Redesign Plan

This folder is a full health check of BetterRobot plus the redesign plan
that follows from it. Read this README first, then the four numbered
documents in order.

## How this plan was made

Written on 2026-07-16, in five steps:

1. **Five audits of the codebase** — architecture, code quality, gaps seen
   by the consumer projects, performance, and the optimization stack. The
   raw reports live in `research/`, each with file:line evidence and small
   benchmarks.
2. **Five digests of reference libraries** — how frax, jaxlie, PyposeWarp,
   jaxopt, and optax solve the same problems
   (`../references/design/{frax,jaxlie,PyposeWarp,jaxopt,optax}.md`).
3. **First-hand verification.** Every claim the plan leans on was re-run
   directly in this repo, not just copied from an audit report.
4. **An external adversarial review** by codex/gpt-5.6
   (`research/codex_plan_review.md`) — a different model, asked to attack
   the plan rather than agree with it. Its corrections are already folded
   in. The main ones: the case for deleting `backends/` is structural, not
   about speed (real dispatch overhead is ~2 µs per op, not the 19 µs one
   audit claimed); the explanation of the bounded-LM failure was corrected;
   the variable-block design gained seven binding constraints; the roadmap
   was re-ordered (M2 split in three, compatibility shims kept until the
   consumers migrate, a separate milestone for sparse trajectories); and a
   cross-cutting engineering contract (dtypes, quaternion conventions,
   threading, serialization, licensing, migration) was added as `03 §9`.
5. **A second adversarial review of the Warp-first revision** (see below).

**Revised the same day: the plan is now Warp-first.** The owner redirected
the backend strategy. The earlier plan said: prepare clean seams now, and
decide about Warp at the end of the roadmap, based on profiling. The new
decision: since the architecture is being redesigned anyway, design the
compute layer around Warp from the start — one rewrite instead of two. The
surface stays PyTorch (every public API takes and returns torch tensors);
the hot inner passes get Warp kernels; CPU support is kept through the
torch implementations. The design is `03 §2` (the "two-lane compute
model"). It is informed by three additional audits:
`research/audit_curobo_warp_integration.md` (how cuRobo really combines
CUDA, Warp, and torch), `research/audit_warp_platform.md` (what warp-lang
can and cannot do), and `research/audit_compute_pass_inventory.md` (every
compute pass in BR today and its loop structure).

A second codex adversarial round (`research/codex_warp_review.md`) then
attacked the Warp-first design. Its corrections are folded in too:

- The torch↔warp bridge must be a **pair** of functional custom ops
  (one forward, one backward). The first draft promised "one ~40-line
  helper that writes into caller-provided buffers" — that is impossible,
  because torch refuses to register autograd for ops that write into their
  arguments. A working prototype is an M1 deliverable, before any wider
  promise is made.
- The batching contract at the kernel boundary (the "execution-batch ABI")
  is frozen in M1, not M3.
- `ModelStructure` keeps a dual representation: static Python data for
  `torch.compile`, plus flat device tensors for kernels. This is what lets
  the torch lane keep its measured 5× compile speedup.
- Every kernel must record an explicit strategy for its backward pass.
  Warp's auto-generated gradients are silently wrong for loops whose
  length is only known at run time — this cannot be left to defaults.
- A kernel counts as a prototype until it is validated on real CUDA
  hardware. This dev box has no working CUDA, so securing a remote GPU
  runner is an M1 action item.
- A performance gap versus cuRobo's hand-written CUDA is expected and will
  be stated openly. The approved way to close it is Warp "static
  specialization" (compiling a kernel variant for one specific robot) —
  never a second kernel language.

**Revision 2026-07-17.** Per-milestone execution instructions were
generated into `plan/for_agents/`; writing them re-verified every plan
claim they touch against the code, and the corrections are folded into
01/03/04 (marked "2026-07-17"). Two owner decisions from the same day:
(a) the redesign is implemented on a **dedicated branch** — BHF/BVR stay
on the pre-redesign branch until the M4 migration, so consumer
compatibility shims are no longer required (see 04's branch-strategy
rule); (b) the mimic reject-at-build scope is owner-gated because the
Panda itself carries a finger mimic.

## Documents

| File | What it is |
|------|------------|
| `01_assessment.md` | The honest current state: what is broken, what is suspect, what is genuinely good. Every critical claim was reproduced first-hand. |
| `02_vision.md` | What BetterRobot should be: identity, scope fence, design principles. |
| `03_architecture.md` | The target architecture and the major redesign decisions, with API sketches: backends removal, the two-lane Warp design, variable blocks, the optimizer stack, the parametric Model, the residual library, the performance strategy. |
| `04_roadmap.md` | The milestones M0–M6, in order, each with acceptance criteria. |
| `for_agents/` | Execution-ready work orders — one file per roadmap step (M0…M6), each with per-task goals, current-state evidence, implementation plans, tests, pitfalls, and acceptance checklists. Start at `for_agents/README.md`. |
| `research/` | Raw audit reports (evidence with file:line). Treat them as input material, not verdicts — the documents above are calibrated against first-hand checks. |

## TL;DR verdicts

1. **Delete the `backends/` Protocol layer — for structural reasons, not
   speed.** It is ~530 lines of indirection with only one real
   implementation behind it. Its dependencies point the wrong way: the
   "bottom" layer imports kinematics and dynamics from the layers above it.
   Nothing consumes `DynamicsOps` at all. And it dispatches per tiny Lie
   operation — a granularity no future Warp backend could ever use. (It is
   not a speed problem: measured overhead is only ~2 µs per op.) The
   deletion lands together with the design of the real seam:
   `ModelStructure`/`ModelValues` plus tensor-only whole-pass functions —
   the exact boundary the Warp lane targets from M1.
2. **The single biggest redesign is variable blocks.** Everything the
   consumer projects cannot do today traces back to one root cause: the
   optimization stack hardwires a single flat `q` tensor. The fix is named
   parameter blocks, each with its own manifold, bounds, and masks — plus
   residuals that declare which blocks they touch. That is the redesign
   that lets BR own the optimization loops in BetterVideoReconstruction and
   BetterHumanForce, and host better_human.
3. **Fix correctness before features.** Five bugs are verified first-hand:
   NaN gradients at θ=0 (poisons exactly the tangent-space autograd the
   redesign needs); mimic joints silently ignored; batched `solve_ik`
   crashes even though "batched by default" is the headline claim; bounded
   LM stalls at active bounds with no honest terminal status; and the
   `AUTODIFF` strategy silently runs finite differences instead.
4. **Warp-first — committed now, honest per kernel.** (Owner decision
   2026-07-16, replacing the earlier "decide at M6" position.) BR adopts
   the two-lane compute model of `03 §2`: a torch reference lane (the
   oracle for testing, the CPU path via `torch.compile`, the only
   second-order path) plus warp-lang whole-pass kernels for every
   tree-scan pass and all geometry, one bridge contract, and CUDA-graph
   capture of solver loops. The cuRobo audit corrected the folklore:
   cuRobo's inner-loop speed comes from hand-written CUDA plus graph
   capture, with warp only as its leaf layer. BR copies the architecture
   (fused passes, one wrapper contract, capture) but keeps warp-lang as
   the single kernel language. Per-op Warp (the PyposeWarp approach) stays
   rejected. Each kernel becomes the default only when it beats the
   compiled torch lane on a committed benchmark.
5. **The keepers:** the Model/Data split, free-flyer-as-joint-1, the Lie
   math core (correct values — only the gradients at zero are broken), the
   pinocchio-parity test suite (the most valuable asset in the repo), the
   IR → build_model pipeline, and the `JointModel` per-kind dispatch shape.
