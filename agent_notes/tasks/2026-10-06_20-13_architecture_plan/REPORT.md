# REPORT — architecture plan for BetterRobot + BetterHuman

Task `2026-10-06_20-13_architecture_plan`, finished 2026-10-06. A plan only: no library source
was changed. Commits analysed: BetterRobot `dev` 7e2008f, BetterHuman `dev` 793d716.

## Deliverables (absolute paths)

- Pages (open `index.html`; five self-contained pages linked by a top navigation):
  `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/site/index.html`
  plus `optim.html`, `speed.html`, `viewer.html`, `human.html` in the same folder.
  Sources in `site/src/`, rebuilt with `python3 site/build.py`.
- Interactive architecture diagrams (archify):
  `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/site/diagrams/br-current/br-current.html`
  (today, every box linked to source lines) and
  `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/site/diagrams/target/target.html`
  (proposed).
- The plan as text, with the review record:
  `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md`

## Result in ten lines

1. The optimization stack (8,509 lines, 35% of BetterRobot) is not used by BVR, which runs its
   own 193-line engine. Proposal: a three-layer optim where a cost is a plain function and the
   same costs run under `torch.optim`, Gauss-Newton or LM.
2. Warp covers only FK and RNEA and its backward pass re-runs torch. Proposal: one torch
   implementation with an operation budget per algorithm; delete Warp after one confirming
   measurement (E1).
3. The library is slow because of many tiny operations (about 220 GPU launches per FK call on
   25 joints), not because of the backend. A level-batched FK prototype cut a 223-joint
   forward + backward from 2,297 to 913 launches; small robots need a second change that has
   no prototype yet.
4. Automatic Jacobians work through the current kinematics (7 variants agree) at 2-5x the cost
   of analytic ones on GPU; so automatic by default, analytic for built-in kinematic terms.
5. Three viewers (about 8,000 lines) become one core in BetterRobot on viser: Viewer, Timeline,
   Layer; BetterHuman and BVR add layers.
6. BetterHuman keeps bake-once/stream-poses; families become data; per-frame expression
   without re-baking; public BetterRobot API only; retargeting and human terms added.
7. Benchmarking becomes part of the library: runner, JSON per commit, CI gate on operation counts.
8. Roadmap: P0 measure and fix, then P1 optim, P2 model/kin/dyn, P3 viewer in parallel, then
   P4 BetterHuman, P5 later.
9. Bugs found: SE3 right-Jacobian series truncated (`lie/tangents.py:187`), two dtype bugs in
   `random_configuration`/FK, pinocchio parity tests skip because pinocchio is not installed.
10. Nine decisions are left to the owner (list in `DESIGN.md` section 4 and on the overview page).

## Measured in this task (raw data in `artifacts/`, notes in `subagents/`)

| What | Result | Note |
|---|---|---|
| CPU timings, 1 thread, float32, Panda | FK 0.65 ms (batch 1), 1.09 (64), 41.9 (4096); RNEA 2.14; CRBA 1.87; ABA 6.21 (batch 1) | `subagents/bench_baseline.md` |
| Batched IK, Panda, batch 1024, CPU, defaults | 4.2 s; 67.5% reached tol 1e-6 in 100 iterations | final errors not recorded |
| GPU launches per FK call, batch 256 | 25 joints: 219 fwd / 796 fwd+bwd; 223 joints: 452 / 2,297 | `subagents/proto_level_fk.md` |
| Level-batched FK prototype | 223 joints: 274 / 913 launches, bit-identical; 25 joints: 221 / 724 | same |
| Autodiff Jacobian vs analytic | 2-5x slower on GPU (indicative), up to 11.5x on CPU at batch 4096 | `subagents/proto_autodiff_jacobian.md` |
| viser 1.0.30 capability probe | all needed primitives present | `subagents/study_viewer.md` |

## Not verified

- No clean GPU timings: all 8 GPUs were busy. GPU times are indicative; launch counts are exact.
- No fresh Warp or pinocchio numbers (neither is installed in the dev environment).
- Nothing was run for BetterHuman (no model files in the dev worktree; benchmark is CUDA-only).
  Its timings come from `docs/contracts.md` in that repo.
- Target sizes in lines and speed targets are estimates.
- Comparison of viewer libraries other than viser rests mostly on prior knowledge.
- The pages were rendered and inspected in a sample of desktop, phone and dark screenshots,
  not every screenshot.

## Review

One adversarial review by a clean-context agent: 17 findings, 2 critical. 16 accepted and
applied (see `DESIGN.md` section 5); name-based injection of term arguments kept as the
recommended default and left as an owner decision.

## Subagent material (kept as evidence, cited by the pages)

`subagents/scan_*.md` (audits), `study_*.md` (design studies), `proto_*.md` + scripts
(prototypes), `bench_baseline.md` + scripts, `factcheck_optim_external.md`, `context.md`.
