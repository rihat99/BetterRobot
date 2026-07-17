# for_agents — Execution Instructions per Roadmap Step

This folder turns `plan/04_roadmap.md` into hand-off-ready work orders.
One file per roadmap step, each self-contained enough to give directly to
a coding agent. Rationale and evidence live in the numbered plan documents
and `plan/research/`; these files tell you *what to do, in what order,
what to test, and what not to forget*.

## Files, in execution order

| File | Roadmap step | Depends on |
|------|--------------|------------|
| `m0_truth_and_correctness.md` | M0 — stop the library lying (10 small fixes) | nothing; items independent |
| `m1_two_lane_seam_and_hygiene.md` | M1 — delete `backends/`, build the ModelStructure/ModelValues seam, bridge prototype, first Warp kernel, hygiene | M0 |
| `m2a_variable_blocks_and_slice.md` | M2a — manifolds, VarSpec/Values/Problem, provider DAG, consumer vertical slice | M0 (θ=0 fix), M1 (seam) |
| `m2b_batched_second_order_solvers.md` | M2b — batched LM/GN, real bounded algorithm, capture-ready `update` | M2a |
| `m2c_first_order_phases_tasks.md` | M2c — matrix-free Adam, phase engine, `solve_ik` re-base, trajopt fix | M2a, M2b |
| `m3_parametric_model_breadth.md` | M3 — batched/parametric ModelValues across all passes, mimic coordinate map, order-preserving build | M1 (frozen ABI), M2 |
| `m4_consumer_packs_and_migration.md` | M4 — vision residuals, contact-force task, utilities, viewer/collision decisions, full consumer migration | M2, M3 |
| `m5_sparse_trajectory_structure.md` | M5 — symbolic block sparsity, banded solvers, Schur elimination | M2 |
| `m6_warp_fast_path_and_cuda_graphs.md` | M6 — GPU baseline, kernel build-out, CUDA-graph capture, external benchmarks | M1 (bridge), M2b (capture-ready), real GPU box |

Within a milestone, each file states which tasks are parallelizable and
which are ordered. Do not start a milestone before its dependencies'
acceptance checklists pass.

## Standing rules — apply to every task in every file

These come from `plan/04_roadmap.md` and are binding:

1. **Branch strategy (owner decision 2026-07-17).** The redesign is
   implemented on a **dedicated branch**; BHF/BVR keep consuming the
   pre-redesign branch until the M4 migration. Consumer-compatibility
   shims are therefore **not required**, and "keep BHF's legacy imports
   working" instructions inside the milestone files are superseded —
   legacy surfaces may be deleted outright once no BR-internal caller
   consumes them. What remains mandatory: record every removed
   consumer-facing symbol (grep both consumer repos,
   `/data3/rikhat.akizhanov/better/BetterVideoReconstruction` and
   `/data3/rikhat.akizhanov/better/BetterHumanForce`) so M4's
   symbol-by-symbol migration table is complete when the consumers switch
   branches.
2. **A misleading capability is repaired, or made to fail fast with an
   honest error, in the same milestone where the lie is discovered.** Never
   leave a false claim advertising itself — in code, docstrings, `docs/`,
   or any `CLAUDE.md`.
3. **Batched-vs-sequential parity uses stated tolerances and per-element
   statuses, not exact equality.** Per-element branch decisions (damping
   accept/reject) legitimately differ near thresholds.
4. **Performance acceptance uses committed benchmark definitions**
   (hardware, dtype, shapes, warmup, statistics — checked into the repo),
   never bare ratios in a chat log.
5. **No new abstraction without a second concrete caller in-tree.**
6. **Every Warp kernel lands with all five of:** (a) an adjoint-strategy
   entry in the design table (03 §2.3); (b) parity vs the torch lane
   (fp32/fp64, several batch shapes, both base types); (c) gradcheck vs
   the torch lane including singular points, branched trees, chains >16
   joints; (d) a warp-CPU parity run in CI; (e) a committed benchmark
   against the compiled torch lane. Warp-CPU parity alone = **prototype**;
   **production** requires CUDA validation on a real GPU runner;
   **default-on** additionally requires benchmark evidence.
7. **CUDA is broken on this dev box.** Local warp validation is warp-CPU
   only. Never claim GPU validation from this machine. Anything needing a
   real GPU is blocked on the remote CUDA runner (an M1 action item).
8. **Evidence-gated decisions stay with the owner.** Where an instruction
   file says "produce the evidence and stop for review", do exactly that —
   do not silently pick a default.

## Repo facts every executor needs

- Repo root: `/data3/rikhat.akizhanov/better/BetterRobot`. Source in
  `src/better_robot/`, tests in `tests/`.
- Run tests: `uv run pytest tests/ -v` (897 pass in ~52 s today; all must
  stay green unless an instruction file explicitly retires one).
- The pinocchio-parity suite (`tests/test_pinocchio/`) is the most
  valuable asset in the repo — it is what makes aggressive refactoring
  safe. It stays green through every milestone.
- Contract tests in `tests/contract/` enforce the layer DAG, API surface,
  hot-path lint, etc., by AST parsing. Some instruction files modify these
  contracts deliberately; never weaken them as a side effect.
- Conventions that never change (see repo `CLAUDE.md`): SE3 pose is
  `[tx,ty,tz,qx,qy,qz,qw]` (scalar-last quaternion); se3 tangent is
  `[linear(3), angular(3)]`; `get_frame_jacobian` returns
  LOCAL_WORLD_ALIGNED; everything batched `(B..., feature)`.
- Docs and per-package `CLAUDE.md` files have a verified history of drift
  — 6 of 10 sampled doc claims were wrong. **Never trust a doc claim
  without checking the code.** When your change makes a doc/CLAUDE.md
  statement false (or true again), update it in the same change.
- Consumer evidence: BVR's `tools/optim.py` (188-line Problem/Phase
  engine) and `tools/human_optim/` are the de-facto requirements spec for
  the optimization redesign; BHF's `scripts/motion/optimize_motion.py`
  and `tools/robot_motion/` hold the live imports that constrain
  deletions.
- Reference digests (how other libraries solve the same problem):
  `references/design/*.md` — frax, jaxlie, jaxopt, optax, pyroki, curobo,
  warp, mujoco_warp, newton, PyposeWarp, pinocchio, and more.
- Plan documents: `plan/01_assessment.md` (verified current state),
  `plan/02_vision.md` (scope fence), `plan/03_architecture.md` (design
  decisions + API sketches), `plan/04_roadmap.md` (milestones), and
  `plan/research/*.md` (raw audits — input material, not verdicts).

## Working style expected from executors

- Transform every task into a verifiable goal before coding: write or
  identify the test that proves it, then make it pass.
- API sketches in the plan show direction, not final signatures — small
  deviations are fine when the code demands them; document them.
- Surgical changes: touch only what the task requires; match existing
  style; update the docs the change invalidates; nothing speculative.
- Report honestly: failing tests, skipped steps, and plan-vs-code
  discrepancies go in your report verbatim. If the plan contradicts what
  you find in the code, say so and stop rather than guessing.
- If workspace skills are available in your session (`python-standards`,
  `write-tests`, `file-naming`, `design-principles`, `sphinx-docs`),
  invoke them before the corresponding work.
