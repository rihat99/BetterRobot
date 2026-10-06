# STATE — BetterRobot / 2026-10-06_23-45_p1_optim (updated 2026-10-07 00:40)

- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_23-45_p1_optim/PLAN.md
- Design: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md (D3, section 6)
- Owner asleep since ~23:30 on 2026-10-06; instruction: work automatically, ask nothing. Record assumptions here.
- Background HTTP server for plan pages on 127.0.0.1:8765 (never bind 0.0.0.0).

## Done: M1 (committed)
- src/better_robot/optim2/ (problem, linear, manifold, kernels, solvers, phases; ~750 lines) and src/better_robot/terms/ (robot.py, ik.py).
- Tests: tests/optim2/ (10 pass). Full suite: only the 2 known pre-existing failures.
- IK numbers: subagents/m1_ik_check.json (stored problems), subagents/m1_ik_batch1024_1thread.json.
  Batch 1024 float32 1 thread: new LM 95.5% / 72.5% reached (spread 0.3 / 1.0) in 2.5 s; old 79% / 66% in 3.7 s
  (old default cost: 7% / 1%). With 4 restarts 98.8% / 98.2% in 12.8 s. float64 + 8 restarts on 256 stored problems: 100%.
- Lessons: hard clamping of bounds stalls LM, the active-set step (Normal.restrict) fixed it; absolute gradient
  tolerance must be 1e-7 in float32 or problems stop early; kernels need reshape not flatten for (B, k) residuals.

## Next steps
- M2: time axis. Variables (B, T, n); stencil terms; block-banded normal equations; port the Panda trajopt example
  and compare with golden/trajopt_panda.npz from the P0 task; batched L-BFGS.
- M3: solve_layer (implicit / unroll), augmented Lagrangian recipe, examples for the four use cases.
- M4: BVR person-fit port (read-only repo). M5: adversarial review (mars_aux), REPORT.md.
- Carried from P0: GPU baselines when a GPU is FREE; BVR clip reference.

## Assumptions (made off-hours)
- Temporary package names `optim2` and `terms`; old optim/residuals/tasks are not deleted without the owner.
- IK recipe default: hard joint limits, no rest/limit regularizers (so reachable targets are reached exactly).
- `restarts` re-solve the whole batch and keep the best per element (fixed shapes).
