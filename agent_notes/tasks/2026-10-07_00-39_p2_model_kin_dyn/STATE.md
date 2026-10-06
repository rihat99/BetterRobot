# STATE — BetterRobot / 2026-10-07_00-39_p2_model_kin_dyn — DONE (2026-10-07 03:30 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_00-39_p2_model_kin_dyn/PLAN.md
- Previous task (done): /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_23-45_p1_optim/REPORT.md
- Design: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md (D1, D2, section 6)
- Benchmark table (CPU, before/after): .../2026-10-07_00-39_p2_model_kin_dyn/subagents/m2_bench_table.txt;
  raw: benchmarks/results/p2_lean_fk_cpu.json vs benchmarks/results/0953f28_cpu.json
- Operator breakdown script: .../subagents/m2_op_breakdown.py. IK rerun: .../subagents/m2_ik_batch1024_1thread.json
- Owner is asleep: work automatically, ask nothing. Never push/merge; commit on `dev`.
- Background HTTP server for plan pages on 127.0.0.1:8765 (never bind 0.0.0.0). Telegram pings do not work.

## Done
- M1 precision (commit 0ebac62): URDF/MJCF parsers keep float64. Float64 Panda vs pinocchio 9e-7 -> 8e-16.
  FK oracle regenerated (v4); pinocchio parity tolerances tightened to 1e-12 / 1e-10.
- M2 lean FK: src/better_robot/kinematics/_fk_lean.py (per-joint affine feature map + one matmul per tree level,
  [R|p] matrices). All FK goes through it: `_fk_matrix.matrix_core` (Data API) and `forward_kinematics_state`.
  `KinematicState` stores `joint_transform`/`frame_transform` (B..., n, 3, 4); poses derived lazily.
  Constant placement matrices cached on ModelValues (not when requires_grad or torch.func-wrapped).
- M3 `frame_jacobian(model, state, frame)` builds the world column table from the state's matrices and masks one
  joint; terms in src/better_robot/terms/robot.py read matrices.
- Numbers (CPU, 1 thread, float32): Panda FK ops 246 -> 48, batch 1 0.655 -> 0.21 ms, batch 4096 32.9 -> 6.6 ms;
  G1 FK 336 -> 70 ops, 2.5-3x; tree200 Jacobian fwd+bwd at 4096: 10.4 s -> 1.17 s. IK batch 1024: 2.54 -> 1.67 s.
- Finding: the prototype's single dense feature map costs O(joints^2) (tree200 batch 1 got 24x slower with it);
  replaced by a per-joint map (linear).
- Full suite: 1 failure left, pre-existing (tests/optim/test_temporal_structure.py::
  test_banded_and_dense_lm_match_with_outer_reduction_and_activity, old optim float32).

## Done since 01:25
- Review of lean FK: 3 findings fixed (a0921d1). Warp removed (688efdb) with CPU record. Collision layer +
  obstacle trajectory example (d79e570). Interim REPORT.md written (same folder). Docs Warp scrub committed (33607d6).
  Docs still describe a generic "fused implementation" in concepts/architecture.md, why_betterrobot.md,
  conventions/extension.md, performance.md (no Warp word; owner may want them reworded).

## Done since 02:10
- M6 dynamics (commit 4503509): one world-frame core `dynamics/_world.py` (support-matrix sums, no joint loops);
  state API `inverse_dynamics`, `mass_matrix`, `forward_dynamics`, `centroidal_map`, `center_of_mass`; Data API on
  the same core; ABA = Cholesky on M. Jacobian columns built per real column (no padding). Two float32 leftovers
  fixed (default gravity, random quaternions); pinocchio dynamics parity now 1e-15, tolerances tightened to 1e-10.
  Table: subagents/m6_bench_table.txt (batch 1: 2-19x; tree200 CRBA fwd+bwd 4096: 823 s -> 5.4 s; forward at
  batch 4096 on 1 CPU thread about equal, memory-bound). Full suite: 1782 passed, 1 pre-existing failure.
- M7 reviews integrated (72de682): float32 reference origin, geometry-only memo, fixed-joint self-collision pairs,
  collision docs. Final REPORT.md in this folder. Full suite 1785 passed, 1 pre-existing failure.

## Next steps (next task)
1. P3 viewer: new task folder; design in .../2026-10-06_20-13_architecture_plan/subagents/study_viewer.md §3 and
   DESIGN.md D4 + Q6/Q7 (offline recording without a browser). Then P4 BetterHuman.
2. GPU baselines only on a FREE GPU (all BUSY at every check through 02:45).
3. Open for the owner: composite joints in dynamics; Warp deletions sitting in commit d79e570.

## Assumptions (made off-hours)
- Regenerating tests/kinematics/fk_reference.npz is justified by the parser fix; evidence is pinocchio parity at 1e-12.
- MJCF parser changed the same way; no MJCF precision measurement was made (tests pass).
- P0 golden files (IK, trajopt) were recorded with the rounded model; expect differences of about 1e-7.
- `KinematicState` field names `joint_transform`, `frame_transform`; method `frame_pose_of(frame_id)`.
- The old kind-group FK in _fk_matrix.py was deleted (replaced, same public functions); it is in git history.

## Compact message
P2 milestones M1-M5 are committed (last 33607d6): precision, lean FK, Warp removed, collision; interim REPORT.md
exists. Continue with step 1 of Next steps (dynamics on KinematicState), then review collision, finalize REPORT.md.
