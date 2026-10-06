# STATE — BetterRobot / 2026-10-07_00-39_p2_model_kin_dyn (updated 2026-10-07 02:05 Dubai)

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
  obstacle trajectory example (d79e570). Interim REPORT.md written (same folder). Docs Warp scrub by mars_light
  (check `git status`; commit if clean).

## Next steps
1. Commit the docs scrub if the agent left the tree clean (`grep -rni warp docs` should show only history).
2. Dynamics on KinematicState: RNEA/CRBA/ABA read matrices instead of 7-vector poses; measure with
   `benchmarks/run.py --algos rnea,crba,aba` against benchmarks/results/0953f28_cpu.json; P0 found 200-joint
   dynamics backward extremely slow (CRBA fwd+bwd at 4096: 800 s).
3. Adversarial review of the collision layer (not yet reviewed). Finalize REPORT.md; drop the index line.
4. Then P3 viewer (new task folder), P4 BetterHuman.
5. GPU baselines only on a FREE GPU (all BUSY at every check so far).

## Assumptions (made off-hours)
- Regenerating tests/kinematics/fk_reference.npz is justified by the parser fix; evidence is pinocchio parity at 1e-12.
- MJCF parser changed the same way; no MJCF precision measurement was made (tests pass).
- P0 golden files (IK, trajopt) were recorded with the rounded model; expect differences of about 1e-7.
- `KinematicState` field names `joint_transform`, `frame_transform`; method `frame_pose_of(frame_id)`.
- The old kind-group FK in _fk_matrix.py was deleted (replaced, same public functions); it is in git history.
