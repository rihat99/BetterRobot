# STATE — BetterRobot / 2026-10-06_23-45_p1_optim (updated 2026-10-07 00:15 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_23-45_p1_optim/PLAN.md
- Design + owner decisions (section 6): /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md
- P0 report (done): /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_21-40_p0_measure_and_fix/REPORT.md
- Review of the new code + dispositions: .../2026-10-06_23-45_p1_optim/subagents/review_optim2.md
- Measurements: .../2026-10-06_23-45_p1_optim/subagents/m1_ik_check.json, m1_ik_batch1024_1thread.json, m2_trajopt_check.json, m2_long_horizon.json
- Owner is asleep (since ~23:30 on 2026-10-06) and said: work automatically, ask nothing. Never push/merge; commit on `dev`.
- Background HTTP server for plan pages on 127.0.0.1:8765 (never bind 0.0.0.0). Telegram pings do not work.

## Done (last commit 65b63bc on dev)
- src/better_robot/optim2/: problem.py (Problem, TermFn, cached, residual/scalar, window=, linearize, gradient,
  check_jacobian), linear.py (Normal, BandedNormal, sinks), solvers.py (LevenbergMarquardt, GaussNewton, LBFGS,
  TorchOptim), phases.py (Phase, run, solve, solve_layer), manifold.py, kernels.py. About 1,300 lines.
- src/better_robot/terms/: robot.py (RobotManifold, kinematic_state, frame_pose, frame_position, joint_limits, rest,
  finite_difference), ik.py (ik_problem, solve_ik with restarts, random_start).
- kinematics/state.py: KinematicState, forward_kinematics_state, frame_jacobian (joint Jacobians memoized).
- tests/optim2/ (16 tests). examples/optim2/01..04 run. Full suite: only the 2 known pre-existing failures
  (tests/optim/test_temporal_structure.py::test_banded_and_dense_lm_match_with_outer_reduction_and_activity,
  tests/kinematics/test_fk_regression.py::test_fk_matches_oracle[g1]).
- Numbers: IK batch 1024 f32 1 thread: new LM 95.5%/72.5% reached (spread 0.3/1.0) in 2.5 s; old 79%/66% in 3.7 s;
  4 restarts 98.8%/98.2% in 12.8 s. Panda trajopt 0.086 s vs old 0.3 s. G1 300 frames (10,500 unknowns): 8.8 s / 30 LM steps.

## Next steps
1. Write REPORT.md for P1 (interim): what exists, numbers above, what is missing, review record, assumptions.
2. Missing from design D3: augmented-Lagrangian recipe; unroll mode of solve_layer; variables shared over time
   under the banded solver (arrowhead; today they force the dense path); BVR person-fit port (BVR is read-only:
   /home/rikhat.akizhanov/better/BetterVideoReconstruction, engine tools/optim.py); obstacle trajopt (needs collision, P2).
3. Old `optim`, `residuals`, `tasks` are NOT deleted: owner's call. Docs/CLAUDE.md guides not yet updated for optim2.
4. From P0: GPU baselines (`uv run python benchmarks/run.py --device cuda`) only on a FREE GPU (all busy at 00:13);
   Warp record; BVR clip reference; URDF float32 rounding (P2).
5. After P1: P2 (model/kin/dyn: lean FK from P0 prototype subagents/proto_lean_fk.py, remove Warp), P3 viewer, P4 BetterHuman.

## Assumptions (made off-hours)
- Temporary package names `optim2`, `terms`; names `KinematicState`, `forward_kinematics_state`.
- IK recipe default: hard joint limits, no rest/limit regularizers.
- Term names have their own namespace; variables, data, cached values share one.
- float32 LM stops on gradient 1e-7 or maximum damping only.
- solve_layer: implicit mode only; rejects scalar terms; value is exactly the solver's result.

## Compact message
P1 optim core is built, reviewed and committed (65b63bc); continue with step 1 of Next steps (REPORT.md), then the missing pieces in order.
