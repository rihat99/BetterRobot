# STATE — BetterRobot / 2026-10-06_21-40_p0_measure_and_fix (updated 2026-10-06 22:30)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_21-40_p0_measure_and_fix/PLAN.md
- Approved redesign + owner decisions (section 6): /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md
- Subagent briefs share: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_21-40_p0_measure_and_fix/subagents/context.md
- Repos: /home/rikhat.akizhanov/better/dev/BetterRobot and .../BetterHuman (branch dev, commit only, never push/merge). BVR read-only.

## Done
- Step 1: `uv sync --all-extras` in the dev venv (pinocchio 3.9, warp 1.15, viser 1.0.26). Suite: 1762 pass, 2 fail, 62 skip.
  The 2 failures exist at the starting commit with this env (not caused by this task):
  tests/optim/test_temporal_structure.py::test_banded_and_dense_lm_match_with_outer_reduction_and_activity (float32, 1.8e-4 vs 4e-5 tol)
  tests/kinematics/test_fk_regression.py::test_fk_matches_oracle[g1] (2e-8 vs 1e-10; was skipped before, robot_descriptions missing).
- Step 2: commit 0953f28. SE3 right Jacobian closed form (was wrong 1e-2 at 3 rad); so3 inverse beyond pi;
  random_configuration dtype + unlimited joints. "FK dtype crash" is not a bug at the public API (DtypeMismatchError is raised).
- IK check (subagents/ik_jacobian_fix_check.json): the Jacobian fix does NOT change batched IK convergence.
  Default IK leaves ~0.4 mm median error because of rest/limit regularizers (golden/summary.json); without them 1e-6 m median.
- Step 8 (partial): golden/ has IK and trajopt inputs + outputs (make_golden.py, summary.json, *.npz). BVR clip golden not done.

## Running
- Subagent "bench runner": writes benchmarks/run.py, results JSON, note subagents/bench_runner.md.
- Subagent "lean FK" (E3): subagents/proto_lean_fk.py + .md.
- Background HTTP server for plan pages on 127.0.0.1:8765 (do not bind to 0.0.0.0).

## Next steps
- Integrate both subagent results; commit benchmarks/run.py + results.
- Step 7: KinematicState shim, shaped by the lean FK result (rotation matrices + translations).
- GPU baselines + Warp record (E1) when a GPU is FREE (gpu_free.py); E2 compile matrix.
- REPORT.md, commit. Then P1 optim (owner: optim first).

## Assumptions pending user confirmation
- Name `KinematicState` (variable `state`).
- Telegram pings do not work (TG_BOT_TOKEN / TG_CHAT_ID unset).
