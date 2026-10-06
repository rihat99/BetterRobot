# STATE — BetterRobot / 2026-10-06_23-45_p1_optim (updated 2026-10-07, night)

Status: DONE (reported, committed on `dev`, not pushed). Open items moved to later phases, listed below.

## Pointers
- Report: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_23-45_p1_optim/REPORT.md
- Reviews: .../subagents/review_optim2.md, .../subagents/review_optim2_additions.md
- BVR fit study: .../subagents/study_bvr_fit.md
- Agent guide of the new package: /home/rikhat.akizhanov/better/dev/BetterRobot/src/better_robot/optim2/CLAUDE.md
- Design + owner decisions (section 6): /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md
- P0 report: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_21-40_p0_measure_and_fix/REPORT.md
- Owner is asleep (since ~23:30 on 2026-10-06): work automatically, ask nothing. Never push/merge; commit on `dev`.
- Background HTTP server for plan pages on 127.0.0.1:8765 (never bind 0.0.0.0). Telegram pings do not work.

## What exists
- src/better_robot/optim2/ (problem, linear incl. BandedNormal + ArrowNormal, solvers, phases, constrained,
  manifold, kernels) and src/better_robot/terms/ (robot terms, IK recipe). tests/optim2 (21). examples/optim2/01..05.
- Full suite: only the 2 known pre-existing failures (tests/optim/test_temporal_structure.py::
  test_banded_and_dense_lm_match_with_outer_reduction_and_activity, tests/kinematics/test_fk_regression.py::test_fk_matches_oracle[g1]).

## Next steps (new tasks; create a new task folder per phase)
1. P2 model/kin/dyn (DESIGN.md D1, D2, roadmap section 3): lean FK from
   .../2026-10-06_21-40_p0_measure_and_fix/subagents/proto_lean_fk.py into the library; KinematicState stores
   rotations + translations; remove Warp (owner decision Q1); URDF parser float32 rounding
   (src/better_robot/io/parsers/urdf.py:29-30); G1 FK oracle mismatch; minimal collision (sphere/capsule,
   sphere-SDF), then obstacle trajopt on optim2. Benchmarks with benchmarks/run.py against
   benchmarks/results/0953f28_cpu.json.
2. GPU baselines (`uv run python benchmarks/run.py --device cuda`) only on a FREE GPU (all BUSY at every check so far);
   Warp record before Warp is removed (if no GPU frees up, record CPU Warp numbers only and say so).
3. P3 viewer (incl. offline recording), P4 BetterHuman (incl. BVR person-fit port on optim2).
4. Owner's calls, do not do unasked: delete old `optim`/`residuals`/`tasks`; rename `optim2` -> `optim`;
   confirm names `KinematicState`, `forward_kinematics_state`.

## Assumptions (made off-hours)
- Temporary package names `optim2`, `terms`; names `KinematicState`, `forward_kinematics_state`.
- IK recipe default: hard joint limits, no rest/limit regularizers.
- Term names have their own namespace; variables, data, cached values share one.
- float32 LM stops on gradient 1e-7 or maximum damping only.
- solve_layer: implicit mode only, no stationarity check, rejects scalar terms. Unroll mode not built.
- BVR person-fit port postponed to P4 (needs BVR body model and data).
- solve_constrained ignores the constraint terms' own weight and kernel.
