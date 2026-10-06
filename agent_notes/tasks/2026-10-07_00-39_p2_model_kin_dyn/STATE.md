# STATE — BetterRobot / 2026-10-07_00-39_p2_model_kin_dyn (updated 2026-10-07 01:00 Dubai)

## Pointers
- Plan: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-07_00-39_p2_model_kin_dyn/PLAN.md
- Previous task (done): /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_23-45_p1_optim/REPORT.md
- Design: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md (D1, D2, section 6)
- Lean FK prototype: /home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_21-40_p0_measure_and_fix/subagents/proto_lean_fk.py (+ .md)
- Owner is asleep: work automatically, ask nothing. Never push/merge; commit on `dev`.
- Background HTTP server for plan pages on 127.0.0.1:8765 (never bind 0.0.0.0). Telegram pings do not work.

## Done
- M1 precision: URDF and MJCF parsers keep float64 (were rounding to float32). Float64 Panda vs pinocchio:
  9e-7 -> 8e-16 (subagents/m1_precision_before.json, m1_precision_after.json). Analytic vs autodiff Jacobian
  1e-7 -> 1e-15. FK oracle regenerated as version 4 (the old one stored the rounded model); pinocchio parity
  tolerances tightened to 1e-12 (FK) and 1e-10 (frame Jacobian). Viewer dtype fix (urdf_mesh.py:254).
- Full suite: 1 failure left, pre-existing: tests/optim/test_temporal_structure.py::
  test_banded_and_dense_lm_match_with_outer_reduction_and_activity (old optim, float32).

## Next steps
1. M2 lean FK in the library behind `forward_kinematics_state` (see PLAN.md checks).
2. M3 Jacobians/terms read the new state; M4 remove Warp (record numbers first); M5 collision; M6 review + REPORT.md.
3. GPU baselines only on a FREE GPU (all BUSY so far).

## Assumptions (made off-hours)
- Regenerating tests/kinematics/fk_reference.npz is justified by the parser fix; evidence is pinocchio parity at 1e-12.
  The old G1 oracle failure disappears with it; its original cause was not separately diagnosed.
- MJCF parser changed the same way; no MJCF precision measurement was made (tests pass).
- P0 golden files (IK, trajopt) were recorded with the rounded model; expect differences of about 1e-7.
