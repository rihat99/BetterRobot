# STATE — BetterRobot / 2026-10-06_21-40_p0_measure_and_fix (updated 2026-10-06 23:40)

Status: DONE except GPU work (no free GPU). Report:
/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_21-40_p0_measure_and_fix/REPORT.md

Carried over to the next task:
- GPU baselines: `uv run python benchmarks/run.py --device cuda` on a FREE GPU (gpu_free.py), and the Warp-vs-torch record (E1).
- BVR clip reference output.
- URDF parser rounds to float32 (P2). G1 FK reference mismatch 2e-8 not investigated.
- Owner has not confirmed the names `KinematicState` / `forward_kinematics_state`.
- Known failing tests (pre-existing): tests/optim/test_temporal_structure.py::test_banded_and_dense_lm_match_with_outer_reduction_and_activity, tests/kinematics/test_fk_regression.py::test_fk_matches_oracle[g1].
