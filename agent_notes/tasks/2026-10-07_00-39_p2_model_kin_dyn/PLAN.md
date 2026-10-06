# PLAN — P2 model, kinematics, dynamics (BetterRobot)

Task id `2026-10-07_00-39_p2_model_kin_dyn`, branch `dev`. Owner asleep: work without asking, record assumptions.
Design: `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md` D1, D2, section 6.
Inputs: P0 report `.../2026-10-06_21-40_p0_measure_and_fix/REPORT.md`, lean FK prototype
`.../2026-10-06_21-40_p0_measure_and_fix/subagents/proto_lean_fk.py`, CPU baseline `benchmarks/results/0953f28_cpu.json`.

## Objective
One torch implementation of kinematics that is several times cheaper per call, exact in float64, with Warp removed.

## Milestones and checks
M1 Precision. URDF parser keeps float64 values (today it rounds to float32, `io/parsers/urdf.py:29-30`).
   Check: float64 Panda/G1 quaternions have unit norm to 1e-15; analytic vs autodiff Jacobian agree to 1e-10;
   find out whether `tests/kinematics/test_fk_regression.py::test_fk_matches_oracle[g1]` is explained by it.
M2 Lean FK in the library. Port the prototype behind `forward_kinematics_state`; `KinematicState` stores world
   rotations `(B, nj, 3, 3)` and translations; 7-vector poses are derived on request.
   Check: equal to the current FK to 1e-12 (float64) on Panda, G1, the 25-joint tree, free-flyer and every joint
   kind in the test models; gradients equal; pinocchio parity tests still pass; operator counts from
   `benchmarks/run.py` at or below the prototype (33 / 57 / 45); CPU time table against the baseline.
M3 Jacobians and optim2 terms read the new state directly (no quaternion round trip). Check: same tests, op counts.
M4 Remove Warp (owner decision Q1). Before deleting: record Warp vs torch numbers (GPU if one is FREE, else CPU
   only and say so). Check: full suite green except the known failures; no `warp` import left.
M5 Minimal collision (sphere, capsule, sphere-to-SDF) and an obstacle trajectory example on optim2.
M6 Adversarial review by a clean-context mars_aux; REPORT.md; commit.

## Rules
- No new dependencies. Never weaken pinocchio parity or contract tests. Commit on `dev` after each milestone; never push.
- Timings on CPU with a fixed thread count; GPU timings only on a FREE GPU.
- Old public functions keep working until their callers are moved; deletions of old packages stay the owner's call
  (Warp is the exception: decided in section 6, Q1).
