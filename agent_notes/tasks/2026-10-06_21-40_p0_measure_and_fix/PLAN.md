# PLAN — P0 measure and fix (BetterRobot + BetterHuman)

Task id: `2026-10-06_21-40_p0_measure_and_fix`. Branch `dev` in both repos (owner: work in dev).
First phase of the approved redesign. The plan and all owner decisions:
`/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md` (sections 2, 3 and 6). Pages: `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/site/index.html`.

## Objective

Make the starting point measurable and correct before any rewrite.

## Steps and checks

1. Environment: install the declared dev group and the `warp` + `demos` extras in the dev venv
   (`uv sync`; they are already in pyproject.toml, nothing new is added).
   Check: `import pinocchio, warp, robot_descriptions` works; `tests/test_pinocchio` no longer skips.
2. Reproduce the known bugs with failing tests, then fix:
   a. `lie/tangents.py:187-204` SE3 right Jacobian truncated series (wrong for |xi| > ~1.5;
      measured 5e-3..4e-2 error in PoseResidual.jacobian far from target). Fix with a closed form
      or an adaptive series; check against finite differences and pinocchio for |xi| up to pi.
   b. `Model.random_configuration`: float32 output on a float64 model; inf for unlimited revolute joints.
   c. FK with mismatched model/q dtype crashes in `index_copy`: raise a clear validation error.
   Check: new tests fail before, pass after; full CPU test gate still green.
3. Benchmark runner (`benchmarks/`): algorithm x model (Panda, G1, 25-joint, 200-joint tree) x
   batch (1, 64, 256, 4096) x device x forward / forward+backward; JSON per commit; counts of
   operations per call (torch.profiler); pinocchio CPU reference. Main batch size is 4096.
   Check: one command writes a JSON; rerun reproduces counts exactly.
4. Baselines: CPU now; GPU when a GPU is FREE (all 8 were BUSY at 21:40 on 2026-10-06).
5. E1 (record, not a gate: owner decided to remove Warp): Warp vs torch, forward and
   forward+backward, to document what forward-only speed is lost.
6. E3 prototype: shrink the constant ~200 operations of FK on a 25-joint tree (rotation +
   translation instead of 4x4, one formula per joint family, no quaternion round trip).
   Target: under 80 forward operations. E2: compile feature matrix.
7. Freeze the `KinematicState` signature (immutable result of `forward_kinematics`; owner chose
   "frozen, renamed", rejected the name `kin`; `KinematicState` / variable `state` is the agent's
   working name) as a thin shim over today's code, so P1 (optim) and P3 (viewer) build on it.
8. Golden outputs of IK, trajopt and one BVR clip before anything is deleted.
9. REPORT.md, commit on `dev`.

## Intake notes

- Metrics: median ms with warm-up and cuda.synchronize; operation counts; parity error vs pinocchio.
- Baseline: commit 7e2008f numbers in `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/subagents/bench_baseline.md`.
- GPU: FREE only for timings; none free at task start.
- Do not touch: main checkouts, BVR (read-only), anything outside the dev worktrees.
