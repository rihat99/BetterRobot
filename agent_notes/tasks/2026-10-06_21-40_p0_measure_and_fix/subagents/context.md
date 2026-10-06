# Context for P0 subagents (read first)

Repo: /home/rikhat.akizhanov/better/dev/BetterRobot (git worktree, branch `dev`). BetterRobot is a
PyTorch robotics library (batched, differentiable FK, Jacobians, RNEA, CRBA, ABA).
Run Python only as `CUDA_VISIBLE_DEVICES="" uv run python <script>` from the repo root (CPU only:
all GPUs are busy; do not use a GPU). Environment has torch 2.13, pinocchio (`import pinocchio`),
warp, robot_descriptions. Scripts in files, never `python -c`.

Rules: never `rm` (leave files; report what should be trashed), never `git commit/push/stash`,
no pip/uv installs, do not edit files outside what your brief names, do not write tests unless
the brief says so, do not spawn agents. Type hints + one-line-start docstrings on public
functions, tensor shapes documented, ruff line length 120. You have about 50k tokens: read only
what you need (grep, then small ranges), and write your result note early, then refine it.

Useful entry points (verify by grep before use):
- `import better_robot as br`; `br.load(urdf_path)` -> Model (float32). `model.to(dtype=, device=)`.
- Panda: `from robot_descriptions import panda_description; panda_description.URDF_PATH`.
  G1: `from robot_descriptions import g1_description` (see tests/kinematics/test_fk_regression.py `_load_g1`).
- `br.forward_kinematics(model, q)` -> Data with `joint_pose_world` (B, njoints, 7) [x y z qx qy qz qw].
  Implementation: src/better_robot/kinematics/forward.py, src/better_robot/kinematics/_fk_matrix.py.
- Dynamics: src/better_robot/dynamics/{rnea,crba,aba}.py. Jacobians: src/better_robot/kinematics/jacobian.py.
- `model.random_configuration(generator)` -> (nq,) one sample.
- Synthetic trees via `better_robot.io.parsers.programmatic.ModelBuilder` + `better_robot.io.build_model.build_model`
  (example: tests/data_model/test_model_data.py). A working builder of random 25- and 223-joint
  trees and an operation counter using torch.profiler exists in
  agent_notes/tasks/2026-10-06_20-13_architecture_plan/subagents/proto_level_fk.py (reuse it).
- Earlier measurements: agent_notes/tasks/2026-10-06_20-13_architecture_plan/subagents/bench_baseline.md
  and proto_level_fk.md. Known: FK on a 25-joint tree costs about 219 operator calls forward.

Redesign direction (owner approved): one torch implementation, Warp will be removed, main batch
size of interest is 4096, every algorithm must stay differentiable.
