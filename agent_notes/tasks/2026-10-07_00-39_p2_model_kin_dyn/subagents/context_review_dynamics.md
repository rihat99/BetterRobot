# Context: adversarial review of the world-frame dynamics core

Repo: `/home/rikhat.akizhanov/better/dev/BetterRobot` (git worktree, branch `dev`, commit 4503509). Read-only task:
you do not edit files, commit, or push. Run Python with `CUDA_VISIBLE_DEVICES="" uv run python ...` from the repo
root; scratch scripts go to `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/review_dynamics/`
(create it). No new dependencies.

## What changed (commit 4503509, `git show --stat 4503509`)
- `src/better_robot/dynamics/_world.py` (new): the only dynamics implementation. World Jacobian columns
  `C (N, 6, nv_full)`, per-joint world rotation/origin, packed inertias. RNEA, CRBA, centroidal as support-matrix
  products (`DynamicsPlan`: `support (nj, nj)`, `column_support (nj, nv_full)`, `upper`/`lower (nv_full, nv_full)`).
- `src/better_robot/dynamics/state.py` (new): public `inverse_dynamics(model, state, v, a, fext=)`,
  `mass_matrix`, `forward_dynamics` (Cholesky on M), `centroidal_map`, `center_of_mass` on a `KinematicState`
  (`src/better_robot/kinematics/state.py`, `forward_kinematics_state(model, q)`).
- `rnea.py`, `crba.py`, `aba.py`, `centroidal.py`: the `Data` API now calls the same core; ABA no longer runs the
  articulated-body recursion (it is M ddq = tau - b with Cholesky).
- `src/better_robot/kinematics/_jacobian_columns.py`: `world_column_table_from_matrices` builds each column from its
  owner joint (no padded `(njoints, max_nv)` grid); plan field `column_subspace` replaces `column_gather`.
- `src/better_robot/data_model/model_values.py`: `packed_inertias_to_6x6` split into `inertia_about_com` +
  `spatial_inertia_matrix` (same output).
- `src/better_robot/io/ir.py`: default gravity now float64; `joint_models/free_flyer.py`, `spherical.py`:
  `random_configuration` samples in the limits' dtype.
- Tests: `tests/dynamics/test_state_dynamics.py` (new), pinocchio parity tolerances tightened to 1e-10 in
  `tests/test_pinocchio/test_{rnea,crba,aba}_matches_pinocchio.py`, `test_rnea_advanced_joints.py`.
- Docs: `src/better_robot/dynamics/CLAUDE.md`, `docs/concepts/dynamics.md`.

## Claims to attack
1. Correctness: RNEA/CRBA/ABA/CCRBA equal pinocchio to 1e-15 relative in float64 on Panda (mimic unconstrained, as in
   `tests/test_pinocchio/conftest.py::panda_both`) and G1 with free-flyer (`m6_dynamics_precision.json` in this
   folder; script: the main agent's scratch `dyn_precision.py`, rewrite your own). Cases I did NOT verify against
   pinocchio: spherical joints with fext, planar/prismatic/helical joints in dynamics, batched (per-sample) body
   inertias and placements (`tests/dynamics/test_batched_values.py` only checks consistency with a loop), mimic
   joints (`tests/dynamics/test_mimic.py` only checks internal consistency), models whose root joint is not joint 1,
   `fext` on the free-flyer base, gravity with a nonzero angular part, float32 behaviour (the world formulation
   subtracts large lever-arm terms: measure float32 error vs float64 for G1 placed 10 m and 100 m from the origin,
   compare with the old local-frame code at commit bbbb669 via `git archive bbbb669 src | tar -x -C <scratch>` and
   `PYTHONPATH=<scratch>/src`).
2. `joint_forces`, `joint_velocity_local`, `joint_acceleration_local` in `Data` results keep their old meaning
   (compare with the old code at bbbb669 on random inputs, G1 with fext).
3. Autograd: gradients through `inverse_dynamics` / `mass_matrix` / `forward_dynamics` w.r.t. q, v, a, tau, fext and
   model values (`body_inertias`, `joint_placements`, `gravity`) are correct (gradcheck in float64) and work under
   `torch.func.vmap` / `jacrev` and `torch.compile` (report, do not fix).
4. Memoisation: `state._memo["world_kinematics"]` is built once per state from `model.values.body_inertias`; a state
   reused with a different model, or after in-place edits of `body_inertias`, gives stale results — is that
   documented and acceptable, or a trap? Compare with how `kinematics/_fk_lean.placement_matrices` versions its cache.
5. Performance claims in `m6_bench_table.txt` (before = `benchmarks/results/0953f28_cpu.json`, after =
   `p2_world_dynamics_cpu.json`): operator counts and that batch-1 and backward are 2-150x faster while forward at
   batch 4096 on one CPU thread is about the same (memory-bound). Re-time two or three rows yourself with
   `benchmarks/run.py --algos rnea,crba,aba --models panda,g1 --batches 1,4096 --threads 1 --out <scratch>/x.json`
   (CPU; the machine is loaded, expect ±30%). Check the operator counts are plausible from the code.
6. Design: anything over-engineered, duplicated, or a simpler formulation (e.g. is the two-pass `torch.where`
   symmetric completion in `mass_matrix_full` correct for multi-dof joints — free-flyer 6x6 block, spherical 3x3 —
   and for models with several multi-dof joints?). Names and shapes in docstrings match the code?

## Report (final message, ≤ 60 lines)
Ranked findings: severity (major/minor/nit), file:line, what is wrong, how you showed it (command + numbers), proposed
fix. Then one line per claim above: confirmed / refuted / not checked. Then anything you were unable to run.
