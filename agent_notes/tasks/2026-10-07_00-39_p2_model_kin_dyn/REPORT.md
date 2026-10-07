# REPORT — P2 model, kinematics, dynamics (2026-10-07, night)

All work is on `dev`, committed, not pushed. Owner was asleep; decisions made off-hours are listed at the end.

## Outcome in one paragraph

Model values loaded from URDF/MJCF are now exact in float64 (parity with pinocchio 9e-7 → 8e-16). Forward
kinematics was rewritten as one per-joint affine map plus one matmul per tree level, with `[R | p]` matrices
throughout; on one CPU thread Panda FK takes 48 torch operators instead of 246 and runs 2.5–5× faster, frame
Jacobians on a 200-joint tree with gradients 9× faster. The Warp lanes are gone (owner decision Q1). A minimal
collision layer (sphere distances, robot spheres, clearance terms) exists and an obstacle-avoiding trajectory
example runs on the new optimizer. Dynamics (RNEA, CRBA, ABA, centroidal) were rewritten as one world-frame
core with no loop over joints: operator counts drop 3–45× (Panda RNEA 1053 → 379, tree200 CRBA 14,184 → 312),
batch-1 calls are 2–19× faster, and the 200-joint CRBA forward+backward at batch 4096 goes from 823 s to 5.4 s.
Two more float32 leftovers were found on the way; dynamics now match pinocchio to 1e-15.

## M1 Precision (commit 0ebac62)

- `io/parsers/urdf.py` and `mjcf.py` rounded every value to float32. Now float64.
- Evidence (`subagents/m1_precision_before.json` / `m1_precision_after.json`, 16 random configurations, 14 Panda
  and 41 G1 frames): float64 Panda vs pinocchio translation 3.4e-7 → 1.7e-16, rotation 8.8e-7 → 7.8e-16;
  G1 1.1e-8 → 1.4e-16. Analytic vs autodiff Jacobian of the IK term: 1e-7 → 1e-15.
- Consequences: FK oracle `tests/kinematics/fk_reference.npz` regenerated (version 4; the old file held the
  rounded model); pinocchio parity tolerances tightened from 2e-6 to 1e-12 (FK) and 1e-10 (frame Jacobians).
  The old G1 oracle failure disappears; its original cause was not diagnosed separately. The P0 golden IK and
  trajectory files were recorded with the rounded model, so they differ from a fresh run by about 1e-7.
- Viewer: one dtype fix (`viewer/render_modes/urdf_mesh.py`).

## M2–M3 Lean forward kinematics (commits a0c3ab1, a0921d1)

- `kinematics/_fk_lean.py`: every common joint transform is affine in a few features of `q` (sin/cos of a
  revolute angle, products of a normalized quaternion, translation coordinates). A per-joint constant map
  built from topology and axes gives all joint transforms in one batched product; the world chain is one
  matmul per tree level. Rare kinds (planar, translation, helical, composite) go through the per-joint
  dispatch. Placements stay differentiable and may be batched; unbatched placements are folded into the map
  and their matrix form is cached on the `ModelValues` instance (version-checked, skipped under grad,
  `torch.func` and inference mode).
- All FK entry points share this core: `forward_kinematics_state` (matrices) and the `Data` API
  (`_fk_matrix.matrix_core`, which converts to quaternions once). The old kind-group FK was deleted.
- `KinematicState` stores `joint_transform` and `frame_transform` `(B..., n, 3, 4)`; `joint_pose` /
  `frame_pose` are converted on first use; `frame_pose_of(frame_id)` converts one frame.
  `frame_jacobian(model, state, frame, reference=)` builds the world column table once per state from the
  matrices and masks one joint (no per-joint Jacobian tensor, no quaternion round trip).
- Finding: the P0 prototype's single dense feature map costs O(joints²); with it the 200-joint tree at batch 1
  became 24× slower. The per-joint map is linear.

Numbers (CPU, 1 thread, float32, `subagents/m2_bench_table.txt`; raw `benchmarks/results/p2_lean_fk_cpu.json`
vs the P0 baseline `benchmarks/results/0953f28_cpu.json`):

| model | batch | FK before (ms, ops) | FK state now (ms, ops) | speed-up | frame Jacobian before | now | speed-up |
|---|---|---|---|---|---|---|---|
| Panda | 1 | 0.655, 246 | 0.206, 48 | 3.2 | 0.981, 384 | 0.325, 84 | 3.0 |
| Panda | 4096 | 32.9, 246 | 6.6, 48 | 5.0 | 38.8, 384 | 11.6, 84 | 3.4 |
| G1 (free-flyer) | 1 | 0.901, 336 | 0.357, 70 | 2.5 | 1.24, 473 | 0.48, 105 | 2.6 |
| G1 | 4096 | 78.4, 336 | 26.9, 70 | 2.9 | 271, 473 | 136, 105 | 2.0 |
| tree200 | 256 | 25.7, 615 | 7.2, 56 | 3.6 | 180, 752 | 12.8, 91 | 14.0 |
| tree200, forward+backward | 4096 | 7427, 3148 | 835, 219 | 8.9 | 10400, 3504 | 1169, 328 | 8.9 |

Operator counts are exact; timings were taken on a loaded machine (load average 30–135) and vary by ±30%
between runs. The P0 CPU target "Panda FK at batch 1 under 0.25 ms" is met (0.21 ms). Batch-1024 Panda IK
(float32, 1 thread): 2.54 s → 1.67 s with the same reach rates (`subagents/m2_ik_batch1024_1thread.json`).

Review (`subagents/review_lean_fk.md`, clean-context mars_aux): one major defect (non-unit axes from the
programmatic builder gave non-rotation matrices) and two minor ones (stale cache after in-place edit, inference
mode) — all fixed and pinned by a test. The reviewer verified every joint kind against the per-joint dispatch
(7e-16), G1 floating-base Jacobians in all references, `torch.func`, batched placements, and judged the oracle
regeneration justified with no test weakened.

## M4 Warp removed (commits d79e570 and 688efdb)

- Deleted: 4 kernel/bridge modules, `tests/warp/`, the Warp bench schema test and baselines, the
  `use_warp` parameters, the `warp` extra. 2,206 lines of kernels plus tests. The file deletions landed in the
  collision commit d79e570 (two agents worked in one tree; the collision commit swept the other's staged
  deletions) and the `use_warp` code paths in 688efdb. Splitting them means rewriting `dev` history: owner's call.
- Record before removal (`subagents/m4_warp_record_cpu.json`, CPU only, no GPU was free all night): forward-only
  at batch 4096 Warp was 3–5× faster than the new torch FK (Panda 2.3 vs 6.6 ms; G1 5.9 vs 27.8 ms); with
  gradients Warp was 4× slower (106 vs 23 ms; 317 vs 75 ms); at batch 1 torch wins both. Whether Warp used
  more than one CPU thread was not controlled. To record on a GPU later: `git checkout a0921d1` and run
  `subagents/m4_warp_record.py <out.json> cuda` on a FREE GPU.
- Left in place: the flat `ModelStructure` index tensors that were added for device kernels (some are used by
  `validate_consistency` and tests). `docs/` cleaned by a separate pass (see commit after this report).

## M5 Collision (commit d79e570)

- `better_robot.collision`: `sphere_sphere`, `sphere_capsule`, `sphere_halfspace`, `sphere_box`, `sphere_sdf`
  (signed, negative = penetration, batched, differentiable with finite gradients at contact); `RobotSpheres`
  (spheres attached to frames, `centers(state)`), `self_collision_pairs`, `self_distances`.
  The old stubs (`Sphere`, `Capsule`, `RobotCollision`, …) were removed.
- `better_robot.terms`: `sphere_clearance`, `self_clearance` (hinge `clamp(margin − d, 0)` or plain for
  `solve_constrained`). Autodiff Jacobians, checked by `check_jacobian`.
- `examples/optim2/06_obstacle_trajectory.py` (Panda, 30 knots, sphere obstacle): straight path penetrates by
  9.2 cm; soft penalty reaches +1.6 cm clearance in 63 LM steps (0.74 s on 4 threads); hard inequality reaches
  +2.0 cm (= margin) in 229 steps (2.1 s). Path length 1.80 → 2.21 rad.
- Tests: `tests/collision/test_collision.py` (8 after the review). Review: see M7.

## M6 Dynamics on the state (commit 4503509)

- `dynamics/_world.py` is the only implementation. Everything is in the world frame: the shared Jacobian column
  table `C (N, 6, nv_full)` (the one `frame_jacobian` uses), each joint's world rotation and origin, the packed
  inertias. Sums over ancestors (body velocity and acceleration) and over subtrees (wrenches, composite inertias)
  are products with constant support matrices memoised on the structure, so there is no Python loop over joints
  and the operator count is independent of the joint count. Wrenches are formed in the joint frames (where the
  inertia is constant) with `3x3` rotations and cross products — no `6x6` adjoints. CRBA: `M[r, c] = C_r · I^c C_c`
  with the composite inertia of the deeper column's subtree, mirrored below the diagonal. ABA: Cholesky solve of
  `M ddq = τ − b` (the articulated-body recursion is sequential in the tree depth and has no advantage in batched
  torch for `nv` up to a few hundred; with mimic joints the projected system is also the only correct form).
  Centroidal map: the CRBA momentum columns moved to the centre of mass.
- State API (`dynamics/state.py`): `inverse_dynamics(model, state, v, a, fext=)`, `mass_matrix`,
  `forward_dynamics`, `centroidal_map`, `center_of_mass`; the world-frame quantities are built once per state and
  memoised on it. The `Data` API (`rnea`, `crba`, `aba`, `ccrba`) runs the same core after its own FK and keeps
  its result fields (`joint_forces` etc. converted to the joint frames).
- Jacobian columns are now built per real column from the owner's rotation and origin; the old padded
  `(njoints, max_nv)` grid made a free-flyer pad every joint to six columns (G1: 246 padded columns for 35).
- Precision: `subagents/m6_dynamics_precision.json`. Before this milestone RNEA differed from pinocchio by 5e-8
  (Panda) and 4e-7 (G1) relative, ABA by 3e-6, with the old test comment blaming the float32 parser. The actual
  causes: the default gravity was a float32 constant (`-9.8100004196167`) and `random_configuration` sampled
  quaternions in float32 and cast them (unit norm off by 4e-9). Fixed; RNEA/CRBA/ABA now 1e-15 (Panda) and
  1e-15 / 9e-12 (G1), tolerances tightened from 1e-5 to 1e-10 in the four parity test files (184 tests pass).
  Panda's centroidal map differs from pinocchio by 2e-2 because BetterRobot counts the mass of `panda_link0`
  (attached to the fixed root) and pinocchio drops it into the universe — documented convention, unchanged.
- Tests: `tests/dynamics/test_state_dynamics.py` (forward/inverse round trip with `fext` and `(2, 3)` batch axes,
  mass matrix equals the acceleration Jacobian of the torque — two different formulas —, gradcheck through the state).

Numbers (CPU, 1 thread, float32, `subagents/m6_bench_table.txt`; raw `benchmarks/results/p2_world_dynamics_cpu.json`
vs `0953f28_cpu.json`):

| algo | model | batch 1 fwd before → after (ms) | batch 1 fwd+bwd | batch 4096 fwd | batch 4096 fwd+bwd | ops |
|---|---|---|---|---|---|---|
| RNEA | Panda | 2.11 → 1.07 | 7.0 → 2.5 | 53 → 52 | 165 → 133 | 1053 → 379 |
| CRBA | Panda | 1.86 → 0.90 | 6.7 → 2.1 | 92 → 57 | 233 → 138 | 910 → 307 |
| ABA | Panda | 6.09 → 1.30 | 16.6 → 3.2 | 200 → 81 | 468 → 211 | 3018 → 444 |
| RNEA | G1 | 4.81 → 1.15 | 17.9 → 2.9 | 174 → 217 | 670 → 498 | 2648 → 396 |
| CRBA | G1 | 4.50 → 1.01 | 19.1 → 2.4 | 300 → 185 | 1173 → 491 | 2551 → 326 |
| ABA | G1 | 6.91 → 1.40 | 29.2 → 3.6 | 359 → 357 | 956 → 792 | 3417 → 458 |
| RNEA | tree200 | 20.6 → 1.42 | 86.6 → 3.3 | 1324 → 1204 | 37,496 → 2,365 | 12,366 → 382 |
| CRBA | tree200 | 23.1 → 1.23 | 116 → 2.9 | 2596 → 2326 | 822,637 → 5,381 | 14,184 → 312 |
| ABA | tree200 | 34.4 → 1.99 | 152 → 5.5 | 2150 → 4583 | 25,973 → 17,259 | 17,367 → 444 |

Reading: batch 1 and backward passes are dominated by operator count and gain 2–150×. Forward passes at batch
4096 on one CPU thread are memory-bound — every operator streams an `(N, njoints, 6)` tensor, whereas the old
per-joint loop worked on cache-resident `(N, 6)` slices — so they are about equal; the 200-joint ABA at 4096 is
slower because the dense `(4096, 200, 200)` mass matrix (655 MB) plus its Cholesky replace the O(n) recursion.
Timings vary ±30% on this loaded machine; operator counts are exact.

## M7 Reviews (commit 72de682)

Two clean-context mars_aux reviews, recorded in `subagents/review_dynamics.md` and `subagents/review_collision.md`.

- Dynamics, major 1: float32 accuracy degraded with the robot's distance from the world origin (G1 at 100 m:
  mass-matrix error 6e-3, Cholesky failed). Fixed by taking all world quantities about the first joint's position
  per sample (torques, mass and centroidal matrices are invariant); after the fix 3e-6 at 100 m, the matrix is
  positive definite. Pinned by `test_float32_accuracy_does_not_depend_on_distance_from_the_world_origin`.
- Dynamics, major 2: the per-state memo held the inertias, so in-place edits of `body_inertias` or reusing a state
  with another model gave stale torques but fresh mass matrices. Fixed: the memo holds only model-independent
  geometry; inertias come from the model on every call with a version-keyed cache. Pinned by
  `test_state_reuse_sees_edited_inertias_and_other_models`.
- Dynamics, minor: `joint_forces[0]` had started to include `fext[0]`; restored. Open: composite joints have a
  configuration-dependent subspace that neither the old nor the new dynamics handles, and nothing tests them against
  pinocchio.
- Collision, major: `self_collision_pairs` treated fixed joints as moving joints, so Panda's link7 and hand (rigidly
  attached) were a pair that penetrated in every configuration, making a self-clearance constraint unsatisfiable.
  Fixed by mapping joints to the nearest ancestor with coordinates; pinned by `test_pairs_ignore_fixed_joints`.
- Collision, minor: the reference page and four tracked autodoc pages still described the removed stubs; rewritten
  and regenerated (new pages for `collision.distance`, `robot_spheres`, `dynamics.state`, `kinematics.state`,
  `optim2.*`, `terms.*`).
- Everything else the reviewers checked held: distance functions to 1e-12 against brute force, gradchecks, term
  shapes and Jacobians, the example's numbers, the dynamics parity, gradients w.r.t. model values, vmap/jacrev,
  re-timed benchmark rows.

## Not done

- Dynamics at large CPU batches are memory-bound; fusing the elementwise chains (`torch.compile`) or chunking the
  batch would be the next step if that regime matters. No GPU timing yet.
- GPU: no baseline, no Warp record (every GPU was BUSY at every check).
- Further FK gains not taken: quaternion joints pad every joint's feature list to 19 columns when a
  free-flyer exists; the CPU chain uses broadcast matmuls; `so3.to_matrix` costs 45 operators when placements
  require gradients (human shape fitting will hit this).
- `docs/concepts` still describe the old FK stages; the kinematics `CLAUDE.md` is updated.

## Test status

Full suite after all commits (72de682): 1785 passed, 2 skipped, 1 failed, the failure pre-existing and untouched:
`tests/optim/test_temporal_structure.py::test_banded_and_dense_lm_match_with_outer_reduction_and_activity`
(old optimizer, float32 tolerance).

## Assumptions (made off-hours)

- Dynamics state API names: `inverse_dynamics`, `mass_matrix`, `forward_dynamics`, `centroidal_map`,
  `center_of_mass` (module `dynamics.state`); the old `rnea`/`crba`/`aba` names stay on the `Data` API.
- ABA is a Cholesky solve on the mass matrix (no articulated-body recursion); `Data.joint_acceleration_local` is
  still filled.
- World dynamics quantities are referenced to the first joint's position; `WorldKinematics.origin` records it.
- Dynamics parity tolerances tightened to 1e-10 and the random-configuration sampler changed to the model dtype
  (changes which random configurations tests draw; all pass).
- Reviewer-found pre-existing issues left open: composite joints in dynamics, the Warp deletions sitting in the
  collision commit.
- Regenerating the FK oracle and tightening parity tolerances (independent evidence: pinocchio).
- Deleting the old kind-group FK and the collision stubs (replaced; in git history).
- Removing Warp without a GPU record, on the owner's standing decision; the CPU record stands in.
- Names: `joint_transform`, `frame_transform`, `frame_pose_of`, `RobotSpheres`, `sphere_clearance`.
- The `trash` helper used by the collision agent put the stub files under the main checkout's `trash/`
  (`/home/rikhat.akizhanov/better/BetterRobot/trash/`), not the worktree's.

## Addendum (2026-10-07 09:30 Dubai): GPU numbers

GPU 4 (RTX PRO 6000 Blackwell) was FREE this morning, so the benchmark ran once on CUDA
(`benchmarks/results/p2_world_dynamics_cuda.json`, commit 9f85980, float32, full table in
`subagents/m6_bench_table_cuda.txt`). There is no pre-P2 GPU file, so this is a baseline for the new code, not
a before/after. The shape is as the operator counts predict: every call is launch-bound and flat in batch size up
to 256 (Panda RNEA 2.6 ms at batch 1, 2.9 ms at 4096; CRBA 2.0 → 2.2 ms; ABA 3.2 → 3.5 ms), and only tree200 at
batch 4096 becomes compute-bound (RNEA 15.5 ms, CRBA 13.1 ms, ABA 28.4 ms; forward+backward 36 / 43 / 98 ms).
Batch-1 GPU latency is 1.3–3.4 ms per algorithm, i.e. per-op launch overhead over ~200–400 ops; a CUDA graph
or `torch.compile` would be the next lever, not the formulation.
