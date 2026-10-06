# REPORT — P2 model, kinematics, dynamics (2026-10-07, night; interim: dynamics not yet moved)

All work is on `dev`, committed, not pushed. Owner was asleep; decisions made off-hours are listed at the end.

## Outcome in one paragraph

Model values loaded from URDF/MJCF are now exact in float64 (parity with pinocchio 9e-7 → 8e-16). Forward
kinematics was rewritten as one per-joint affine map plus one matmul per tree level, with `[R | p]` matrices
throughout; on one CPU thread Panda FK takes 48 torch operators instead of 246 and runs 2.5–5× faster, frame
Jacobians on a 200-joint tree with gradients 9× faster. The Warp lanes are gone (owner decision Q1). A minimal
collision layer (sphere distances, robot spheres, clearance terms) exists and an obstacle-avoiding trajectory
example runs on the new optimizer. Dynamics (RNEA/CRBA/ABA) still run on the old `Data` path; that is the
remaining P2 item.

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

## M4 Warp removed (commit 688efdb)

- Deleted: 4 kernel/bridge modules, `tests/warp/`, the Warp bench schema test and baselines, the
  `use_warp` parameters, the `warp` extra. 2,206 lines of kernels plus tests.
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
- Tests: `tests/collision/test_collision.py` (7). Not reviewed adversarially yet.

## Not done

- Dynamics (RNEA, CRBA, ABA, centroidal) still run on `Data` and 7-vector poses; the P0 finding that
  200-joint dynamics backward is extremely slow is untouched.
- GPU: no baseline, no Warp record (every GPU was BUSY at every check).
- Further FK gains not taken: quaternion joints pad every joint's feature list to 19 columns when a
  free-flyer exists; the CPU chain uses broadcast matmuls; `so3.to_matrix` costs 45 operators when placements
  require gradients (human shape fitting will hit this).
- `docs/concepts` still describe the old FK stages; the kinematics `CLAUDE.md` is updated.

## Test status

Full suite after all commits: 1775 passed, 1 failed, the failure pre-existing and untouched:
`tests/optim/test_temporal_structure.py::test_banded_and_dense_lm_match_with_outer_reduction_and_activity`
(old optimizer, float32 tolerance).

## Assumptions (made off-hours)

- Regenerating the FK oracle and tightening parity tolerances (independent evidence: pinocchio).
- Deleting the old kind-group FK and the collision stubs (replaced; in git history).
- Removing Warp without a GPU record, on the owner's standing decision; the CPU record stands in.
- Names: `joint_transform`, `frame_transform`, `frame_pose_of`, `RobotSpheres`, `sphere_clearance`.
- The `trash` helper used by the collision agent put the stub files under the main checkout's `trash/`
  (`/home/rikhat.akizhanov/better/BetterRobot/trash/`), not the worktree's.
