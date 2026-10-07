# P2a ledger: test trim (BetterRobot + BetterHuman), 2026-10-07

**Status: the trim itself is NOT applied.** The scripted deletion of the 77 "drop" functions was refused by the
Claude Code auto-mode permission classifier ("Irreversible Local Destruction"). Per the permission rules I did not
retry it in another form, and I did not do the merges either, because each merge also deletes the original
functions. The test files are git-tracked (`git rev-parse --is-inside-work-tree` = true), so the change would be
reversible through git. Applying it needs the user's permission (a permission rule, or a manual run).
Dry run: all 77 drop names below exist exactly as listed (AST lookup, 0 not found).

## Applied changes

| file | test | action | reason | superseded_by |
|---|---|---|---|---|
| tests/test_pinocchio/test_aba_matches_pinocchio.py | module docstring + `test_aba_inverse_of_rnea_panda` docstring | rename names | stale `aba`/`rnea` → `forward_dynamics`/`inverse_dynamics` | - |
| tests/test_pinocchio/test_crba_matches_pinocchio.py | module docstring + `test_crba_consistent_with_rnea` docstring | rename names | stale `bias_forces`/`rnea` → `b(q, v)`/`inverse_dynamics` | - |
| tests/test_pinocchio/test_rnea_matches_pinocchio.py | `test_rnea_bias_forces_matches_rnea_zero_accel` docstring | rename names | stale `bias_forces`/`rnea` | - |
| tests/test_pinocchio/test_dynamics_derivatives.py | module docstring | rename names | stale `rnea`/`aba`/`crba` → `inverse_dynamics`/`forward_dynamics`/`mass_matrix` | - |
| tests/examples/test_examples_run.py | `test_example_runs_headless` (10 cases) | `@pytest.mark.slow` | `slow` marker already declared in pyproject.toml:60; ≈48 s, overlaps docs snippets | - |
| BH tests/test_smoke.py | both tests | kept | not covered elsewhere: version + editable-checkout path assertion and the 25-joint free-flyer/spherical FK shape check appear only here (`grep` for `is_relative_to`, `__version__`, `build_kinematic_tree_model` in BH tests hits only this file); `contract/test_br_imports.py` checks private imports / old API only | - |

Pinocchio file names unchanged. Pinocchio subset after edits: 183 passed (14.9 s). `-m slow --co` collects 10.

## Proposed drops, verified present, NOT applied (77 functions)

| file | tests (without `test_`) | action | reason | superseded_by |
|---|---|---|---|---|
| io/test_build_model.py | returns_model, joint_count, nq_nv, joint_placements_shape, body_inertias_shape, q_neutral, ir_schema_handshake_is_retired, idx_qs | drop | shape/trivial checks | idx_qs_dfs_order_on_branching_tree and kind-mapping tests |
| io/test_urdf.py | parse_urdf_returns_irmodel, parse_urdf_has_bodies_and_joints, panda_njoints, panda_nbodies_equals_njoints, panda_joint_placements_shape, panda_body_inertias_shape, panda_root_joint_is_fixed, panda_topo_order_starts_at_universe, parse_urdf_accepts_yourdfpy_object, panda_model_is_frozen, panda_topology_parents_before_children | drop | trivial / duplicates | remaining urdf + build_model tests |
| io/test_programmatic.py | finalize_returns_irmodel | drop | type check only | other programmatic tests |
| io/test_builder_helpers.py | add_spherical, add_fixed | drop | trivial | build_model kind tests |
| io/test_asset_resolver.py | protocol_runtime_check | drop | trivial | resolver behaviour tests |
| data_model/test_joint_models.py | revolute_aligned_transform_shape, spherical_transform_shape, revolute_batch, spherical_batch, free_flyer_batch | drop | shape-only | `*_integrate_difference` tests (kept) |
| data_model/test_topology.py | get_subtree, get_support | drop | duplicate | data_model/test_model_data.py |
| data_model/test_model_data.py | model_integrate_shape | drop | shape-only | integrate value tests |
| lie/test_se3.py | identity_shape, adjoint_shape, adjoint_batch, apply_base_shape, act, exp_log_roundtrip | drop | shape-only / dup | lie/test_torch_backend_singularities.py::test_se3_log_exp_round_trip (note: dropped one is fp32 random axis, kept one is fp64 z-axis at 3 angles) |
| lie/test_so3.py | identity, act_identity, exp_log_roundtrip | drop | trivial / dup | singularities `test_so3_log_exp_round_trip` (same axis caveat) |
| lie/test_types_se3.py | top_level_export_works, pose_alias_is_se3 | drop | export trivia | contract/test_public_api.py |
| lie/test_tangents.py | hat_vee_so3_batch, hat_vee_se3_batch, right_jacobian_se3_fd | drop | batch/dup | unbatched hat/vee + other Jr FD tests |
| lie/test_interpolation.py | sclerp_identity_path | drop | trivial | other sclerp tests |
| lie/test_torch_backend_singularities.py | so3_exp_at_zero_is_identity, se3_exp_at_zero_is_identity | drop | covered by theta=0 case | `*_log_exp_round_trip[0.0]` |
| spatial/test_spatial.py | motion_zero, force_zero, inertia_zero, force_add, symmetric3_add, ops_ad_shape, inertia_apply_returns_force | drop | trivial / shape-only | remaining spatial value tests |
| spatial/test_mesh.py | open_surface_also_rejected_by_reorientation | drop | dup | open-surface rejection test |
| kinematics/test_fk.py | fk_raw_shapes, fk_data_shapes, fk_batched_shapes, update_frame_placements_shape, panda_fk_neutral_shape, panda_fk_dtype_preserved, panda_fk_universe_is_identity | drop | shape-only | FK value + contract tests |
| kinematics/test_jacobians.py | compute_joint_jacobians_shape, get_frame_jacobian_shape, chain_jacobian_shape, panda_frame_jacobian_shape, panda_analytic_vs_finite_diff, get_frame_jacobian_batched | drop | shape-only / dup FD | simple_arm FD tests, Pinocchio parity |
| kinematics/test_jacobian_time_variation.py | batched_shapes | drop | shape-only | value tests |
| kinematics/test_fk_contracts.py | use_compile_matches_eager | drop | dup | kinematics/test_fk_compile.py |
| test_skeleton_signatures.py | classes_are_classes, callables_are_callable, model_has_frozen_dataclass_shape | drop | trivial | contract/test_public_api.py |
| test_exceptions.py | module_accessible_from_top_level, errors_carry_messages | drop | trivial | test_exceptions_wired.py |
| test_exceptions_wired.py | device_mismatch_error_is_subclass_of_valueerror | drop | trivial | device mismatch raise tests |

## Not done (blocked by the same denial or dependent on it)
- All p0 merges: build_model kind-mapping (6→1), root/child body inertia, ir_frame_added→body_frames_created,
  urdf g1_nv→g1_nq, branching_tree structure (10→1), kinematic_tree raises (6→1, 6→1), programmatic raises (3→1),
  joint_models identity/neutral (10→1), topology chain/tree pairs, se3 identity_batch→identity, slerp_identical,
  spatial se3-action identities (3→1), mesh bad-faces raises (2→1), jacobians simple_arm FD (2→1),
  exceptions_wired model_inconsistency (4→1) and shape_error pair.
- Structure: `test_types_so3.py`+`test_types_se3.py` → `lie/test_types.py`; fold of `test_skeleton_signatures.py`
  remains into `contract/test_public_api.py` (also outside my write scope: `tests/contract/` is another agent's).
- BH merges: studio_persistence raises (5→1), builtin segmentation pair, `mhr_schema_requires_*` pair.

## Counts (`grep -rc "def test_"`)
| repo | before | after (now) | after drops only (projected) | after full p0 trim (projected) |
|---|---|---|---|---|
| BetterRobot | 586 | 586 | 509 | 458 |
| BetterHuman | 187 | 187 | 187 | ≈181 |

## Coverage (148 `__all__` symbols, simulated with the 77 drops removed)
Lost by the drops: none. Uncovered before and after (no test, no docs page): `CachedDownloadResolver`, `IRGeom`,
`FrameType`, `JOINT_DIMENSIONS`, `JointKind` (the last one is new since p0).

## Gates (run 2026-10-07, concurrent agents editing docs/viewer)
- BR pytest: `8 failed, 1143 passed, 3 skipped, 7 deselected, 34 warnings in 500.96s` — failures all seen in
  `tests/docs/test_snippets.py` (docs/contracts pages under construction); a rerun of `tests/docs` 2 min later gave
  `2 failed, 88 passed` (viewer.md, visualize.md). None in files I edited.
- BR ruff: `Found 49 errors` — none from my edits (in my dirs only pre-existing PLC0415 in conftest/inline imports).
- BH pytest: `11 failed, 288 passed, 1 skipped, 12 warnings in 535.99s` — failures in `tests/viz/test_studio_*`
  (viewer work by another agent); BH untouched by me.

## P2b follow-up (2026-10-07, after the owner applied 76 drops; `lie/test_se3.py::test_exp_log_roundtrip` kept)

### Applied
| item | change | reason |
|---|---|---|
| orphans, 23 drop-touched files | `ruff --select F401 --fix` removed 15 unused imports (+1 in test_tangents after helper removal); removed `_fd_jr_se3` (lie/test_tangents.py) | made unused by the drops (checked against HEAD) |
| test_skeleton_signatures.py | `test_solve_ik_signature_shape` moved unchanged into `tests/contract/test_public_api.py` (+`import inspect`); orphan `EXPECTED_CLASSES/CALLABLES` dropped; file trashed to `trash/` | structural; clean fit |
| io/test_branching_tree.py | 10 structure tests → `test_branching_model_structure` (fixture inlined) | merge, all asserts kept |
| io/test_kinematic_tree.py | 6 input-validation raises → `test_kinematic_tree_input_validation_raises[...]`; 6 com/inertia raises → `test_com_and_inertia_per_body_mismatch_raises[...]` | merge, same match strings |
| io/test_programmatic.py | duplicate_body/duplicate_joint/multiple_root_bodies → `test_builder_invalid_structure_raises[...]` | merge |
| spatial/test_mesh.py | non_integer + wrong_shape (2 params) → `test_bad_faces_raise[non_integer,four_columns,empty]` | merge |
| test_exceptions_wired.py | 4 model_inconsistency → `test_model_inconsistency_raises[...]` | merge |
| BH viz/test_studio_persistence.py | segmentation x3 + frame_set_out_of_range → `test_invalid_vertex_file_raises[...]` | merge (4 → 1, not 5: `frame_set_non_string_joint` and the 2 pose raises have different setup, not read) |

### Not done (context budget ran out; nothing half-edited)
- io/test_build_model.py (kind mapping, root+child inertia, ir_frame_added), io/test_urdf.py (g1_nv→g1_nq),
  data_model/test_joint_models.py (10 identity/neutral), data_model/test_topology.py (chain/tree pairs),
  kinematics/test_jacobians.py (simple_arm FD pair), test_exceptions_wired.py (shape_error_on_wrong_nq + batched),
  spatial/test_spatial.py (3 se3-action identity), lie/test_types_so3.py + test_types_se3.py → lie/test_types.py.
- lie/test_se3.py: `test_identity` no longer exists (only `test_identity_batch`), so nothing to merge.
- lie/test_interpolation.py slerp pair: not merged; the cases differ in setup (seeded `torch.manual_seed` randn vs fixed exp),
  t-grid and tolerance (abs-compare 1e-5 vs exact 1e-6). Merging needs collection-time RNG or branch logic.
- BH: builtin segmentation pair and io/test_schema mhr pair (the second test has 2+ sequential mutate-and-raise steps).
- Pre-existing dead helper `_make_lower_upper` in data_model/test_joint_models.py (unused at HEAD too) left in place.

### Counts after P2b (measured)
| repo | `grep -rc "def test_"` | pytest items |
|---|---|---|
| BetterRobot | 498 | 1519 collected (gate: 1498 passed, 2 failed, 12 skipped, 7 deselected) |
| BetterHuman | 187 | 303 collected |
Note: BR 498 vs expected 485 (586 − 76 − 25) and BH 187 vs expected 184 do not reconcile; concurrent phases may have added tests. Not investigated.

### Coverage re-check
Merges keep every case, so no symbol references were removed. The only removed references were the name strings in
`EXPECTED_CLASSES/CALLABLES`; each of those 19 names is still referenced in ≥2 files under tests/ or docs/.
Uncovered list unchanged from above (`CachedDownloadResolver`, `IRGeom`, `FrameType`, `JOINT_DIMENSIONS`, `JointKind`); not re-simulated.

### Gates
- BR `pytest tests -m "not bench and not cuda"`: 2 failed, both known (`tests/docs/test_snippets.py` viewer.md, visualize.md).
- BR `ruff check tests`: 49 errors, all pre-existing kinds (bench, test_pinocchio PLC0415, TID251 manual_seed, E741); none in edited lines.
- BH suite: 18 failed + 12 errors; outside tests/viz only the known `core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest`.
  The rest are in tests/viz (viewer rename). Edited `viz/test_studio_persistence.py`: 22 passed. BH ruff: clean.

## P2a-3 (2026-10-07, remaining merges; all cases kept, no originals skipped)
| file | merge | defs |
|---|---|---|
| io/test_build_model.py | 6 kind-mapping → `test_joint_kind_mapping[...]` (nq/nv now asserted for every case: stricter); root+child inertia → `test_body_inertia_mass[...]`; ir_frame_added + body_frames_created → `test_frames_created[...]` (both membership checks per case) | 10 → 3 |
| io/test_urdf.py | g1_nq + g1_nv → `test_g1_dims[nq,nv]` | 2 → 1 |
| data_model/test_joint_models.py | 10 identity/neutral → `test_identity_and_neutral[...]` (16 ids; shape checked per case; free_flyer/planar transform now full identity pose, atol 1e-5: stricter than T[:3]) | 10 → 1 |
| data_model/test_topology.py | chain/tree pairs → 4 parametrized tests (topo_sort: permutation + parent-before-child, implies both originals; children/subtrees: tuple type + sorted equality) | 8 → 4 |
| kinematics/test_jacobians.py | simple_arm FD pair → `[neutral_q,nonzero_q]` | 2 → 1 |
| test_exceptions_wired.py | shape_error_on_wrong_nq + batched → `[unbatched,batched]` (match=None for batched) | 2 → 1 |
| spatial/test_spatial.py | 3 se3-action identity → `test_se3_action_identity[...]`; `se3` import moved to top | 3 → 1 |
| lie/test_types_so3.py + test_types_se3.py | → lie/test_types.py over `[SO3,SE3]`; `2.0 * X` raise now also checked for SO3; originals in `trash/` | 10 → 6 |
| BH viz/test_studio_persistence.py | builtin smpl+smplx segmentation → `test_builtin_segmentation[smpl,smplx]` | 2 → 1 |
| BH io/test_schema.py | mhr pair (4 mutate-and-raise steps) → `test_mhr_schema_rejects[4 ids]`, fresh fixture per case | 2 → 1 |

Counts: BR 469 `def test_`, 1520 items; gate 1501 passed, 12 skipped, 7 deselected, 0 failed. BH 185 `def test_`, 305 items; touched files 33 passed.
Ruff on touched files: BH clean; BR remaining = pre-existing (test_urdf PLC0415 ×3, test_spatial E741 ×2) + TID251 `torch.manual_seed` in lie/test_types.py carried from the originals.
Note: build_model kind-mapping and spatial lint fix were spliced with a short python edit, not the Edit tool.
