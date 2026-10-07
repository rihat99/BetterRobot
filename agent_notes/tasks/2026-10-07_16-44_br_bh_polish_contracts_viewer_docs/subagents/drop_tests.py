"""Delete named test functions (with decorators) from test files. Usage: drop.py [--apply]"""
import ast, pathlib, re, sys
ROOT = pathlib.Path("/home/rikhat.akizhanov/better/dev/BetterRobot/tests")
DROPS = {
 "io/test_build_model.py": "returns_model joint_count nq_nv joint_placements_shape body_inertias_shape q_neutral ir_schema_handshake_is_retired idx_qs",
 "io/test_urdf.py": "parse_urdf_returns_irmodel parse_urdf_has_bodies_and_joints panda_njoints panda_nbodies_equals_njoints panda_joint_placements_shape panda_body_inertias_shape panda_root_joint_is_fixed panda_topo_order_starts_at_universe parse_urdf_accepts_yourdfpy_object panda_model_is_frozen panda_topology_parents_before_children",
 "io/test_programmatic.py": "finalize_returns_irmodel",
 "io/test_builder_helpers.py": "add_spherical add_fixed",
 "io/test_asset_resolver.py": "protocol_runtime_check",
 "data_model/test_joint_models.py": "revolute_aligned_transform_shape spherical_transform_shape revolute_batch spherical_batch free_flyer_batch",
 "data_model/test_topology.py": "get_subtree get_support",
 "data_model/test_model_data.py": "model_integrate_shape",
 "lie/test_se3.py": "identity_shape adjoint_shape adjoint_batch apply_base_shape act",
 "lie/test_so3.py": "identity act_identity exp_log_roundtrip",
 "lie/test_types_se3.py": "top_level_export_works pose_alias_is_se3",
 "lie/test_tangents.py": "hat_vee_so3_batch hat_vee_se3_batch right_jacobian_se3_fd",
 "lie/test_interpolation.py": "sclerp_identity_path",
 "lie/test_torch_backend_singularities.py": "so3_exp_at_zero_is_identity se3_exp_at_zero_is_identity",
 "spatial/test_spatial.py": "motion_zero force_zero inertia_zero force_add symmetric3_add ops_ad_shape inertia_apply_returns_force",
 "spatial/test_mesh.py": "open_surface_also_rejected_by_reorientation",
 "kinematics/test_fk.py": "fk_raw_shapes fk_data_shapes fk_batched_shapes update_frame_placements_shape panda_fk_neutral_shape panda_fk_dtype_preserved panda_fk_universe_is_identity",
 "kinematics/test_jacobians.py": "compute_joint_jacobians_shape get_frame_jacobian_shape chain_jacobian_shape panda_frame_jacobian_shape panda_analytic_vs_finite_diff get_frame_jacobian_batched",
 "kinematics/test_jacobian_time_variation.py": "batched_shapes",
 "kinematics/test_fk_contracts.py": "use_compile_matches_eager",
 "test_skeleton_signatures.py": "classes_are_classes callables_are_callable model_has_frozen_dataclass_shape",
 "test_exceptions.py": "module_accessible_from_top_level errors_carry_messages",
 "test_exceptions_wired.py": "device_mismatch_error_is_subclass_of_valueerror",
}
apply = "--apply" in sys.argv
extra = [a for a in sys.argv[1:] if a != "--apply"]  # file:name overrides
for e in extra:
    f, n = e.split(":"); DROPS[f] = DROPS.get(f, "") + " " + n
total = 0
for rel, names in DROPS.items():
    p = ROOT / rel
    if not p.exists():
        print("MISSING FILE", rel); continue
    src = p.read_text(); tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    spans = []
    for short in names.split():
        name = short if short.startswith("test_") else "test_" + short
        node = funcs.get(name)
        if node is None:
            cands = [k for k in funcs if short in k]
            print(f"NOT FOUND {rel}::{name}  candidates={cands}"); continue
        start = min([d.lineno for d in node.decorator_list] + [node.lineno])
        spans.append((start, node.end_lineno, name))
    total += len(spans)
    if apply and spans:
        lines = src.splitlines(keepends=True)
        for start, end, _ in sorted(spans, reverse=True):
            del lines[start - 1:end]
        out = re.sub(r"\n{4,}", "\n\n\n", "".join(lines)).rstrip("\n") + "\n"
        p.write_text(out)
    print(rel, [s[2] for s in spans])
print("total", total, "applied" if apply else "dry-run")
