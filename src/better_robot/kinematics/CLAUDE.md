# kinematics/ — Forward Kinematics and Jacobians

## Entry Points

- `forward_kinematics(model, q_or_data, compute_frames=False, use_warp=False, use_compile=False)` — selects one whole FK pass, fills `joint_pose_world` (and `frame_pose_world` if `compute_frames=True`)
- `forward_kinematics_raw(structure, values, q)` — pure Torch pass returning a named `FKResult`
- `frame_placements_raw(structure, values, joint_pose_world)` — pure frame-table pass returning a named result
- `joint_jacobians_raw(structure, q, joint_pose_world)` — sequencing-free Jacobian pass returning a named result
- `frame_jacobian_raw(structure, values, q, joint_pose_world, frame_id, reference=..., joint_jacobians=None)` — pure one-frame Jacobian extraction with optional reuse of an existing joint-Jacobian pass
- `joint_jacobians_time_variation_raw(structure, q, v, joint_pose_world, joint_jacobians=None)` — sequencing-free WORLD `J̇` pass returning a named result (echoes `J` as a byproduct)
- `frame_jacobian_time_variation_raw(structure, values, q, v, joint_pose_world, frame_id, reference=..., joint_jacobians=None, joint_jacobians_dot=None)` — pure one-frame `J̇` extraction with optional reuse of an existing `J`/`J̇` pass
- `update_frame_placements(model, data)` — fills `frame_pose_world` from existing `joint_pose_world`
- `compute_joint_jacobians(model, data)` — fills `data.joint_jacobians` for all joints
- `compute_joint_jacobians_time_variation(model, data)` — fills `data.joint_jacobians_dot` (WORLD `J̇`) for all joints; requires a set `data.v`
- `get_frame_jacobian(model, data, frame_id, reference=...)` — extracts `(B..., 6, nv)` for one frame
- `get_joint_jacobian(model, data, joint_id, reference=...)` — same for joints
- `get_frame_jacobian_time_variation(model, data, frame_id, reference=...)` — extracts `(B..., 6, nv)` `J̇` for one frame
- `get_joint_jacobian_time_variation(model, data, joint_id, reference=...)` — same for joints

All passes use one execution batch: the right-aligned broadcast of `q` with
the joint-placement, body-inertia, and frame-placement value tables. Callers
must add semantic singleton axes explicitly (for example `(B, 1, njoints, 7)`
alongside `(B, T, nq)`); no auto-unsqueeze is performed.

## Jacobian Reference Frames (critical)

`get_frame_jacobian` returns **LOCAL_WORLD_ALIGNED** by default:
- Linear rows: velocity of frame origin in world frame
- Angular rows: angular velocity in world frame

To convert LWA to body-frame:
```python
# CORRECT — rotate only, don't apply full adjoint:
R_ee = so3.to_matrix(T_ee[..., 3:])
J_local = torch.cat([R_ee.mT @ J_world[..., :3, :], R_ee.mT @ J_world[..., 3:, :]], dim=-2)

# WRONG — adds spurious cross-term when J is LWA:
# J_local = se3.adjoint_inv(T_ee) @ J_world
```

`compute_joint_jacobians` returns WORLD-frame Jacobian (velocity at world origin).
`get_joint_jacobian` accepts the same three references and defaults to
**WORLD** (the cached frame). Its `local_world_aligned` translates the linear
rows to the joint origin (angular rows unchanged) — a translation only, not a
full adjoint, exactly like the frame LWA above.

**J̇ acceleration pairing (critical).** `get_joint_jacobian_time_variation` and
`get_frame_jacobian_time_variation` return `J̇` for `a = J v̇ + J̇ v`, but the
`a` that closes it depends on the reference. `world` and `local` pair with the
**spatial** acceleration; `local_world_aligned` pairs with the **classical**
(point) acceleration, because its basis point is the moving frame origin. Do
not compute `a` in one reference's `J̇` and interpret it as the other's
acceleration — the mismatch is an order-one term, not roundoff.

## Residual Jacobian Strategy

Kinematics owns no residual-Jacobian dispatcher. Object-referenced `Problem` accepts
the literal strategies `"auto"`, `"analytic"`, `"jacrev"`, `"jacfwd"`, and
`"finite_difference"`; finite differences remain an explicit debug oracle.

## FK Hot Path

`forward_kinematics_raw` runs the batched matrix lane in `_fk_matrix.py`, not a
per-joint quaternion-composition loop. Three stages: (1) group joints by kind
(from `ModelStructure`) and evaluate each kind's local transform for all its
joints in one batched call, as rotation matrices scattered into a per-joint
`(B..., nj, 3, 3)`/`(B..., nj, 3)` pair — free-flyer, spherical, revolute
(axis-aligned + arbitrary), prismatic, and fixed/universe are batched; rare
kinds (planar, translation, helical, composite) take a per-joint fallback
through the shared `joint_dispatch.joint_transform`; mimic is expanded by
`expand_configuration` and carries its concrete kind code; (2) chain world
placements with one batched `4x4` matmul per joint over `topo_order` — the only
irreducible sequential dependency, ~1 launch/joint; (3) one batched
matrix→quaternion for world and local. This drops the per-joint launch volume
from ~78 to ~2. The kind-grouping plan is a pure function of `ModelStructure`,
memoised on the structure instance (`_fk_matrix_plan`); `ModelStructure.to`
mints a fresh instance, so a moved model rebuilds its plan.

The lane emits **canonical-sign** quaternions (Shepperd selection via
`so3.from_matrix`): a returned `q` may be `-q` relative to a
quaternion-composition lane (same rotation). It also normalises output
quaternions, so unnormalised placement quaternions produce results that differ
from a non-normalising lane by a benign fp32-level gauge (forward and the
meaningless radial placement gradient). Everything stays autograd- and
`torch.func`-safe (jacrev and jacfwd); no host syncs.

`use_compile=True` routes the raw pass through a `torch.compile`d callable
cached per `ModelStructure` (`dynamic=True`, so batch size varies without a
per-size recompile; first use pays a multi-second compile). `use_warp=True`
opts into the CUDA-validated fused whole-pass Warp lane; **`use_warp` wins**
over `use_compile`. Unsupported Warp runtime, kind, dtype, or layout cases fall
back to the eager Torch raw pass with a one-shot warning; a decline during CUDA
graph capture raises. There is no per-Lie-operation or process-global compute
selection.

## Jacobian Hot Path

`joint_jacobians_raw` assembles every joint Jacobian in one batched
shared-column pass (`_jacobian_columns.py`), not a per-joint Python loop. The
world Jacobian has a purely static sparsity: joint `a`'s column block is
`Ad(oM_a) @ S_a` and appears in row joint `j`'s Jacobian iff `a` supports `j`,
independent of the row joint. So the pass is three batched stages: (1) one
batched `so3.to_matrix` / `hat_so3` over `joint_pose_world` and the constant
`ModelStructure.joint_motion_subspaces` table build every joint's own column
block at once — the per-joint `joint_motion_subspace(q)` calls are gone (every
shipped joint's subspace is configuration-independent); (2) a static gather
scatters the padded own blocks into a shared `(B..., 6, nv_full)` column table;
(3) a static supports-derived mask broadcasts that table into per-joint
Jacobians, and mimic reduction is applied last on full-width columns. This drops
launches from ~50 per joint to a constant ~52 for the whole pass, independent of
joint count (203-joint model, batch 256: forward 47.3 to 0.84 ms). The plan
(owner map, gather index, support mask) is a pure function of `ModelStructure`,
memoised on the structure instance (`_joint_jacobian_plan`) exactly like
`_fk_matrix_plan`; `ModelStructure.to` mints a fresh instance, so a moved model
rebuilds its plan.

`joint_jacobians_time_variation_raw` reuses that same column table and owner
map. `J̇` is transport-only (constant local motion subspaces, no `dS/dt` term):
`dJ[:, cols(a)] = ad(ov_a) @ J[:, cols(a)]` with `ov_a = J_a @ v` the WORLD
spatial velocity of the owning joint, evaluated column-wise with batched cross
products, masked per joint, and mimic-reduced last. It adds a constant ~8
launches over the Jacobian alone (one einsum for `ov`, owner gather, two cross
pairs, cat, transpose, a second mask-multiply, a second mimic reduce) — also
independent of joint count. LOCAL and LOCAL_WORLD_ALIGNED `J̇` are derived in
the getters from the single WORLD cache via verified closed forms. Everything
stays autograd- and `torch.func`-safe; no host syncs.
