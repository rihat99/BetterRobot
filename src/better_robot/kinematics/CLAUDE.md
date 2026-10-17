# kinematics/ — Forward Kinematics and Jacobians

## Entry Points

- `forward_kinematics_state(model, q, *, check_quaternion_norm=False)` — one FK pass into an immutable
  `KinematicState` of `[R | p]` matrices (`joint_transform`, `frame_transform`); `joint_pose` / `frame_pose`,
  `*_rotation` / `*_translation` and `frame_pose_of(id)` are views converted on first use
- `frame_jacobian(model, state, *, frame, reference=...)` and `joint_jacobian(model, state, *, joint, reference=...)`
  — `(B..., 6, nv)`; both default to `local_world_aligned`
- `point_jacobian(model, state, *, frames, points)` — `(B..., K, 3, nv)` world velocity of points fixed to frames
  (the `local_world_aligned` linear rows moved to each point)
- `frame_jacobian_time_variation(model, state, *, v, frame, reference=...)` and
  `joint_jacobian_time_variation(model, state, *, v, joint, reference=...)` — `J̇` for the same references
- Raw kernels below them (`_fk_lean.py`, `_jacobian_columns.py`, `jacobian.py::*_time_variation_raw`)
  take `(structure, values, ...)` and trust validated inputs; `dynamics/` reuses them

All passes use one execution batch: the right-aligned broadcast of `q` with
the joint-placement, body-inertia, and frame-placement value tables. Callers
must add semantic singleton axes explicitly (for example `(B, 1, njoints, 7)`
alongside `(B, T, nq)`); no auto-unsqueeze is performed.

## Jacobian Reference Frames (critical)

`frame_jacobian` and `joint_jacobian` return **LOCAL_WORLD_ALIGNED** by default:
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

`reference="world"` gives the velocity at the world origin, `"local"` the body-frame twist. For a joint,
`local_world_aligned` translates the linear rows to the joint origin (angular rows unchanged) — a
translation only, not a full adjoint, exactly like the frame LWA above.

**J̇ acceleration pairing (critical).** `joint_jacobian_time_variation` and
`frame_jacobian_time_variation` return `J̇` for `a = J v̇ + J̇ v`, but the
`a` that closes it depends on the reference. `world` and `local` pair with the
**spatial** acceleration; `local_world_aligned` pairs with the **classical**
(point) acceleration, because its basis point is the moving frame origin. Do
not compute `a` in one reference's `J̇` and interpret it as the other's
acceleration — the mismatch is an order-one term, not roundoff.

## Residual Jacobians

Kinematics owns no residual-Jacobian machinery. `better_robot.terms` builds analytic term Jacobians from
`frame_jacobian`; `better_robot.optim` falls back to autodiff when a term has no `jac`.

## FK Hot Path

All FK goes through `_fk_lean.lean_core`. Every common joint transform is affine in a few
features of `q` (`sin`/`cos` of revolute angles, quaternion products, translation
coordinates), so one `addmm` with a constant matrix built from topology and joint axes gives
all joint transforms; the world chain is one matmul per tree level. Transforms are `[R | p]`
matrices `(N, njoints, 3, 4)`; nothing is converted to quaternions inside the pass. Rare kinds
(planar, translation, helical, composite) go through `joint_dispatch.joint_placement` one by
one. The plan is memoised on the structure per dtype/device (`_fk_lean_plans`).

Placements stay differentiable and may be batched. Unbatched placements are folded into the
affine map (cost independent of batch size); their `4x4` form is cached on the `ModelValues`
instance unless they require grad or are wrapped by `torch.func`.

`forward_kinematics_state` stores the `joint_transform` and `frame_transform` matrices; `joint_pose` /
`frame_pose` are converted on first use with canonical-sign quaternions (Shepperd selection via
`so3.from_matrix`), so a returned `q` may be `-q` relative to a quaternion-composition lane (same
rotation). `frame_jacobian` builds the shared world column table once per state and masks it for the
frame's parent joint. Panda: 44 operators for FK. Everything stays autograd- and `torch.func`-safe
(jacrev and jacfwd); no host syncs.

There is no per-Lie-operation or process-global compute selection; `torch.compile` the caller if needed.

## Jacobian Hot Path

`_jacobian_columns.py` assembles the Jacobian column table in one batched
shared-column pass (`_jacobian_columns.py`), not a per-joint Python loop. The
world Jacobian has a purely static sparsity: joint `a`'s column block is
`Ad(oM_a) @ S_a` and appears in row joint `j`'s Jacobian iff `a` supports `j`,
independent of the row joint. So the pass is two batched stages: (1) a static
gather of each column's owner rotation and origin times the constant
joint-frame subspace column (`plan.column_subspace`) builds the shared
`(B..., 6, nv_full)` column table at once — no padded `(njoints, max_nv)` grid,
so a free-flyer does not pad every joint to six columns; (2) a static
supports-derived mask broadcasts that table into per-joint
Jacobians, and mimic reduction is applied last on full-width columns. This drops
launches from ~50 per joint to a constant ~52 for the whole pass, independent of
joint count (203-joint model, batch 256: forward 47.3 to 0.84 ms). The plan
(owner map, gather index, support mask) is a pure function of `ModelStructure`,
memoised on the structure instance (`_joint_jacobian_plan`) exactly like
`_fk_lean_plans`; `ModelStructure.to` mints a fresh instance, so a moved model
rebuilds its plan.

`joint_jacobians_time_variation_raw` reuses that same column table and owner
map. `J̇` is transport-only (constant local motion subspaces, no `dS/dt` term):
`dJ[:, cols(a)] = ad(ov_a) @ J[:, cols(a)]` with `ov_a = J_a @ v` the WORLD
spatial velocity of the owning joint, evaluated column-wise with batched cross
products, masked per joint, and mimic-reduced last. It adds a constant ~8
launches over the Jacobian alone (one einsum for `ov`, owner gather, two cross
pairs, cat, transpose, a second mask-multiply, a second mimic reduce) — also
independent of joint count. LOCAL and LOCAL_WORLD_ALIGNED `J̇` are derived in
`state.py` from the WORLD result via verified closed forms. Everything
stays autograd- and `torch.func`-safe; no host syncs.

Design contracts (rule IDs `C-*`, enforcement matrix, ADRs): `docs/contracts/00-index.md`.
