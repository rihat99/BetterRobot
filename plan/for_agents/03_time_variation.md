# Order 03 — Jacobian time variation (J̇) for joints and frames

Read `plan/README.md` first — its "Ground truth" section is the parity
contract, verified numerically against pinocchio 3.9.0; do not re-derive
or deviate. Depends on order 02 (consumes its private column-table
helper).

## Math being implemented (from README, condensed)

- WORLD cache: `dJ[:, cols(a)] = ad(ov_a) @ J[:, cols(a)]` per owner joint
  `a`, with `ov_a` the world spatial velocity of joint `a` and
  `ad([v;w]) = [[ŵ, v̂],[0, ŵ]]`. Transport-only — every shipped joint has
  a constant local motion subspace (leave a one-line comment naming this
  assumption and pointing at the `base.py` subspace-derivative hooks).
- `ov_a = J_a @ v` — one batched einsum against the per-joint world
  Jacobian. Width discipline (this is the mimic trap, be exact):
  - `ov` comes from the **reduced** per-joint Jacobian × reduced `v`
    (`J_reduced_a @ v_reduced == J_full_a @ v_full` — verified identity);
  - the transport `dJ[:, cols(a)] = ad(ov_a) @ J[:, cols(a)]` is keyed on
    each column's owner and is therefore valid on **full-width** owner
    columns only — a reduced mimic column blends columns with different
    owners and is unrepresentable there. The pass always builds the
    full-width column table via the order-02 helper; a passed reduced
    `joint_jacobians` supplies `ov` (and the getter formulas' `J` terms)
    but never replaces the full-width build;
  - `reduce_jacobian` (`@ v_expansion`) is applied last.
- Getter frames (joint origin `oMi` / frame `oMi·placement`, verified to
  machine precision):
  - `world`: the cache.
  - `local_world_aligned`: `J̇_ang = dJ_ang`;
    `J̇_lin = dJ_lin − p×dJ_ang − v_pt×J_W_ang`,
    `v_pt = ov.lin + ov.ang×p`. The third term is the one a naive
    static-style translation misses (O(1) error — this is the known trap).
  - `local`: `J̇_col = Ad(oMref)⁻¹ dJ_col − ad(v_ref)(Ad(oMref)⁻¹ J_col)`,
    `v_ref` the ref's LOCAL spatial velocity (for a frame:
    `placement.actInv(v_parent_joint)`).
- Mimic: owner-indexed formulas are valid on **full-width** columns only;
  compute full, then `J̇_reduced = J̇_full @ v_expansion` (constant map) —
  mirror `reduce_jacobian`.

## Surface

`src/better_robot/kinematics/jacobian.py` (or a sibling module if the file
grows past taste — follow `file-naming`; the round does not mandate a
split):

- `joint_jacobians_time_variation_raw(structure, q, v, joint_pose_world,
  *, joint_jacobians=None) -> JointJacobiansTimeVariationResult` — WORLD
  J̇, `(B..., njoints, 6, nv)`. Broadcast `v` to the execution batch
  exactly as the existing pass broadcasts `q`
  (`broadcast_to_execution_batch`, `jacobian.py:51-57`) — a silently
  mis-broadcast `v` in the `ov` einsum is the failure mode. A passed
  reduced `joint_jacobians` is reused per the width discipline above
  (precedent for the parameter: `frame_jacobian_raw`); the full-width
  column table always comes from the order-02 helper.
- `frame_jacobian_time_variation_raw(structure, values, q, v,
  joint_pose_world, frame_id, *, reference="local_world_aligned",
  joint_jacobians=None, joint_jacobians_dot=None) -> Tensor` — one frame,
  three references, closed forms above.
- `compute_joint_jacobians_time_variation(model, data) -> Data` — requires
  `KinematicsLevel.PLACEMENTS` and a set `data.v` (raise `StaleCacheError`
  with a clear "set data.v / run a velocity-producing call first" message
  if `data.v is None`); fills the pre-wired `data.joint_jacobians_dot`
  slot (declared `data.py:106`, invalidation already in
  `_VELOCITY_CACHES`). Also fills `data.joint_jacobians` if it computes
  them as a byproduct — do not recompute what the cache already holds.
  Known asymmetry, document it in the docstring: the slot lives in the
  velocity cache bucket but is filled while `_kinematics_level` stays
  `PLACEMENTS` (setting `data.v` never promotes the level, `data.py:135-140`,
  and this wrapper must not promote it either — levels track the kinematic
  recursion, not cache presence).
- `get_joint_jacobian_time_variation(model, data, joint_id, *,
  reference="world")` and `get_frame_jacobian_time_variation(model, data,
  frame_id, *, reference="local_world_aligned")` — derive LOCAL/LWA from
  the WORLD cache per the closed forms; compute-on-miss like
  `get_joint_jacobian` does (`jacobian.py:189-190`).
- Docstrings must state the acceleration pairing explicitly: WORLD/LOCAL
  J̇ satisfies `a_spatial = J v̇ + J̇ v`; LWA J̇ satisfies
  `a_classical = J v̇ + J̇ v`. This is the semantic contract users will
  reach for.
- Exports: the three public functions at `better_robot/__init__.py`
  top-level (alongside the existing jacobian trio) and everything under
  `kinematics/__init__.py`; pin new public names in
  `tests/contract/test_public_api.py` `REQUIRED` and the submodule
  contract's raw-pass list.

Performance bar: the WORLD J̇ pass adds ~one einsum, one gather, and a few
batched cross/matmul ops on top of the order-02 pass — launches independent
of njoints. No per-joint Python loops, no host syncs, autograd- and
`torch.func`-safe (jacrev and jacfwd), fullgraph-compilable.

## Tests

`tests/test_pinocchio/test_jacobian_time_variation_matches_pinocchio.py`
(net-new; oracle = `pin.computeJointJacobiansTimeVariation(model, data, q,
v)` then `pin.getJointJacobianTimeVariation` /
`pin.getFrameJacobianTimeVariation`):

- Panda: joints (incl. tip) and an operational frame with nontrivial local
  placement, all three references, `sample_panda_q` plus a new
  `sample_panda_v` added to `tests/test_pinocchio/conftest.py` (same
  style/seeding as `sample_panda_q`); batched case included. Tolerances:
  start from the frame-Jacobian file's (`atol=2e-6, rtol=1e-5`) and keep
  the tightest that passes — J̇ is one multiplication deeper than J, so a
  modest widening may be justified; document it in the file header if so,
  never silently.
- Free-flyer (G1) and spherical chain: use the fixtures promoted to
  `tests/test_pinocchio/conftest.py` by order 01 (from the module-local
  builders in `test_rnea_advanced_joints.py:28-121`) — this is where
  nq≠nv and quaternion tangents would break a wrong implementation.

`tests/kinematics/test_jacobian_time_variation.py` (net-new, fp32, builder
models, mirroring `test_jacobians.py` conventions):

- Finite-difference cross-check: central difference of the same-frame
  Jacobian along `v` using `model.integrate` retraction,
  `(J(q ⊕ v·dt) − J(q ⊖ v·dt)) / (2dt)` vs the getter, all three
  references, joints and frames (fp32 eps/tolerances per the existing
  `_analytic_vs_finite_diff` style). **Non-degeneracy is mandatory** or
  the test greens a wrong implementation: `v` nonzero in every coordinate
  block (fixed seed), tested joints/frames at a nonzero world origin with
  angular DOF above the point (identity placements at the origin make
  LWA == WORLD and kill the `−v_pt×J_ang` term — the exact trap term this
  test exists to catch, O(1) magnitude when non-degenerate). Assert
  `WORLD J̇ != LWA J̇` on at least one case so the fixture itself is
  proven non-degenerate.
- Raw-vs-workspace agreement; reuse-path equivalence (passed
  `joint_jacobians`/`joint_jacobians_dot` vs recompute); batched shapes;
  jacrev == jacfwd through the raw pass; fullgraph `torch.compile` smoke
  (mirror `test_jacobians.py:124`).
- Cache semantics: `data.joint_jacobians_dot` invalidated by `v`
  reassignment, surviving `a` reassignment (the `_VELOCITY_CACHES` wiring
  — one small test, since this is the slot's first producer).
- Mimic: extend `tests/kinematics/test_mimic.py` with the covector
  reduction identity (mirror lines 69-73: constrained J̇ == full J̇ @
  `v_expansion`).

## Acceptance

- All new tests green; full CPU gate green; `ruff` clean; no fp64 outside
  `tests/test_pinocchio`.
- Contract tests updated for the new public symbols (additive only).
- `03_results.md`: parity max-errors per reference/model, FD max-errors,
  and the launch count of the combined J+J̇ pass on MHR-203 (same bench
  protocol as order 02).

## Out of scope

Docs pages and changelog (order 04). ∂J/∂q. Warp. Any
`KinematicsLevel.VELOCITIES` producer or `joint_velocity_world` fill
(README non-goals).
