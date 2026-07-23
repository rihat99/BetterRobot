# Plan — Jacobian round: joint LWA, time variation, batched pass

State at planning time (2026-07-23, branch `dev`, `8c4e619`): the BVR optim
round (orders 01–05) and the FK/RNEA launch-volume round are both delivered
and verified; full CPU gate 1,672 passed, CUDA 60 passed. The previous plan
(BVR round) is fully executed; its files were deleted with this commit and
its durable outcomes live in `docs/CHANGELOG.md`. Recover the old orders via
`git log -- plan/`.

## Why this round

Two owner-identified gaps against Pinocchio, plus one performance debt this
round is the natural place to pay:

1. `get_joint_jacobian` supports only `"world"` and `"local"`
   (`kinematics/jacobian.py:194-200`); Pinocchio's `getJointJacobian`
   supports LOCAL_WORLD_ALIGNED for joints too. Frame Jacobians already
   have all three.
2. There is no Jacobian time variation (J̇ given `q`, `v`) for joints or
   frames. Pinocchio 3.9.0 ships `computeJointJacobiansTimeVariation`,
   `getJointJacobianTimeVariation`, `getFrameJacobianTimeVariation`.
3. `joint_jacobians_raw` is still a per-joint Python loop of scalar glue
   (~15–25 launches × njoints, `jacobian.py:70-101`) — the exact
   launch-bound shape the FK round eliminated. J̇ should not inherit it, so
   the batched rewrite lands first and J̇ is built on the batched pass.

## Ground truth (verified against installed pinocchio 3.9.0, fp64, all
## claims numerically checked; scripts recorded in the planning session)

- **Joint LWA** is the world Jacobian with linear rows translated to the
  joint origin: `lin_LWA = lin_W − hat(p_joint) @ ang_W`, angular rows
  unchanged (max err ≤ 1.1e-16 across revolute/spherical/free-flyer).
- **The time-variation getter is the plain d/dt of the same-frame Jacobian
  in all three frames** (central FD match ≤ 2e-10 at dt=1e-6 for WORLD,
  LOCAL, and LWA, joints and frames). There is no frame in which the getter
  is something other than d/dt of that frame's Jacobian.
- **`data.dJ` recursion**: `dJ[:, cols(i)] = ad(ov_i) @ J[:, cols(i)]`
  where `ov_i` is the WORLD spatial velocity of joint `i` and
  `ad([v;w]) = [[ŵ, v̂],[0, ŵ]]` (column-wise match ≤ 8.9e-16). There is
  no `dS/dt` term: every standard joint's motion subspace is constant in
  its local frame. The same holds for every shipped BetterRobot joint
  (verified over all joint model classes; `base.py` has derivative hooks
  but all dispatch to zero) — **J̇ is transport-only**.
- **Closed-form getters from the WORLD cache** (verified to machine
  precision, joints and frames; `p` = ref origin, `ov` = world spatial
  velocity of the ref, `v_ref` = local spatial velocity of the ref):

  ```text
  WORLD:  J̇ = dJ                          (supporting columns)
  LWA:    J̇_ang = dJ_ang
          J̇_lin = dJ_lin − p × dJ_ang − v_pt × J_W_ang
          v_pt  = ov.lin + ov.ang × p     (world velocity of the point)
  LOCAL:  J̇_col = Ad(oMref)⁻¹ dJ_col − ad(v_ref) (Ad(oMref)⁻¹ J_col)
  ```

  The LWA trap: translating `dJ` the way the static Jacobian is translated
  misses the `− v_pt × J_W_ang` term and is off by O(1) (measured 2.64).
- **Acceleration identity**: `a = J v̇ + J̇ v` holds with the **spatial**
  acceleration for WORLD and LOCAL, and with the **classical** (point)
  acceleration for LWA (≤ 2e-15 for the matching pair, O(1) for the
  mismatched pair). Docstrings must state this pairing.
- **∂J/∂q**: pinocchio's kinematic-Hessian API is joints-only and not even
  exposed in its Python bindings. BetterRobot gets ∂J/∂q through autograd
  already. Explicit non-goal.

## Verified BetterRobot-side facts

- `Data.joint_jacobians_dot (B..., njoints, 6, nv)` is already declared and
  listed in `_VELOCITY_CACHES` (`data.py:47,106`) — invalidation on `q`/`v`
  reassignment is pre-wired; the slot just has no producer. Field name is
  settled; the forbidden legacy aliases `dJ` and `ov` must not appear as
  `Data` attributes (`tests/test_skeleton_signatures.py:88,95`; `ov` as a
  local variable or formula name is fine).
- `structure.joint_motion_subspaces` is a constant `(njoints, 6, max_nv)`
  table (`model_structure.py:162`) that RNEA already consumes batched; the
  per-joint `joint_motion_subspace(q_j)` calls in the current Jacobian loop
  are redundant with it for every shipped joint.
- `structure.supports` is materialized as ragged device tensors
  (`support_offsets`/`support_indices`, `model_structure.py:141-142`); a
  static column-owner map and a `(njoints, nv_full)` support mask derive
  from the structure once.
- Mimic reduction is a constant linear map: `J_reduced = J_full @
  v_expansion` (`reduced_coordinates.py`), hence
  `J̇_reduced = J̇_full @ v_expansion`. Owner-indexed column formulas apply
  on **full-width** columns only; reduce last, exactly like
  `joint_jacobians_raw` does today.
- `get_joint_jacobian` has zero internal callers in `src/` (residuals use
  frame Jacobians); the LWA addition migrates nobody.
- No joint-Jacobian parity test exists at all today — only
  `test_frame_jacobian_matches_pinocchio.py`. Order 01 closes that hole.
- Parity fixtures for nq≠nv already exist: G1 free-flyer and a hand-built
  pinocchio spherical chain in `test_rnea_advanced_joints.py:28-121`.
- `tests/test_pinocchio/conftest.py` has `sample_panda_q` but no velocity
  sampler; J̇ parity adds `sample_panda_v`.

## Design decisions

1. **API mirrors the existing trio and Pinocchio's names.** New public
   symbols: `compute_joint_jacobians_time_variation(model, data)`,
   `get_joint_jacobian_time_variation(model, data, joint_id, *,
   reference=...)`, `get_frame_jacobian_time_variation(model, data,
   frame_id, *, reference=...)`, exported top-level; raw passes
   `joint_jacobians_time_variation_raw` and
   `frame_jacobian_time_variation_raw` stay qualified under
   `better_robot.kinematics`. Defaults mirror the static getters:
   `"world"` for joints, `"local_world_aligned"` for frames.
2. **The cache holds WORLD J̇** (`data.joint_jacobians_dot`), exactly as
   `data.joint_jacobians` holds WORLD J; LOCAL and LWA are derived in the
   getters via the verified closed forms. One stored representation, no
   per-frame caches.
3. **Velocity source is `ov_j = J_j @ v`** (one batched einsum over the
   cached/reused world Jacobian). No standalone velocity-FK pass, no new
   `KinematicsLevel` producer, no dependency on RNEA. The compute wrapper
   requires PLACEMENTS and a set `data.v` (clear `StaleCacheError` if
   `data.v` is None).
4. **Batched pass first (order 02), J̇ second (order 03).** Both passes
   share one private helper that produces the full-width world column
   table + static owner map; J̇ adds one einsum, one gather, and batched
   cross products — target ~O(1) launches independent of njoints for both.
5. **Transport-only J̇ is a documented contract**: a comment at the formula
   site names the assumption (constant local motion subspaces) and points
   at the `base.py` derivative hooks that a future q-dependent joint would
   have to wire in. No speculative code path.
6. **Additive round**: no MIGRATION.md rows; new symbols are pinned in
   `tests/contract/test_public_api.py` `REQUIRED` and the submodule
   contract; `docs/CHANGELOG.md` gets one entry.

## Non-goals (conscious)

- No ∂J/∂q kinematic-Hessian API (autograd covers it; pinocchio's Python
  doesn't bind it either).
- No Warp lane for Jacobians or J̇ — the batched Torch pass is ~O(1)
  launches; there is nothing left for a kernel to win.
- No standalone velocity-FK pass and no filling of the currently dead
  `Data.joint_velocity_world` slot (pre-existing; noted for the owner, not
  touched). `KinematicsLevel.VELOCITIES` still has no exact producer —
  also pre-existing, also untouched.
- No FK-internals change to export world rotation matrices to the Jacobian
  pass (a possible future micro-opt; one batched `so3.to_matrix` per pass
  is cheap and keeps the passes decoupled).

## Orders

| Order | Depends on | Scope |
|---|---|---|
| 01 | — | `local_world_aligned` on `get_joint_jacobian`; net-new joint-Jacobian parity tests |
| 02 | 01 committed | Batched rewrite of `joint_jacobians_raw`; launch-count bench |
| 03 | 02 | Time-variation raw passes, Data wiring, getters, parity + FD + mimic + autograd tests |
| 04 | 01–03 | Docs, changelog, roadmap, API regen, final gates |

Orders run **sequentially**. 01 and 02 edit disjoint regions of the same
`jacobian.py`, and this tree has no per-order branch isolation (working
rules below forbid checkout/stash), so parallel agents would see each
other's half-finished file during test runs. 01 is tiny — land and commit
it first, then start 02.

## Gates (every order)

```bash
UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q -m "not bench and not cuda"
uv run ruff check src/ tests/
uv run sphinx-build -b html docs docs/_build/html   # order 04, strict
```

Pinocchio parity (`tests/test_pinocchio`) is ground truth and is never
weakened. Tests outside `tests/test_pinocchio` are fp32-only. Public
wrappers validate; raw passes trust inputs; everything autograd- and
`torch.func`-safe; no host syncs in hot paths.

## Working rules (concurrent tree)

Another session may work in this tree. Implementation agents: never run
`git add/commit/stash/checkout/restore`; keep a `git diff >
<scratchpad>/orderNN.patch` backup after every edit round; do not touch
`optim/`, `residuals/`, `tasks/`, `MIGRATION.md`; run only the test
directories your order owns plus the final gate. The orchestrator commits
finished orders immediately with explicit pathspecs.
