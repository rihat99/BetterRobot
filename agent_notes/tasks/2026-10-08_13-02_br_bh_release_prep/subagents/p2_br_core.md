# P2 sweep — br_core (`lie, spatial, data_model, kinematics, dynamics`)

Budget note: this agent ran with a ~50k-token context, so the ~6200-line slice was NOT read end to end.
Coverage was by public-API probes, an independent Pinocchio parity script, and targeted reads. The
"read" list below is honest about depth; the unread files still need an eyes-on pass if the owner
wants the full Method step 1.

## Files read
- Read in full or in the relevant part: `kinematics/state.py` (lines 152-286: FK wrapper, all four
  Jacobian wrappers, `_world_jacobian`, `_express`), `kinematics/_fk_lean.py` (240-260),
  `dynamics/state.py` (35-54), `lie/so3.py` (`to_euler`), `tests/kinematics/test_state.py`,
  `tests/test_pinocchio/conftest.py` (fixtures).
- Signatures only (grep of `def`): `lie/{so3,se3,tangents}.py`, `kinematics/jacobian.py`,
  `dynamics/state.py`, `data_model/model.py`.
- Exercised through probes but source not read: the rest of `lie/`, `spatial/*`, `data_model/*`
  (incl. `joint_models/*`), `kinematics/_jacobian_columns.py`, `dynamics/{_world,state_manifold}.py`.

## Fixed (what, where, test)
1. Zero-size batch crashed FK and dynamics: `reshape(0, -1)` / `view(0, nj, -1)` cannot infer
   the trailing size when there are 0 elements (`RuntimeError: cannot reshape tensor of 0 elements`).
   Replaced `-1` with the known size in three places:
   - `src/better_robot/kinematics/state.py:173`: `q_full.reshape(count, structure.nq_full)`
   - `src/better_robot/kinematics/_fk_lean.py:258`: `.view(count, nj, plan.feature_index.numel() // nj)`
   - `src/better_robot/dynamics/state.py:52`: `.reshape(math.prod(batch), model.structure.nv_full)`
   After the fix, all of these return correct `(0, ...)` shapes on G1+free-flyer: FK, `frame_jacobian`,
   `joint_jacobian`, `frame_jacobian_time_variation`, `mass_matrix`, `center_of_mass`,
   `centroidal_momentum`, `inverse_dynamics`, `forward_dynamics`, `integrate`, `difference`.
   Test: `tests/kinematics/test_empty_batch_flows_through_fk_and_dynamics` in
   `tests/kinematics/test_state.py` (Panda + free flyer: FK, ABA, RNEA with B=0). Before the fix the
   probe raised; after, it passes.

## Reported, not changed (ranked)
1. MEDIUM, numerics: `lie.so3.to_euler` gives the wrong rotation at exact gimbal lock. Probe:
   `e = [0.3, pi/2, -0.4]`; `to_matrix(from_euler(to_euler(from_euler(e))))` differs from
   `to_matrix(from_euler(e))` by 1.76 (max abs entry). The docstring says roll and yaw cannot be
   identified separately at pitch ±pi/2. That is true, but the returned triple should still
   reproduce the rotation. At pitch = pi/2 both `atan2` calls get (≈0, ≈0) and return arbitrary
   angles. Suggested fix: when `|sin_pitch| > 1 - eps`, set yaw = 0 and
   roll = `atan2(2(qx qy - qw qz), 1 - 2(qx² + qz²))` (sign depends on the branch), plus a
   round-trip test. Not changed because it alters numerical output and gradients at the singularity.
   A 90° pitch is common in hand-written configs, so this case does occur in practice.
2. LOW, API shape: `joint_jacobian(..., joint: int)` and `joint_jacobian_time_variation(..., joint: int)`
   take only an int and index straight into the state without `model.joint_id(joint)`. A joint
   name raises a raw indexing error, and a negative or out-of-range int silently picks a wrong joint
   or gives an unclear `IndexError`. `frame_jacobian` accepts `int | str` and validates. Suggested fix: call
   `joint = model.joint_id(joint)` and widen the type to `int | str` to match `frame_jacobian`.
   This changes the API, so the owner decides.
3. LOW, perf/design: `frame_jacobian_time_variation` and `joint_jacobian_time_variation` recompute
   the Jacobian and dJ for every joint on each call. Nothing is memoised on the state, unlike
   `frame_jacobian`'s column table. One call costs about 4x a full FK (numbers below). Querying K
   frames costs K times as much. A memo would have to be keyed by `v`, so this is a design change.
4. INFO: frame names are prefixed `body_` (e.g. `body_torso_link`), while Pinocchio uses the bare link
   name. This is intentional, but it is easy to get wrong in a parity check. The first version of my
   probe silently compared 0 frames.

## Pinocchio parity (independent script, G1 + free flyer, nq=36, nv=35, float64, B=4 random q/v/a/tau)
Script: `agent_notes/tasks/2026-10-08_13-02_br_bh_release_prep/subagents/p2_br_core_probes/pin_spot.py`
| quantity | max abs err |
|---|---|
| FK, 40 frames x 4 configs (160 frame poses) | 7.8e-16 |
| frame Jacobian WORLD / LOCAL / LOCAL_WORLD_ALIGNED (3 frames) | 6.7e-16 / 1.3e-15 / 5.6e-16 |
| joint Jacobian LWA (left knee) | 4.4e-16 |
| RNEA | 2.6e-13 |
| CRBA (upper triangle mirrored) | 2.1e-14 |
| ABA | 9.4e-10 (random tau on light links gives large accelerations; relative error is small) |
| `integrate` / `difference` | 4.4e-16 / 8.9e-16 |

## Edge-case probes (all pass unless noted)
Script: `.../p2_br_core_probes/edge.py`. Checked: nested batch `(2,3,nq)` through FK and all three
dynamics calls; broadcasting a `(nv,)` tangent against `(B,nq)`; float32 model preserves its dtype;
autograd through local frame Jacobian + RNEA + CRBA w.r.t. q is finite. Gradients are finite at
the singular points: `so3.exp`/`se3.exp` at 0, `so3.log`/`se3.log` at identity, and
`right_jacobian_so3`/`right_jacobian_inv_se3` at 0. Also checked: `so3.log` near pi; the
double cover (`log(-q) == log(q)`); `slerp` takes the shortest path across a hemisphere sign flip
and stays finite for `slerp(a, a)`. The only failure was the Euler gimbal-lock case (item 1 above).

## Performance notes (measured; CPU, 8 threads, G1+FF float32, B=64, torch.utils.benchmark median)
| call | median |
|---|---|
| `fk` | 951 us |
| `frame_jacobian`, column table already memoised on state | 86 us |
| `frame_jacobian` on a fresh state (FK + table) | 1337 us |
| `frame_jacobian_time_variation`, 1 frame | 3780 us |
No change made. The memoised Jacobian path works as designed (about 15x cheaper on a warm state).
The dJ path is the outlier (item 3 above).

## Gate results
- Slice tests before edits (`tests/{lie,spatial,data_model,kinematics,dynamics,test_pinocchio,autograd}`):
  523 passed, 1 skipped.
- `tests/kinematics/test_state.py` after edits: 4 passed (includes the new test).
- ruff on touched files: the 3 findings in `test_state.py` existed before my edits (PLC0415 at lines 13/47/83);
  the new test adds none, and the touched `src` files are clean.
- Full suite (`-n 8`, shared worktree with concurrent edits from other slices): 1586 passed, 11 skipped,
  3 failed. All 3 failures are viewer docs snippets: `tests/docs/test_snippets.py::{test_getting_started_outputs_fresh[visualize.md],
  test_page_snippets_execute[docs/concepts/viewer.md], test_page_snippets_execute[docs/getting-started/visualize.md]}`.
  They are outside br_core. `tests/docs/test_snippets.py` was already modified in the worktree by another slice.
  I did not diagnose them. Log: scratchpad `p2_br_core_full.log`.
- ruff `src tests`: 82 findings, the same as the stated baseline.
- mkdocs: not run (no docstrings referenced by docs were changed).
