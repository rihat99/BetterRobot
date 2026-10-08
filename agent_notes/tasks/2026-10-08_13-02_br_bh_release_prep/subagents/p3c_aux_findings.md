# P3c — adversarial check of the site (mars_aux, read-only; saved by the orchestrator)

Paths relative to /home/rikhat.akizhanov/better/dev/{BetterRobot,BetterHuman}. Ranked by consequence.

## Claims to fix
1. optim prep "document optim batch is one leading axis" — WRONG (false gap): BR `docs/contracts/02-tensors.md:81-84`
   already says "no arbitrary leading batch axes here: flatten extra axes into B". Drop the item.
2. kindyn prep "replace the functorch probe → fullgraph compile" — INCOMPLETE: with `is_functorch_wrapped_tensor`
   (`_fk_lean.py:213`) patched, compile next fails at `torch.is_inference_mode_enabled` (`:216`); the `_version` /
   `object.__setattr__` caches at `:196`, `:218-222` are likely next. Say "make the placement-cache path compile-safe
   (guard with `torch.compiler.is_compiling()`), add a fullgraph test for FK, RNEA and CRBA"; RNEA/CRBA not probed.
3. toolbox/index "three names for one SE(3) quantity" — OVERSTATED: `joint_placements` is the model's local constant
   7-vector (`model_values.py:60`), a different quantity; `joint_transform` (3x4 world, `kinematics/state.py:108`) and
   `joint_pose` (7-vector world, lazy, `:134`) are one quantity in two forms. Pinocchio has the same placement/pose
   split. Real gap: name the 3x4 form consistently (`*_transform` vs `*_pose`). Blast radius: `frame_pose` 43 files,
   `joint_transform` 26, `joint_pose` 23, `frame_transform` 7 (BR+BH).
4. index "row-by-row autograd Jacobians" — MISLEADING: `problem.py:336, :565` use `is_grads_batched=True`
   (vectorized VJPs, no Python loop). Keep the cost statement "one reverse pass per residual row, run vectorized".
5. viewer page says the timeline is a "frame counter" — breaks `docs/contracts/01-naming.md:62-63` ("frame always
   means a robot frame, never an animation frame"; time axis is `step`). Use "step counter". (`timeline.py:12,19-23`
   docstring has the same wording — code item for the owner.)
6. collision "Panda has 9 mesh-only links" — PARTIAL: 11 links have collision geometry, 9 mesh geoms and 8 boxes;
   "mesh-only" not shown. Say "9 of the 11 links carry mesh collision geometry".
7. human "time axis (B,T,nq)+fps missing from contracts" — PARTIAL: `docs/contracts/bh-addendum.md:92` already gives
   `(B, T, nq)`; only fps and semantics are missing.
8. human: closed registry is `better_human/__init__.py:12` (not `families/__init__.py`).
9. toolbox "parsers drop damping/armature/friction" — CONFIRMED, but fields exist in `model_values.py:68-70`, so it is
   wiring only (say so; makes the prep change smaller).
10. kindyn understates what exists: `fext` already in `inverse_dynamics` (`dynamics/state.py:55-58`) — mention.

## Prep-change attacks
- optim: items 2–3 (fixed `State.extra` keys; optional prior state to `Solver.init`) are additive, ≤ 1 day.
- kindyn: ≤ 1 day if phrased as "cache path compile-safe"; add fullgraph tests for RNEA/CRBA.
- collision: keyword-only `ignore=None` — one definition, one call site; non-breaking.
- human: `Rig` is a `Protocol` (`core/rig.py:41`) with 3 implementations; passing `identity_params` unconditionally
  from `body.py:285` breaks outside Rigs → it IS a protocol change (mark breaking-for-third-party-Rigs); the
  "feature" memo at `body.py:285` must be invalidated when identity changes.
- toolbox: both changes breaking (page admits); needs a name for the 3x4 form first.
- viewer: seconds-based time changes contract `01-naming.md:62` (time axis is `step`) — update in the same change;
  per-layer time seam is 1–2 days, not 1.

## Missing roadmap items (robotics/biomechanics researcher)
- Mocap import: C3D, BVH, AMASS — absent from code and site.
- Marker sets / virtual markers for OpenSim-style marker IK.
- Muscles / musculotendon models (not even "maybe").
- Force-plate / GRF input (relates to existing `fext`).

## Readability
- index lacks a one-line verdict at the top ("Release-ready after X; do these N now").
- "Top 10 gaps" renders as a run-on paragraph on a phone → numbered list, one per line.
- viewer page: "time base" / "time seam" / "time model" for one thing → one term.
- index vs optim wording for the Jacobian cost → one wording.
- "sink", "seam", "brick" never defined on first use (index).

Confirmed (no change): optim imports only torch; sink seam `linear.py:296-388`, 3 sinks, no dense-fallback warning;
term kinds residual|scalar (`problem.py:61`); `Result` has no solver state (`phases.py:38-45`; `State.extra` at
`solvers.py:27`); no derivatives/energy/Coriolis accessors; `_memo` in frozen state (`kinematics/state.py:111`);
no `ignore=` in `self_collision_pairs` (`robot_spheres.py:64`); only `PlotPanel`; `Rig.corrective_features` has no
identity (`core/rig.py:61,:123`, `mhr_rig.py:89`); `FamilySpec.meta` read by core (`body.py:59,269,277`, `spec.py:205`);
`random_configuration` returns `(nq,)` (`model.py:245`); fixes (URDF inertia `urdf.py:197-200`, `Viewer.url`
`viewer.py:57-61`, FK at B=0 → `(0,14,3,4)`, gizmo `gizmo.py:55` + test `:62`) confirmed.
