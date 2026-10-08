# PLAN — implement the "now" items and the documented roadmap (2026-10-08 16:37 Dubai)

Follow-up of `2026-10-07_16-44…` and `2026-10-08_13-02_br_bh_release_prep` (site + REPORT there). Owner: "do everything
marked now; next and later become part of the documented roadmap in the docs; prepare to finalize this version."

## Owner decisions (2026-10-08, verbatim meaning)
Toolbox: SE(3) names — document placement / pose / transform(3x4), no rename. Identifiers keyword-only after
`(model, state)` everywhere incl. terms. Composition lives on the IR (public `io.parse`). `BackendNotAvailableError`
kept as a reserved hook for a kernel brick. `Trajectory` in BR as a minimal frozen dataclass.
Optim: batch axis — document now, flatten later. Constraints as term kinds on `Problem` (reserve vocabulary now).
Sparse: pure torch for now, CHOLMOD maybe later. DDP/MPC in scope (roadmap). Keep reverse-mode Jacobians.
Kin/dyn: batch-first target, compile-clean as the path to batch-1. Derivatives: autograd contract first, analytic as an
option later. Contacts: only contact Jacobians (constrained-dynamics brick far future). Kernels: Warp stays the
reserved option, no work now. Biomechanics joints: BH extension, BR stays generic. `to_euler` gimbal fix now.
Collision: separate `RobotGeometry` brick (ADR now); capsules too (next); lazy trimesh allowed; closest point/normal as
separate functions; contact for dynamics in BR (generic).
Viewer: BVR later; time model prep only (optional per-layer `times`, integer steps default); viser only;
`viewer.server.gui` is the sanctioned studio API; npz export is enough. MJCF: ship as is, mark experimental.
BetterHuman: `vertices()`/`frame_poses()` become properties before merge; STAR next; fitting not on the roadmap;
retargeting a separate package (not BH); third-party families in-tree only. `test_reduction` compares at float32 ulp.

## Work (every item ≤ 1 day; breaking changes allowed, API not fixed)
BR wave 1 (5 agents, disjoint files):
- A `br_api`: keyword-only ids after `(model, state)` in kinematics/dynamics; `int | str` joint ids; `io.parse` → IR;
  `IRJoint` damping/friction/armature wired to `ModelValues`; `Trajectory`; naming contract + glossary (placement/pose/
  transform, step vs frame); `BackendNotAvailableError` documented as reserved; MJCF experimental note.
- B `br_optim`: fixed `State.extra` diagnostic keys; `Result` carries final solver state, `solve`/`run` accept it
  (warm start); warn once on dense fallback for `time=` problems; contract 08 names the sink seam; docs reserve
  `equality`/`inequality` term kinds; batch-axis rule cross-referenced.
- C `br_kindyn`: compile-safe placement-cache path + `fullgraph=True` tests for FK, RNEA, CRBA; ADR 0007 derivatives
  contract; document `_memo` keys; `to_euler` gimbal-lock fix + test; bench smoke test, `-m bench` collection, README.
- D `br_collision`: keyword-only `pairs`/ids in collision + terms; `ignore=` on `self_collision_pairs`;
  `RobotSpheres.to()`; ADR 0008 geometry brick; document `meta["ir"]` schema freeze, `distance_fn` protocol, SDF
  outside-grid, batched offsets.
- E `br_viewer`: optional per-layer `times` + one index mapping; contract 01 time-axis update; `Viewer.on_pointer`/
  `clear_pointer`; `_nearest_on_ray` → `tools/picking.py`; `CameraLayer` own file; lazy `step -> array` convention
  documented; viewer.md "limits" section; timeline docstring step wording; viser-only + `server.gui` sanctioned (ADR 0004 note).
Wave 2: F `bh` (identity in correctives + memo invalidation; meta keys documented; fps/time semantics; properties;
  test tolerance; studio pointer API switch + `frames.active` reset on model switch; BH call-site fallout from A/D/E);
  G `roadmap` docs in both repos (next/later/maybe from the site, owner's decisions); H `mars_aux` adversarial pass
  on the BR diff. Wave 3: gates, REPORT, commits.

## Success
All tests pass in both repos (BH known failure resolved per decision), ruff not worse, strict docs; each item above
has a test or a doc line; adversarial pass findings fixed or recorded.
