# M4 results — consumer packs and migration

**Status:** the authorized in-repository scope is complete on branch `dev`
(2026-07-17). Full M4 milestone acceptance is intentionally not claimed:
collision and all external-consumer migration work remain deferred under the
owner's repository boundary.

## Outcome

- Added a camera-thin `ProjectionResidual` with analytic configuration
  Jacobians, point weights/masks, clamped depth, and optional named observation
  parameters for implicit-differentiation visibility.
- Added masked Chamfer and a single-pass `SceneSDFProvider` feeding penetration,
  attraction, and clearance heads. Invalid padded NaNs produce finite zero rows
  and gradients.
- Added per-item Geman--McClure use and kept BetterRobot's kernel normalization:
  `weight = 2 rho'`, unit near-zero weight, and loss cap `c**2 / 2`.
- Added `solve_contact_forces` on the named-block LM stack, including batched
  clips, per-clip gravity, active contacts, base-wrench/magnitude/force-smooth/
  torque-smooth terms, and differentiable `rnea(fext=...)` assembly.
- Added exactly the four requested utilities: Euler/quaternion conversion,
  homogeneous-matrix/SE(3) conversion, weighted batched Umeyama alignment, and
  batched manifold trajectory smoothing.
- Added a public viewer primitive handle and frame-update surface supporting
  per-joint color/scale. Preserved `Visualizer` and `ForceVectorsOverlay` while
  deleting unused recorder, camera, interaction, overlay, collision-mode,
  offscreen-backend, and transport-only stubs with their exports/tests/docs.
- Regenerated the tracked API reference so additions and deletions match the
  source tree. CI remains `workflow_dispatch`/manual-only.

## Verification

- Focused M4 gate: **180 passed, 1 skipped**; the skipped test requires CUDA.
- Viewer suite: **56 passed**; roadmap stub inventory: **1 passed**.
- Compile/hot-path gate: **79 passed, 1 skipped**, and Umeyama passes a
  `torch.compile(fullgraph=True)` parity probe.
- Full CPU gate: **1,436 passed, 2 skipped, 3 deselected** with
  `-m "not bench and not cuda"` (58.78 s).
- Sphinx HTML succeeds; only four expected offline intersphinx warnings remain.
- Scoped Ruff, formatting, `git diff --check`, and offline lock validation pass.

## Deviations and deferred work

1. **T4.5 is not resolved.** The plan requires current-consumer evidence and an
   owner port-or-cut decision. The owner prohibited BHF/BVR access, so collision
   was neither inspected, ported, nor cut. The package is unchanged, and M6's
   collision kernel row must remain void/deferred rather than inventing an
   oracle.
2. **T4.6 was not executed.** BHF/BVR were not read or modified. Consequently,
   the symbol-by-symbol import table was not re-grepped, whole-pipeline parity
   benchmarks were not defined/measured, consumer imports were not checked,
   consumer line deletions are zero, and migration-gated BR shims were not
   deleted. This also means feature commits could not be paired with consumer
   deletions as the plan requested.
3. The padded `(tensor, bool validity_mask)` point-set convention was accepted
   using the owner's unattended-work instruction to take recommended decisions,
   rather than a synchronous review stop. It is fixed-size, batchable, and
   shared by Chamfer/SDF; no ragged container or camera abstraction was added.
4. The plan's Geman--McClure test text says the loss is bounded in `[0, 1)`, but
   the repository's frozen robust-kernel contract requires `weight = 2 rho'`
   with unit near-zero L2 normalization. The implementation therefore uses the
   internally consistent standard BR scaling and approaches `c**2 / 2`.
   Exact external formula parity remains unverified because consumer access is
   forbidden.
5. The contact-force result exposes final solver diagnostics, not a per-step
   loss trace, so the requested fixed-seed monotonic-history assertion was not
   fabricated. Tests instead cover base-wrench reduction, term formulas,
   `fext` gradcheck, batched/sequential parity, active-mask validation, and
   per-clip gravity. External pipeline quality parity remains part of T4.6.
6. Viewer API design and deletion evidence were limited to the checked-in plan,
   BetterRobot source/tests/docs, and the roadmap inventory. The external
   `_scene`/`_backend` call-site migration is deferred with T4.6.

## Commits

- `293e7c6` — add inverse contact-force solve
- `fa2d3e4` — add consumer-requested geometry helpers
- `7401c23` — add robust projection residual
- `c6a33d4` — add masked point-cloud scene terms
- `554095f` — expose viewer styling and delete stubs
- `f6c903d` — integrate public M4 API/docs/contracts
- `6568bae` — close M4 correctness gaps
- `6dd9dba` — keep geometry helpers compile-safe
