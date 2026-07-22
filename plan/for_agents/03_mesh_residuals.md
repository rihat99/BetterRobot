> **Implementation log (2026-07-22): complete on `dev`.** Composed point Nodes now feed Scene-SDF, Chamfer, and explicit-point projection; per-head gates, point-to-plane distance, and linear confidence are implemented.
> **Validation:** 1,634 CPU tests and 49 CUDA tests pass; strict HTML and all 30 doctests pass; source is 22,724 → 22,947 lines (`+223`).
> **Finding:** Exact old Scene confidence numerics are generally unrecoverable because geometric confidence also changed from linear row scaling to square-root scaling. See [`03_results.md`](03_results.md).

# Order 03 — Mesh-reading residuals: node clouds, gates, point projection

Read `plan/README.md` and `DESIGN_RULES.md` first. Orders 01 and 02 must be
landed. This order makes BR's shipped point-cloud residuals usable on a
*computed* (posed) mesh, which is what BVR forked them for.

## Motivation (verified)

- `SceneSDFState.query_points` and `MaskedChamferResidual.source/target`
  accept only leaf variables or fixed tensors (`scene_sdf.py:36-48`,
  `chamfer.py:25-31`) — a posed mesh cannot feed them.
- Scene residuals gate only by confidence and validity, always sum, and
  multiply confidence into the rows *before* squaring, so the objective is
  quadratic in confidence (`scene_sdf.py:126-134,179-182`); BVR rebuilt
  them for per-head distance/sign/dead-zone gates and active-mean
  normalization (`tools/human_optim/residuals.py:281-341`).
- BVR's re-derived signed distance is point-to-**plane**
  (`losses.py:139`), BR's is point-to-**point** with a sign
  (`scene_sdf.py:119-125`) — a real semantic gap, standard in scan
  alignment.
- `ProjectionResidual` projects model frame-table rows only
  (`projection.py:132-138`); skinned landmark vertices are unreachable, so
  the reprojection fork survives any composition feature.

## Work items

1. **Node-valued inputs.** `SceneSDFState` and `MaskedChamferResidual`
   accept, for each point/mask/confidence input: a `Node` (read via
   `value()` per epoch), a `Variable` (static ones updatable via
   `problem.update()`; bool masks use order 02's non-floating statics), or
   a raw tensor (captured at construction, documented as such). The
   consuming residuals list the state/node in `nodes` as today.
2. **Gates live on the penalty residuals, not the shared state.**
   Penetration, attraction, and clearance have *different* active sets, so
   `SceneSDFState` keeps producing only the shared geometric result
   (signed distance, nearest distance, confidence, validity) and each
   penalty residual owns its own predicate. Exact semantics, all
   comparisons on **detached** quantities, every control optional with
   `None` = today's behavior:
   - common controls on all three: `min_confidence` (drop
     `confidence < min_confidence`), `max_distance` (drop
     `nearest_distance > max_distance`), and `mask` — an external
     per-query-point boolean input (static Variable or Node; this is where
     downstream contact/no-contact labels enter);
   - `ScenePenetrationResidual`: `max_penetration` drops rows with
     `signed_distance < -max_penetration` (too deep to trust the
     correspondence), and `margin` shifts the hinge to
     `relu(-(signed) - margin)` (a dead zone where shallow contact is
     free);
   - `SceneAttractionResidual`: `band` keeps only
     `|signed − target_distance| ≤ band` (one-sided pull toward
     surface within a trust band);
   - `SceneClearanceResidual`: `margin` shifts its hinge symmetrically to
     penetration's.
   The gate result is the residual's order-01 `active_groups()` mask;
   gated rows are exact zeros.
3. **Linear confidence.** Confidence is **detached** gating quality, not
   an optimization signal. Scene and chamfer multiply rows by a
   gradient-safe (double-`where`) square root of the detached confidence
   so the objective is linear in confidence with no zero-derivative trap.
   This changes current numerics: MIGRATION row, affected tests
   recalibrated deliberately, never by loosening tolerances.
4. **Point-to-plane option.** `SceneSDFState(distance="point"|"plane")`;
   `"plane"` projects the delta onto the nearest point's normal. Default
   stays `"point"`. Acceptance covers both regimes: query points aligned
   with their nearest point's normal (modes must agree) and tangentially
   offset queries (modes must differ, plane < point in magnitude).
5. **`PointProjectionResidual`.** Projects a point set through the
   existing camera convention of `projection.py` — no second camera
   convention. Event semantics are explicit: the source supplies
   `(*batch, T, P, 3)` for a trajectory (or `(*batch, P, 3)`
   non-temporal); `T`/`P` are event axes, `dim = 2·T·P` (resp. `2·P`),
   rows flattened in the same order as the observation tensor;
   `group_size=2`. Per-point confidence enters as a per-group outer
   weight; a visibility/validity mask input may be a Node (recomputed per
   epoch) and becomes `active_groups()`. An analytic Jacobian is optional;
   if absent, tests pass an explicit strategy (the
   `AutodiffFallbackWarning` policy stays). Include tests with arbitrary
   leading batch axes and with unbatched trajectory input — `T` must never
   be mistaken for an execution batch axis.
6. Update `residuals/CLAUDE.md` (vision pack section) and the concepts
   page; MIGRATION rows for the confidence-linearity change and any
   constructor signature changes.

## Boundaries

- Do not import BVR's constants, thresholds, or sign conventions; the
  predicates above are the design — implement them as written, and note
  in the results file where BVR's forks differ.
- Nearest-neighbour indices stay detached; distances stay differentiable
  (existing contract).
- No visibility *model* (front-facing tests, renderers) enters BR — the
  mask input is the boundary; computing it is downstream domain code.

## Acceptance

- An integration test poses a small synthetic "body" from `q` in a node
  and feeds the same posed points to scene penetration + attraction +
  chamfer + point projection **without subclassing any of them**, with
  gates and `mean_active` on — the BVR shape, in-tree.
- Gate defaults (`None` everywhere) reproduce current numerics exactly,
  modulo the documented confidence-linearity change (regression-pinned).
- Point-vs-plane aligned/tangential test; linear-confidence hand
  computation; a zero-confidence row produces zero gradient and no NaN.
- `problem.update()` swaps the scene candidate cloud and an external
  bool mask mid-solve in a test and the next evaluation reflects both.
- Full gate green; CUDA suite green (chamfer/scene have device-sensitive
  paths); net src growth ≤ +250.
