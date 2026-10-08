# w2 G `roadmap` — report

Done. Both strict docs builds pass (exit 0, no warnings). BR `tests/docs` + `tests/contract`: 873 passed, 8 skipped
(the roadmap inventory test included). BH `tests/docs`: 22 passed. Nothing committed.

## Files changed
- BR `docs/reference/roadmap.md`
  - New intro + "How to read the plan" (now / next / later / maybe; no brick starts without an owner decision).
  - "Landed in this version (now)": items from w1_br_api, w1_br_api2, w1_br_optim, w1_br_kindyn, w1_br_collision,
    w1_br_viewer, each with its contract/ADR/concept link (C-ERR-2, 07, 08, ADR 0004/0007/0008, C-STATE-4,
    C-NAME-4/5, glossary, io, optimization, collision, viewer).
  - The explicit-raise inventory block is byte-identical (contract test passes).
  - Old "Dynamics and spatial algebra" / "Model building" / "Collision" sections: text unchanged, now `###` under a
    new `## Current limits` parent.
  - "Deferred directions" list replaced by `## Plan by theme` with five tables (brick | why | prerequisite | when):
    Optimization, Kinematics and dynamics, Collision, Viewer, Model building and tools. Every old deferred item
    (matrix-free route, Schur, compute_minverse, Coriolis, centroidal derivatives, offset contact points, parameter
    accessors, B-splines, external benchmarks, CI) is folded into a row.
- BH `docs/reference/roadmap.md`: vision text untouched; `## Roadmap` appended (intro, "Landed (now)", tables
  "Families and bodies", "Motion and data", and "Not on the roadmap": fitting, retargeting = separate package,
  third-party families in-tree only, video reconstruction later).
- BR `mkdocs.yml`: nav lines `0007 Derivatives contract` and `0008 Geometry brick` after 0006.
- BR and BH `docs/concepts/architecture.md` "Where next": one link line each to the roadmap page.

## Owner decisions applied (override the site)
Capsules moved later -> next. Video panel (BVR) next -> later. Own three.js client dropped (viser only, ADR 0004).
npz export kept as later, "the only export format". Contacts = contact Jacobians (next); constrained dynamics +
derivatives "far future". Warp backend = maybe. CHOLMOD = maybe; pure-torch block-sparse = next. DDP/iLQR/MPC = later
(in scope). Batch axis flattening = later. Fitting recipes and `register_family` removed; retarget module removed
(separate package). STAR, mocap import (C3D/BVH/AMASS) next; marker sets later; muscles maybe. MJCF hardening later.

## Choices I made (please check)
1. Forward-mode Jacobians: site said next; I moved it to maybe ("reverse mode stays the default") because of the
   owner's "Keep reverse-mode Jacobians". Move back to next if that decision meant something else.
2. Biomechanical joints in BH: placed as "later" (the owner fixed placement, not timing).
3. BH "Landed (now)" list is written from w2 brief F (w2_bh.md did not exist when I finished; on disk BH still had
   `def vertices`/`corrective_features` without `identity`). Re-check it against w2_bh.md before committing; drop
   any bullet F did not land.

## Open issue
BH vision text (`## Relationship to BetterRobot`, kept untouched per brief) still says the humanoid side of BH is
"limited to human<->robot tools (retargeting), with a namespace reserved". That contradicts the owner decision and the
new "Not on the roadmap" line. Suggest editing that sentence (one-line change) in wave 3 or by slice F.
Minor: BR roadmap now has two `### Collision` headings (current limits vs plan); mkdocs gives the second `#collision_1`.
