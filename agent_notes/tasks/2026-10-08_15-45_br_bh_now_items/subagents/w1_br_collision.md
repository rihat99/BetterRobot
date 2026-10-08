# w1 slice D `br_collision` — report

All five items done. Full suite green, my files ruff-clean, mkdocs --strict clean.

## What changed (file: change)
1. Keyword-only ids/pairs (rule: ids after `(model, ...)` are keyword-only, same as agent A)
   - `src/better_robot/terms/robot.py`: `frame_pose(model, *, frame, target, state, q)`,
     `frame_position(model, *, frame, target, state, q)` (were `(model, frame, *, ...)`).
   - `src/better_robot/terms/collision.py`: `self_clearance(spheres, *, pairs, margin, hinge, state)` (was positional `pairs`).
   - Unchanged by design: `self_distances(spheres, state, *, pairs)` was already keyword-only;
     `sphere_clearance(spheres, distance_fn, *, ...)` (`distance_fn` is a function, not an id).
   - Call sites updated: `src/better_robot/terms/ik.py:39`, `tests/kinematics/test_jacobians.py:143`,
     `tests/optim/test_robot_terms.py:30`, `tests/collision/test_collision.py` (self_clearance),
     `examples/02_trajectory.py:47`, `examples/viewer/02_play_trajectory.py:47`, `examples/05_hard_constraints.py:39`,
     `docs/guides/hard-constraints.md:95`, `docs/getting-started/trajectories.md:59`,
     `docs/contracts/05-api-shape.md:58`, `docs/contracts/08-extension-points.md:17` (C-EXT-1 example signature),
     `docs/concepts/collision.md` (self_clearance line).
2. `src/better_robot/collision/robot_spheres.py`: `self_collision_pairs(model, spheres, *, ignore=None)`;
   `ignore` = iterable of `(frame_a, frame_b)` names or ids, either order, resolved through `model.frame_id`
   (bad name -> `UnknownNameError`). Module docstring updated.
3. Same file: `RobotSpheres.to(*, device=None, dtype=None)` (offsets/radii cast, `frame_ids` stay long);
   class docstring now states `offsets (S, 3)` or batched `(B..., S, 3)` broadcast against the state batch.
   No code change needed for batched offsets (`rotation @ offsets[..., None]` already broadcasts); now tested.
4. `docs/contracts/decisions/0008-geometry-brick.md` (new, Accepted, 2026-10-08): RobotGeometry built from IR
   (GeometryModel style, separate from Model), spheres + capsules as robot primitives, lazy trimesh (C-DEP-2) for
   mesh->primitive fitting, scalar distances stay, `*_closest` functions for points/normals, generic contact for
   dynamics in BR, `meta["ir"]` geometry schema frozen until it lands. Row added to `decisions/index.md` (after
   C's 0007 row); linked from `docs/concepts/collision.md` and the new contract-08 section.
5. Docs: `docs/concepts/collision.md`: `to()` + batched offsets paragraph; SDF outside-grid behaviour (clamped
   to boundary, zero gradient, cover the workspace); `ignore=`; new "Extending collision" section (distance_fn
   protocol, ADR 0008). `docs/contracts/08-extension-points.md`: appended section
   "## Collision: obstacles plug in through `distance_fn` (MUST)" at the end of the file (unnumbered, `.no-run`
   snippet), separate from B's optim section.

## Tests added (`tests/collision/test_collision.py`)
- `test_pairs_ignore_frame_pairs`: name/id mixed, both orders; bad name raises `br.exceptions.UnknownNameError`.
- `test_spheres_to_and_batched_offsets`: dtype/device move keeps long ids; offsets `(2, S, 3)` against a B=2
  state give per-sample centres equal to the unbatched result row by row; `self_distances` shape `(2, P)`.

## Gates
- Slice: `pytest tests/collision tests/optim/test_robot_terms.py tests/kinematics/test_jacobians.py` -> 22 passed (before the 2 new tests); `tests/collision` -> 10 passed.
- Full: `1621 passed, 12 skipped, 7 deselected, 33 warnings in 195.63s` (includes other agents' in-flight edits).
- Ruff: whole repo `Found 86 errors` (brief says 82 pre-existing). Every file I touched is clean
  (`ruff check` on terms/, collision/, tests/collision, tests/kinematics/test_jacobians.py,
  tests/optim/test_robot_terms.py -> "All checks passed!"); the +4 come from other slices' concurrent edits.
- `mkdocs build --strict`: built, no warnings. INFO only: `0008-geometry-brick.md` not in `nav`.

## Outside my slice (not edited; orchestrator please)
- `mkdocs.yml` nav (~line 161): add `- 0008 Geometry brick: contracts/decisions/0008-geometry-brick.md`
  (and C's 0007, also missing from nav).
- `docs/contracts/08-extension-points.md` intro says "There are three seams"; with B's sink seam and the
  collision section it should list them (left for whoever merges B's section). The collision section is
  unnumbered to avoid a `C-EXT-7` clash with B; number it when merging if desired.
- `docs/reference/changelog.md`: breaking entries for `frame_pose`/`frame_position(frame=)` and
  `self_clearance(pairs=)` not written (not in my file list).
- Examples were edited (call sites only) though `examples/` is A's slice; single-line changes, no conflict expected.

## BH call sites that break (wave 2; not edited)
- Any BH call of `terms.frame_pose(model, X, target=...)`, `terms.frame_position(model, X, ...)` or
  `terms.self_clearance(spheres, pairs, ...)` with positional id/pairs. Check with
  `grep -rn "frame_pose(\|frame_position(\|self_clearance(" /home/rikhat.akizhanov/better/dev/BetterHuman/src /home/rikhat.akizhanov/better/dev/BetterHuman/tests | grep "terms\."`
  -> no matches, so no known BH break from this slice.
