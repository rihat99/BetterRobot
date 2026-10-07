# P1 violations ledger (ruff findings, contract-test allowlists, p0 §3 items; see p1_state.md)

Counts from `uv run ruff check --select <code> src` on 2026-10-07. Codes enabled in pyproject with open violations:

```
Found 25 errors.
src/better_robot/data_model/joint_models/revolute.py:165:9: ANN201 Missing return type annotation for public function `joint_motion_subspace`
src/better_robot/data_model/joint_models/revolute.py:170:9: ANN201 Missing return type annotation for public function `joint_velocity`
src/better_robot/data_model/joint_models/revolute.py:31:5: ERA001 Found commented-out code
src/better_robot/__init__.py:45:5: ERA001 Found commented-out code
src/better_robot/__init__.py:52:5: ERA001 Found commented-out code
src/better_robot/__init__.py:55:5: ERA001 Found commented-out code
src/better_robot/__init__.py:57:5: ERA001 Found commented-out code
src/better_robot/__init__.py:64:5: ERA001 Found commented-out code
src/better_robot/io/parsers/urdf.py:118:24: BLE001 Do not catch blind exception: `Exception`
src/better_robot/optim/constrained.py:15:37: FBT001 Boolean-typed positional argument in function definition
src/better_robot/optim/problem.py:171:9: ANN201 Missing return type annotation for public function `cached`
src/better_robot/optim/problem.py:226:9: ANN201 Missing return type annotation for public function `with_`
src/better_robot/spatial/inertia.py:58:9: ERA001 Found commented-out code
src/better_robot/viewer/layer.py:62:27: FBT001 Boolean-typed positional argument in function definition
src/better_robot/viewer/layers/plot.py:52:27: FBT001 Boolean-typed positional argument in function definition
src/better_robot/viewer/render/rasterizer.py:144:59: FBT001 Boolean-typed positional argument in function definition
src/better_robot/viewer/timeline.py:19:64: FBT001 Boolean-typed positional argument in function definition
src/better_robot/viewer/timeline.py:19:64: FBT002 Boolean default positional argument in function definition
src/better_robot/viewer/tools/gizmo.py:94:24: BLE001 Do not catch blind exception: `Exception`
src/better_robot/viewer/viewer.py:145:41: FBT001 Boolean-typed positional argument in function definition
src/better_robot/viewer/viewer.py:194:20: FBT001 Boolean-typed positional argument in function definition
src/better_robot/viewer/viewer.py:194:20: FBT002 Boolean default positional argument in function definition
src/better_robot/viewer/viewer.py:207:48: FBT001 Boolean-typed positional argument in function definition
src/better_robot/viewer/viewer.py:207:48: FBT002 Boolean default positional argument in function definition
src/better_robot/viewer/viewer.py:213:37: FBT001 Boolean-typed positional argument in function definition
```

Deferred codes (commented in pyproject): N 113, D 305, ANN001 132, SLF001 31.

Still to add (not done in P1): keyword-only violations (C-API-2), frozen/in-place/dtype findings, import-DAG edges, items from p0_architecture_br §3.

## C-API-2 keyword-only violations (allowlist in `tests/contract/test_keyword_only.py`, 2026-10-07)

Positional count after `self`; limit is 2. Paths under `src/better_robot/`. Re-exports at the top level
(`better_robot.frame_jacobian` etc.) are the same functions.

| package | callable | positional | file:line |
|---|---|---|---|
| kinematics | `frame_jacobian`, `joint_jacobian` | 3 | `kinematics/state.py:183`, `:206` |
| kinematics | `frame_jacobian_time_variation`, `joint_jacobian_time_variation` | 4 | `kinematics/state.py:222`, `:241` |
| dynamics | `inverse_dynamics`, `forward_dynamics` | 4 | `dynamics/state.py:55`, `:83` |
| dynamics | `centroidal_momentum` | 3 | `dynamics/state.py:109` |
| lie | `umeyama` | 3 | `lie/alignment.py:67` |
| collision | `sphere_sphere`, `sphere_halfspace` | 4 | `collision/distance.py:20`, `:40` |
| collision | `sphere_capsule`, `sphere_box`, `sphere_sdf` | 5 | `collision/distance.py:27`, `:47`, `:63` |
| collision | `self_distances` | 3 | `collision/robot_spheres.py:90` |
| io | `ModelBuilder.add_collision_geom` | 4 | `io/parsers/programmatic.py:129` |
| optim | `Manifold.tangent_bounds`, `Euclidean.tangent_bounds` | 3 | `optim/manifold.py:30`, `:49` |
| optim | `run` | 3 | `optim/phases.py:48` |
| terms | `RobotManifold.tangent_bounds` | 3 | `terms/robot.py:40` |
| terms | `ik_problem`, `random_start`, `solve_ik` | 3 | `terms/ik.py:15`, `:47`, `:58` |

## C-STATE-1 non-frozen public dataclasses (allowlist in `tests/contract/test_frozen.py`)

- optim: `Phase`, `Result` (`optim/phases.py:14`, `:37`) — plain records, should be `frozen=True`.
- viewer: `MeshSpec` (`viewer/`, exported in `viewer.__all__`) — plain record, should be `frozen=True`.
- Accepted as lifecycle objects (C-API-1), documented in the allowlist: `optim.State` (`optim/solvers.py:19`),
  `io.IRBody/IRFrame/IRGeom/IRJoint/IRModel` (`io/ir.py`).

## Enforcers still missing (00-index marks them "pending, ledger")

- `tests/contract/test_dtype_device.py` (C-TENS-1, C-TENS-2) and `tests/contract/test_no_inplace.py`
  (C-STATE-3, C-DIFF-2/3): need a small entry-point registry (model fixture + per-function argument factory);
  not written in P1.

## p0_architecture_br.md §3 items the contracts now forbid

- C-NAME (naming): viewer `frame` = time step (`viewer/layer.py:52`, `num_frames`) vs robot frame elsewhere;
  viewer `n_joints` (`viewer/layers/skeleton.py:52`) vs `njoints`.
- C-API-4 (`int | str` ids): `KinematicState.frame_pose_of(frame_id: int)` int-only (`kinematics/state.py:146`);
  private name lookup `terms/robot.py:58 _frame_id`; `ik_problem` targets keyed by name only (`terms/ik.py:17`).
- C-API-2/3 (signatures): `terms/collision.py:19,42` positional `margin`, `hinge`, `state`;
  `random_start(model, q0, generator)`; `solve_layer(problem, solver, iters=50, x=None, damping=...)`
  (`optim/phases.py:81`) vs keyword-only `solve` (`:76`).
- C-STATE-1/2/4 (mutability): `optim.Problem` mutable registry with `with_` view; `Phase.on_start` writes
  `problem.data` (`optim/phases.py:20`); `KinematicState._memo` mutable dict inside frozen dataclass
  (`kinematics/state.py:110`); `dynamics/_world` cache on `ModelValues` invalidated by in-place edits
  (`dynamics/state.py:39`).
- C-NAME (one name per thing): `optim.State` vs `KinematicState` vs cache key `"state"` (`terms/ik.py:36`).
- C-ERR: built-in exceptions raised 111x (heaviest `spatial/inertia.py`, `optim/problem.py`,
  `io/builders/kinematic_tree.py`, `lie/alignment.py`, `data_model/model_structure.py`, `io/parsers/programmatic.py`);
  `IRError(ValueError)` not a `BetterRobotError` (`io/ir.py:97`); never raised: `ConvergenceError`,
  `SingularityWarning`, `UnsupportedJointError` (`exceptions.py:79,101,108`).
- C-DOC: Sphinx roles `:class:`/`:func:` 47x in 15 files (e.g. `kinematics/state.py:152`, `exceptions.py`,
  `io/build_model.py`).
- C-DEP-3/C-EXT-3 (private leaks): `dynamics/_world.py:30` imports `kinematics._jacobian_columns`; tests import
  `kinematics._fk_lean.lean_core`, `io.build_model._check_topology_invariants`, `viewer._math`,
  `viewer.robot._torque_colors`; duplicate plan caches `get_plan`/`build_plan` (`kinematics/_fk_lean.py:128,191`).

## Ledger entries referenced by `[planned: L-<n>]` tags on contract pages (P1c, 2026-10-07)

- L-1 (C-EXT-2): consumers still `isinstance` on joint classes: `kinematics/state.py:84`, `io/build_model.py:387`,
  `io/parsers/programmatic.py:394`, `data_model/joint_models/composite.py:32`. A joint-type registry needs an ADR.
- L-2 (C-API-4): widen `Model.frame_id` (`data_model/model.py:195`) to `int | str`; route `KinematicState.frame_pose_of`
  (`kinematics/state.py:146`), `terms/ik.py:17` targets and `terms/robot.py:58 _frame_id` through it.
- L-3 (C-DEP-1): make `kinematics._jacobian_columns` public-internal (`kinematics.jacobian_columns` or re-export) so
  `dynamics/_world.py:30` imports a public name; then rank dynamics above kinematics in `test_layer_dependencies.py:23`.
- L-4 (C-TENS-4): widen `tests/contract/test_hot_path_lint.py:28` scan to `spatial` and `terms`; add `# bench-ok`
  on once-per-model plan building (`kinematics/_fk_lean.py:51 .tolist()`).
- L-5 (C-STATE-1): freeze `optim.Phase` and `optim.Result` (`optim/phases.py:15,38`); `Phase.on_start` writes
  `problem.data` (`:20`) must move to a returned value.
- L-6 (C-DIFF-1): `dynamics/state_manifold.py:64,72` `.detach()` — add a `Differentiability` docstring section or
  remove the detach.
- L-7 (C-NAME-1..3, C-EXT-3, C-DOC-1..3): ruff `N`, `SLF001`, `D`, `ANN001` are cited on pages but not in
  `pyproject.toml` select (would add N806 x96, N803 x17, SLF001 x31, D102 x164, D107 x35, ANN001 x132). Enable after
  the polish pass clears them; per-file ignores for math packages do not exist yet.
- L-8 (C-DIFF-2): write `tests/autograd/test_vmap.py` (vmap over a leading axis for each public entry point).
- L-9 (C-DOC-4): 50 Sphinx roles (`:class:`, `:func:`, ...) in `src/better_robot` docstrings render as literal
  text under MkDocs (count: `grep -rEo ':(class|func|meth|mod|attr|obj|ref):' src/better_robot | wc -l`). Replace
  with backticks or mkdocstrings references, then add a no-role check to `tests/contract/test_docstrings.py`.

## P2b status (2026-10-07, non-viewer polish; commit pending)

Ruff `src` (incl. viewer) 58 -> 35 findings (measured in this worktree while the viewer agent also edited);
`src tests` is 83 after (no clean baseline measured; of the 35 in `src`, 15 are intended PLC0415 lazy imports).

- FIXED ruff block above, non-viewer rows: ERA001 x7 (`revolute.py:31`, `__init__.py` x5, `inertia.py:58` — comments
  reworded, no code lost), ANN201 x4 (`revolute.py` x2, `optim/problem.py cached/with_`), BLE001 `urdf.py:118`
  (`except (OSError, ValueError)`), FBT001 `optim/constrained.py:15` (`inequality` keyword-only), F401 x6
  (`free_flyer.py`, `io/assets.py` x2, `urdf.py` x2, `spatial/force.py`) + `tests/bench/bench_forward_kinematics.py`.
- LEFT ruff (not in the brief's rule list): viewer rows (other agent); `kinematic_tree.py` E741 x2; `mjcf.py`/`urdf.py`
  PLR0912/PLR0915 x7; `mjcf.py:96` F841; `spatial/force.py:40`, `motion.py:65` F821 (forward refs); PLC0415 x15 lazy.
- C-API-2 keyword-only (P2c pass, 2026-10-07): FIXED 21 of 30 allowlist rows (`tests/contract/test_keyword_only.py`, now 9 rows).
  Arguments after the second are keyword-only in: `collision.sphere_sphere/sphere_capsule/sphere_halfspace/sphere_box/sphere_sdf`
  (`collision/distance.py`), `collision.self_distances(spheres, state, *, pairs)`, `ModelBuilder.add_collision_geom(body, kind, *, params, origin)`,
  `Manifold/Euclidean/RobotManifold.tangent_bounds(x, *, lower, upper)`, `optim.run(problem, phases, *, x=None)`,
  `terms.random_start(model, q0, *, generator=None)`, `terms.ik_problem/solve_ik(model, targets, *, q0, ...)`,
  `kinematics.frame_jacobian_time_variation/joint_jacobian_time_variation(model, state, *, v, frame|joint, reference=...)`,
  `dynamics.centroidal_momentum(model, state, *, v)`. Callers updated in src/tests/examples/docs/README (`tests/contract/_public_ops.py`
  case table included). BH `src/` has no call site of any changed function (grep 2026-10-07).
  LEFT (9 rows): `frame_jacobian`, `joint_jacobian` (x2 each: top-level + kinematics), `inverse_dynamics`, `forward_dynamics`
  (x2 each), `lie.umeyama` — 25/9/54/15/12 BR call sites; `inverse_dynamics` has 1 BH call site.
- FIXED L-5 / C-STATE-1: `optim.Phase`, `optim.Result` now `frozen=True`; allowlist rows removed from
  `tests/contract/test_frozen.py`. LEFT: `Phase.on_start` writing `problem.data` — it writes the phase's own
  `with_` view, never the caller's `Problem` (docstring already says so); moving it to a return value changes the
  callback signature used by callers -> owner decision.
- FIXED C-ERR: `IRError(BetterRobotError, ValueError)` (`io/ir.py`); `ConvergenceError`, `SingularityWarning`,
  `UnsupportedJointError` removed from `exceptions.py` and `__all__` (no mentions in docs/, BH, or src);
  `tests/test_exceptions.py`: dropped `test_singularity_warning_is_filterable` and two parametrize rows (subjects gone).
- FIXED L-9 / C-DOC-4: 49 Sphinx roles replaced by backtick spans in 15 non-viewer files; 1 role left in `viewer/`.
  Not done: the no-role check in `tests/contract/test_docstrings.py`.
- FIXED L-3 / C-DEP-1: `better_robot.kinematics.jacobian_columns` public-internal alias of `_jacobian_columns`
  (`kinematics/__init__.py`, commented, not in `__all__`); `dynamics/_world.py` imports it. Not done: re-rank
  dynamics above kinematics in `test_layer_dependencies.py`.
- LEFT L-2 (`int | str` ids), L-1, L-4, L-6, L-7, L-8, C-API-2/3 signature rows, C-STATE-2/4, C-NAME rows: not started.
