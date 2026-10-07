# P0 — BetterRobot architecture and convention audit (mars_aux, 2026-10-07 17:00)

Paths relative to `src/better_robot/` unless absolute. Read-only audit (greps + reading); test suite not run.

**Verdict:** layering and core data model are clean and enforced by tests. Main inconsistencies: the word
"frame" means two things (robot frame vs time step); typed errors only partly used, three never raised; term
signatures uneven; `optim` wires pieces by string names; Sphinx roles still in docstrings although banned.

## 1. Package map and import order
- Ranks in `tests/contract/test_layer_dependencies.py:23-35`: _typing 1, lie 2, spatial + optim 3, data_model 4,
  io/kinematics/dynamics/collision 5, terms 6, viewer 7. `optim` imports only torch. Matches
  `docs/concepts/architecture.md:20-33`.
- Optional deps pinned: viser only in `viewer/viewer.py`, trimesh only in `viewer/robot.py`; no pyrender/imageio/pypose.
- Lines: io ≈2.6k (`build_model.py` 1165), data_model ≈2.2k, viewer ≈2.1k, optim 1.75k, lie ≈1.1k, kinematics 0.96k,
  spatial 0.9k, dynamics 0.5k, terms 0.33k, collision 0.2k.
- `__all__`: data_model {Model, ModelStructure, ModelValues, Frame, Body, Joint, JOINT_DIMENSIONS}; kinematics
  {KinematicState, forward_kinematics_state, frame/joint_jacobian(+_time_variation)}; dynamics {inverse_dynamics,
  mass_matrix, forward_dynamics, centroidal_map, centroidal_momentum, center_of_mass}; optim 24 names (Problem,
  TermFn, Phase, Result, State, solvers, kernels, solve, run, solve_layer, solve_constrained, check_jacobian);
  terms {RobotManifold, ik_problem, solve_ik, frame_pose, frame_position, joint_limits, rest, finite_difference,
  sphere/self_clearance}. `spatial/__init__.py:15` re-exports SE3/SO3/Pose from `lie` → two public import paths.

## 2. Conventions already followed
- `docs/contributing/conventions.md:7-19`: `njoints`/`nq`/`nv`, `_raw` suffix, booleans `is_`/`has_`.
- Pose `[t, qxyzw]`, twist `[linear, angular]` (conventions.md:26). Typed errors, validate once at the boundary (:32-38).
- Frozen dataclasses: Model (`data_model/model.py:53`), KinematicState (`kinematics/state.py:102`), SE3/SO3,
  Motion/Force/Inertia, Term/TermFn. New Model via `Model.with_values` / `.to` (`model.py:259,298`).
- `reference=` is a `Literal` string everywhere (`kinematics/state.py:188,211,228,247`); no enum. Consistent.
- Quaternion-norm check opt-in because it forces a host sync (`kinematics/state.py:77-95,151`).

## 3. Inconsistencies
- **"frame" = time step in viewer** (`viewer/layer.py:52 set_frame(frame)`, `num_frames` ×42) vs robot frame in
  terms/kinematics (`terms/robot.py:62 frame: int | str`); Model uses `nframes` (`model.py:79`). Viewer uses
  `n_joints` (`viewer/layers/skeleton.py:52`, 4 places) vs `njoints` elsewhere (86 places).
- **Frame-id types differ**: `KinematicState.frame_pose_of(frame_id: int)` int only (`kinematics/state.py:146`);
  `frame_jacobian` and terms take `int | str`; name lookup is private `terms/robot.py:58 _frame_id`; `ik_problem`
  targets keyed by name only (`terms/ik.py:17`).
- **Term signatures uneven**: `terms/robot.py` terms take `model` first, keyword-only `state=`, `q=`;
  `terms/collision.py:19,42` take `spheres` first with `margin`, `hinge`, `state` positional;
  `random_start(model, q0, generator)` positional generator (`terms/ik.py:47`), `solve_ik` keyword-only.
- **Keyword-only rules differ**: `solve(problem, solver, *, iters, x)` (`optim/phases.py:76`) vs
  `solve_layer(problem, solver, iters=50, x=None, damping=...)` not keyword-only (`phases.py:81`).
- **Mutability mixed in optim**: Problem is a mutable registry (`residual`/`scalar`/`cached`) yet `with_` returns a
  view; `State` (`optim/solvers.py:19`), `Phase`, `Result` (`phases.py:14,37`) not frozen; `Phase.on_start` may
  write `problem.data` (`phases.py:20`). `KinematicState` frozen but holds mutable `_memo` dict
  (`kinematics/state.py:110`) — compile-unfriendly (not verified). `dynamics/_world` caches on ModelValues,
  "invalidated by in-place edits" (`dynamics/state.py:39`) → Model only shallowly frozen.
- **"State" means three things**: `optim.State`, `KinematicState`, and the cache name `"state"` that terms look up
  (`terms/ik.py:36`): user must call `problem.cached(kinematic_state(model), name="state")` with exactly that
  name or terms silently find nothing.
- **Error types**: built-ins raised 111× (ValueError 81, TypeError 22, KeyError 5, NotImplementedError 2,
  RuntimeError 1); heaviest: `spatial/inertia.py` 11, `optim/problem.py` 11, `io/builders/kinematic_tree.py` 12,
  `lie/alignment.py` 10, `data_model/model_structure.py` 10, `io/parsers/programmatic.py` 8.
  `IRError(ValueError)` (`io/ir.py:97`) is not a `BetterRobotError`. Never raised: `ConvergenceError`,
  `SingularityWarning`, `UnsupportedJointError` (`exceptions.py:79,108,101`).
- **Sphinx roles** (`:class:`, `:func:`) 47× in 15 files despite `conventions.md:70` (e.g. `kinematics/state.py:152`,
  `exceptions.py`, `io/build_model.py`).
- **Private code leaks**: `dynamics/_world.py:30` imports `kinematics._jacobian_columns`. Tests import
  `kinematics._fk_lean.lean_core`, `io.build_model._check_topology_invariants`, `viewer._math`,
  `viewer.robot._torque_colors`. Two plan caches `get_plan`/`build_plan` in `kinematics/_fk_lean.py:128,191` and
  `_jacobian_columns.py:50,88`. `viewer/_math.py:28 compose_poses` re-implements quaternion compose in NumPy.
- **Missing return types**: `Problem.cached`, `Problem.with_` (`optim/problem.py:171,226`). `Term.kind` plain `str`
  not `Literal` (`problem.py:52`).
- **Host syncs**: `.tolist()` in `kinematics/_fk_lean.py:51`, `collision/robot_spheres.py:81` (build-time, not hot).
- Free-flyer check only looks at `joint_models[1]` (`kinematics/state.py:84`).

## 4. Legacy leftovers
- `src`: no `optim2`, `create_data`, `Data(`, `tasks`/`residuals`, Warp, TODO/FIXME/XXX.
- `rnea`/`crba`/`aba` survive only in `tests/test_pinocchio/` filenames and docstrings; docstrings mention
  `bias_forces(q, v)` (not public) at `test_crba_matches_pinocchio.py:5,41`, `test_rnea_matches_pinocchio.py:120`.
- No dead public functions found; dead code = the 3 unused exceptions.

## 5. optim
- `Problem(variables, manifolds, bounds, data, time)` + `.residual/.scalar/.cached`; `TermFn` = fn, reads,
  optional jac, window (`problem.py:27-43`); Manifold protocol tangent_dim/retract/project/tangent_bounds
  (`manifold.py:15`). Solvers `init`/`step`; `TorchOptim` wraps torch.optim. `run` with Phases detaches start
  values (`phases.py:50`) → not differentiable; `solve_layer` is the differentiable path. `solve_constrained` is an
  outer penalty loop rewriting terms with `dataclasses.replace` (`constrained.py:15-25`).
- Awkward downstream: wiring by string name (`reads`, cache names, `target/<frame>` data keys); Jacobians from
  `linearize` not reusable without rebuilding normal equations; a new term must know the cache name `"state"`.
- vs references: Crocoddyl separates residual model from cost model (≈ TermFn + weight/kernel); Theseus `Objective`
  is also a mutable name-keyed registry; Drake `MathematicalProgram` uses typed variable handles + Bindings.

## 6. API sugar candidates
1. `KinematicState.frame_pose_of(int | str)` (uses existing `model.frame_id`, `model.py:195`).
2. `frame_pose_of` accepting a list of names.
3. `Problem.cached` default `name=` from fn; terms take the cached object not the string `"state"`.
4. `model.fk(q)` alias of `forward_kinematics_state(model, q)`.
5. `ik_problem` target keys `int | str`.
6. `Result.q` / `result["q"]` instead of `result.x["q"]`.
7. `solve_layer` and `random_start` keyword-only (match `solve`, `solve_ik`).
8. `br.load(path, *, dtype, device)` (check whether it already has them).
9. Export `optim.L2` (`kernels.py:24`, defined, not in `__all__`).
10. `terms.ik_state(model)` for `problem.cached(kinematic_state(model), name="state")`.

## Open questions
- Rename viewer time-step `frame`/`num_frames` → `step`/`num_steps`? (API change)
- `IRError` → subclass `BetterRobotError`?
- Remove or start raising the 3 unused exceptions?
