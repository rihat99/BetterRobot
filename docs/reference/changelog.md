# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each release groups its entries by change type;
every removal names its replacement. The [migration page](migration-0.3.0.md) maps every public name of the 0.2 line
that 0.3.0 renames or removes, one table row each.

## Unreleased

### Added
- `better_robot` exports its twelve packages (`exceptions`, `lie`, `spatial`, `model`, `kinematics`, `dynamics`,
  `collision`, `io`, `optim`, `terms`, `viewer`, `testing`) and `JointFreeFlyer`, `solve_ik` and `Viewer` beside the
  core functions.
- `model.Geometry`: the shapes of a robot as a value beside `Model`, made of final records `Sphere`, `Box`,
  `Cylinder`, `Capsule` and `Mesh` attached to bodies by `AttachedShape(body, shape, collision)`; built by
  `io.build_geometry(ir, model)` or `io.load_geometry(path, model)`, extended by `ModelBuilder.add_shape`.
- `model.JointRevolute(axis)` and `model.JointPrismatic(axis)` for any unit axis; each joint class declares its
  `model.JointManifold`, and `kinematics.integrate(model, q, v)` and `kinematics.difference(model, q0, q1)` step
  and compare whole configurations through it.
- `Model.structure_key` (a hashable value of the static fields), `Model.device` and `Model.dtype`.
- `kinematics.subtree(model, joint)`, `kinematics.support(model, joint)`, `kinematics.full_configuration(model, q)`
  (mimic joints filled in) and `kinematics.euclidean_columns(model)`.
- `kinematics.point_jacobian(model, state, *, frames, points)`: the `(B..., K, 3, nv)` world-velocity Jacobian of
  points fixed to frames.
- `kinematics.frame_jacobian_time_variation` and `kinematics.joint_jacobian_time_variation`: the `J̇` of
  `a = J v̇ + J̇ v` in the world, local and local-world-aligned references.
- `KinematicState.joint_pose_of(joint)`, the world pose of one joint; `KinematicState.jacobian_columns`, the world
  Jacobian column table every joint and frame Jacobian shares; `KinematicState.world_geometry` for `dynamics`.
- `spatial` functions on tensors: `cross_motion`, `cross_force`, `transform_motion`, `transform_force`,
  `transform_inertia`, `inertia_matrix`, `rotational_inertia`, `inertia_from_mass_com` and `inertia_of_sphere`,
  `inertia_of_box`, `inertia_of_capsule`, `inertia_of_ellipsoid`, `inertia_of_mesh`.
- `io.parse_urdf` and `io.parse_mjcf` (through MuJoCo's compiler, the `mjcf` extra); `io.IRShape` and
  `IRModel.shapes`; `IRJoint.damping`, `IRJoint.armature` and `IRJoint.neutral`.
- `exceptions.ArgumentError` (also a `ValueError`) for an argument outside its documented values.
- `collision`: `RobotSpheres` and `RobotCapsules` (frame-attached spheres and segments), `self_collision_pairs`
  (with `ignore=`), `self_distances`, `self_capsule_distances` and the pair distances `sphere_sphere`,
  `sphere_capsule`, `capsule_capsule`, `sphere_halfspace`, `sphere_box`, `sphere_sdf`.
- `optim`: named-tensor problems of plain-function terms. `Problem` with `residual`, `scalar` and `cached` terms
  (`TermFn`, `Term`, `RobustKernel` and the kernels `Huber`, `Cauchy`, `Tukey`, `GemanMcClure`), the solvers
  `LevenbergMarquardt`, `GaussNewton`, `LBFGS` and `TorchOptim` (an abstract `Solver` with `warm_keys`), `solve`,
  `run` with `Phase`s, `solve_layer` (differentiable), `solve_constrained`, `check_jacobian`, the `Normal`,
  `BandedNormal` and `ArrowNormal` linear systems, the `Manifold` base and `Euclidean`.
- `Problem.evaluate(x)` returns the loss, the per-term losses and the `info` a term returns beside its value;
  `Problem.gradient(x)` adds the per-term losses; `Problem.cached_values` lists the cached values.
- `State.solver`, `State.extra["terms"]` and `Result.diagnostics` (`grad_norm`, `step_norm`, `accepted`, LM
  `damping`); `Result.state` warm-starts `solve` and `run` through `state=`.
- `terms`: `frame_pose`, `frame_position`, `joint_limits`, `rest`, `finite_difference` (any `order`,
  `world_translation=`), `velocity_limit(model, *, dt, fraction=1.0)`, `sphere_clearance`, `self_clearance`,
  `kinematic_state`, `RobotManifold`, `ik_problem`, `random_start` and `solve_ik`.
- `viewer`: `Viewer` (viser) with `add`, `replace`, `remove_group`, `set_offset`, `set_visible`, `set_fps`,
  `aim`, `play`, `record`, `save`, `on_pointer` and `clear_pointer`; a `Timeline`; the layers `FramesLayer`,
  `PointsLayer`, `SpheresLayer`, `ArrowsLayer`, `MeshLayer`, `RigidMeshLayer`, `SkinnedMeshLayer`,
  `SkeletonLayer`, `GroundLayer`, `CameraLayer`, the panels `ImagePanel` and `PlotPanel`; `Theme`, `LIGHT`,
  `DARK`; `robot_meshes` and `robot_skeleton`.
- `Viewer.add(..., times=, valid=)`: per-layer sample times, and a per-step (or per-element) mask that hides
  samples; `Layer.set_opacity` and `Layer.opacity`; `Layer.elements`.
- `robot_skeleton(..., tau=)` shows each joint's `|tau| / effort_limit` in colour (`Theme.torque_ramp`,
  `Theme.over_limit`) and thickness; `end_points=` adds bones out to points in a frame; co-located joints are drawn
  apart along their axes.
- `ArrowsLayer(shaft_radius=, head_radius=, min_length=)` draws solid arrows and skips vectors shorter than
  `min_length`; `GroundLayer(origin=, normal=, fill=, fill_color=)` draws a tilted, filled floor.
- `better_robot.testing`: `FakeServer` (viser without sockets), `check_joint_model`, `public_api`,
  `public_objects`, `imported_packages`, `dead_citations`, the docs snippet runner `snippets` and the `no_skips`
  pytest plugin; the `testing` extra installs pytest for it.
- `FakeServer.gui` declares the GUI calls of an app built on the viewer, with viser's keywords: `add_dropdown`,
  `add_text`, `add_number`, `add_markdown`, `add_tab_group` (tabs by `add_tab`) and `add_slider(..., disabled=)`;
  each control reads back the value it was given.
- `tests/contract/public_api.json`, the tracked public API, kept by `tools/api_snapshot.py`.

### Changed
- `Model` is one flat frozen record: the fields of `ModelStructure` and `ModelValues` live on it, it compares by
  identity, and `Model.with_values(**fields)` replaces any tensor field by keyword.
- Field names say what they hold: `Model.parent_joint_ids`, `Model.mimic_source_ids`, `Model.mimic_multipliers`,
  `Model.mimic_offsets`, `Model.lower_position_limit`, `Model.upper_position_limit`, `Model.joints`; the
  `KinematicState` accessors are plural (`joint_poses`, `frame_poses`, `joint_rotations`, ...).
- `build_model` orders joints parents first and keeps the file's order wherever it already is: joint ids change
  for some robots (the G1). Joint 0 is a `JointFixed` named `"universe"`.
- Every body's frame carries the body's name (`"panda_hand"`, not `"body_panda_hand"`); a frame named like a body
  or another frame is a `ModelError`.
- `Model.q_neutral` is clamped into the joint limits at build.
- `load(path, *, format=None, root_joint=None, device=None, dtype=torch.float32)` and
  `parse(path, *, format=None)` take a path only; `root_joint=JointFreeFlyer()` makes a floating base.
- `ModelBuilder.add_revolute` and `add_prismatic` take `axis=` as a unit 3-vector; `add_frame` takes `parent=`;
  `add_joint` takes `kind=` as a joint object.
- `IRJoint.joint` holds the joint object (with its axis and pitch).
- Keyword renames: `iters=` is `iterations=` (`solve`, `solve_layer`, `solve_constrained`, `Phase`, `Result`,
  `solve_ik`), `fext=` is `external_forces=` (`inverse_dynamics`, `forward_dynamics`), `jac=` is `jacobian=`
  (`TermFn`, `Problem.residual`), `gtol`, `ftol`, `xtol` are `gradient_tolerance`, `loss_tolerance`,
  `step_tolerance`, `State.it` is `State.iteration`.
- `TorchOptim(factory, *, scheduler=None)` takes a function from the parameters to a `torch.optim.Optimizer`.
- `Phase.on_start(problem, x)` returns `(x, data)`; `Problem.with_(*, data, weights, free)` replaces data entries.
- `Solver` and `optim.Manifold` are abstract base classes; `RobustKernel` is the kernels' base.
- `Model.random_configuration(*, generator)` and `terms.random_start(model, q0, *, generator)` require the
  generator.
- `forward_kinematics(..., check_quaternion_norm=True)` checks every quaternion joint.
- `kinematics.integrate` raises `DtypeMismatchError` on mixed dtypes.
- Public dataclasses and constructors are keyword-only after their leading arguments: `KinematicState`,
  `RobotSpheres`, `RobotCapsules`, `Normal`, `BandedNormal`, `ArrowNormal`, `TermFn`, `Phase`, `State`, `Result`,
  the IR records, `MeshSpec`, `Theme`, `Timeline`, and the layers (`SkeletonLayer(..., parent_ids=)`).
- `Viewer.add(layers, *, group)` takes an iterable of layers; `Viewer.play(*, block=True)` pauses the timeline when
  the serve ends.
- A second `Viewer.on_pointer` registration raises `ArgumentError` (`clear_pointer` first); the pointer callbacks
  receive viser's click and rect-select events.
- `Viewer.add` and `Viewer.replace` reject a layer added twice with `ArgumentError`, and a failing layer leaves
  the viewer as it was; `set_offset` and `remove_group` raise `UnknownNameError` for an unknown group.
- A raising `PoseGizmo` callback stops the gizmo; `wait()` raises the callback's error.
- `forward_kinematics` returns canonical-sign, normalized quaternions (`q` and `-q` are one rotation): compare
  rotations, or align signs, before comparing raw components.
- `inverse_dynamics` and `mass_matrix` include `Model.armature` (motor inertia; MJCF `armature`, zero from URDF),
  so `forward_dynamics` does too, as in Pinocchio.
- URDF `<dynamics>` and MJCF `damping` and `armature` reach `Model.damping` and `Model.armature`.
- `inverse_dynamics`, `forward_dynamics`, `RobotSpheres.centers` and `RobotCapsules.segments` have bitwise
  reproducible CUDA gradients (`tests/collision/test_collision.py::test_cuda_backward_is_bitwise_reproducible`).
- `LBFGS` evaluates the first line-search trial with its gradient and stops the search once every problem has a
  step; with bounds it holds the coordinates an active bound pins and tests the Armijo condition on the clipped
  step; it tries up to 20 step sizes.
- `Model.name` is the robot name of the file (`<robot name=...>`, `<mujoco model=...>`).
- The URDF and MJCF parsers raise `ModelError` naming the file and the element at fault.
- Errors are the `exceptions` classes throughout, `optim` included.
- Tensor arguments of the `lie` and `spatial` functions whose batch axes do not broadcast raise `ShapeError` naming
  both, not torch's `RuntimeError`; so do `so3.slerp`'s tensor `t` and `sphere_halfspace`'s tensor `offset`.
- `ik_problem` and `solve_ik` check `q0` as `(B, nq)` and each `targets[<frame>]` as `(..., 7)` in the model's dtype
  and device, with a batch that broadcasts with `q0`'s, and name the argument at fault.
- `Problem` raises `DtypeMismatchError` for a variable that is not float32 or float64 or for variables in two
  dtypes, and `DeviceMismatchError` for two devices; a term output of the wrong shape raises `ShapeError` naming the
  term; `iterations=` and `damping=` below 0 raise `ArgumentError`.
- `RobotSpheres` checks its tables at construction (`frame_ids` `(S,)` long, `offsets` `(..., S, 3)`, `radii`
  `(S,)`); `centers` and `self_distances` raise `ArgumentError` for a frame or sphere index outside its table.
- `KinematicState` checks the shapes of `q` and its transform tables at construction (`ShapeError`).
- `io.ModelBuilder`, `optim.Problem`, the solvers and `Euclidean` print a one-line repr with their settings.
- The `no_skips` plugin makes a run with a skipped, xfailed or xpassed test exit nonzero.
- Python 3.12 or newer; the `io-mjcf` extra is `mjcf`; the `viewer` extra needs `viser>=1.0.30` and `pillow`;
  the `dev` extra is a dependency group, beside `test-oracles` for the parity oracles.
- `optim.State` is a frozen dataclass: assigning to a field raises `FrozenInstanceError` (build a new state with
  `dataclasses.replace`).
- `testing.snippets` is stricter: a printed float with no fraction digits must match exactly and any other within
  half a unit of its last digit; `py`, `pycon` and `~~~python` fences, a blank `data-reason`, a printing fence
  without an output fence and a fence that exits early are reported.
- The contributing pages are gone: [extending](../guides/extend.md) and [running the tests](../guides/run-the-tests.md)
  are guides, and [the test suite](tests.md) is reference.
- `benchmarks/run.py` is the one benchmark runner: the full tier fails when a graph break, recompile, operator or
  host sync of its counter rows (forward kinematics, Jacobian, RNEA, CRBA, ABA, `solve_ik`) exceeds
  `benchmarks/thresholds/torch-<version>.json` (`tests/contract/test_compile_counters.py`); wall time is recorded
  with the load average and page faults, never gated.

### Removed
- `ModelStructure`, `ModelValues`, `Model.structure`, `Model.values`: use the fields of `Model`.
- `Frame`, `Model.frames`: use `Model.frame_parent_joint_ids` and `Model.frame_placements`.
- `Body`: use `Model.body_names` and `Model.body_inertias`.
- `Joint`, `FrameType`, `JOINT_DIMENSIONS`, `Trajectory`, `StateMultibody`: no replacement; no caller used
  them (joint objects carry `nq` and `nv`; keep clips in your own type).
- `SE3`, `lie.SE3`, `lie.SO3`, `lie.Pose` and their `spatial` re-exports: use the `lie.se3` and `lie.so3` functions
  on tensors.
- `spatial.Motion`, `spatial.Force`, `spatial.Inertia`, `spatial.Symmetric3`: use `(..., 6)`, `(..., 10)` and
  `(..., 3, 3)` tensors with the `spatial` functions (`transform_motion`, `cross_force`, `inertia_of_box`, ...).
- `forward_kinematics_state`, `Model.fk`: use `forward_kinematics`.
- `frame_jacobian_time_variation`, `joint_jacobian_time_variation`, `centroidal_map`, `centroidal_momentum` at the
  root: use `kinematics.frame_jacobian_time_variation`, `kinematics.joint_jacobian_time_variation`,
  `dynamics.centroidal_map` and `dynamics.centroidal_momentum`.
- `data_model`: use `model`.
- `Model.get_subtree`, `Model.get_support`, `Model.children`, `Model.subtrees`, `Model.supports`,
  `Model.topo_order`: use `kinematics.subtree` and `kinematics.support`; joints are stored parents first.
- `Model.expand_configuration`, `Model.has_mimic`, `Model.q_expansion`, `Model.q_offset`, `Model.v_expansion`: use
  `kinematics.full_configuration`.
- `Model.integrate`, `Model.difference`, `JointModel.integrate`, `JointModel.difference`: use
  `kinematics.integrate` and `kinematics.difference`.
- `Model.joint_name_to_id`, `Model.body_name_to_id`, `Model.frame_name_to_id`: use `Model.joint_id`,
  `Model.body_id` and `Model.frame_id`.
- `Model.body_inertia`, `Model.spatial_inertias`: use `Model.body_inertias` and `spatial.inertia_matrix`.
- `Model.meta`, `IRModel.meta`: use `io.load_geometry` and `io.build_geometry`.
- `Model.rotor_inertia`, `Model.friction`, `Model.reference_configurations`, `IRJoint.friction`: no replacement;
  they were always zero or never read.
- `JointRX`, `JointRY`, `JointRZ`, `JointRevoluteUnaligned`, `JointPX`, `JointPY`, `JointPZ`,
  `JointPrismaticUnaligned`: use `JointRevolute(axis)` and `JointPrismatic(axis)`.
- `JointUniverse`: use `JointFixed`.
- `JointComposite`, `JointKind`, `JointModel.kind`, `joint_bias_acceleration`, `joint_motion_subspace_derivative`:
  no replacement; nothing called them, and a joint's kind is its class.
- `joint_placement`, `joint_motion_subspace`, `joint_velocity`, `JointModel.random_configuration`: use
  `JointModel.placement`, `JointModel.motion_subspace` and `JointModel.random`.
- Kind strings (`"revolute_rx"`, `"prismatic_px"`, `"revolute_unbounded"`, `"composite"`, `"ball"`, `"free"`, ...):
  use the joint classes (`JointRevolute`, `JointPrismatic`, ...).
- `se3.adjoint_inv` and the `tangents.*_jacobian_inv_*` functions: use `se3.adjoint_inverse` and
  `tangents.*_jacobian_inverse_*`.
- `se3.apply_base`: use `se3.compose(base[..., None, :], poses)`.
- `so3.adjoint`: use `so3.to_matrix`.
- `exceptions.BackendNotAvailableError`: use `exceptions.OptionalDependencyError`.
- `exceptions.ModelInconsistencyError`, `io.IRError`: use `exceptions.ModelError`.
- `load(free_flyer=)`, `load(preserve_joint_order=)`, `build_model(preserve_joint_order=)`, `format="builder"`, a
  callable or `yourdfpy` source: use `root_joint=JointFreeFlyer()`, one joint order, and `io.build_model` on an IR.
- `io.register_parser` and the parser registry: use `build_model(my_parser(path))`.
- `io.AssetResolver`, `FilesystemResolver`, `PackageResolver`, `CompositeResolver`, `CachedDownloadResolver`: use
  `viewer.robot_meshes(geometry, root=...)`, which reads the mesh URIs under `root`.
- `io.IRGeom`, `IRBody.visual_geoms`, `IRBody.collision_geoms`: use `io.IRShape` in `IRModel.shapes`.
- `IRJoint.kind`, `IRJoint.axis`, `IRJoint.pitch`, `IRJoint.joint_model`: use `IRJoint.joint`.
- `IRFrame.frame_type`, `ModelBuilder.add_frame(frame_type=)`: no replacement; every frame has one type.
- `ModelBuilder.add_revolute_x/y/z`, `add_prismatic_x/y/z`: use `ModelBuilder.add_revolute` and
  `ModelBuilder.add_prismatic` with `axis=`.
- `ModelBuilder.add_collision_geom`: use `ModelBuilder.add_shape`.
- `io.builders` (`build_kinematic_tree_model`, `build_kinematic_tree_body`): use a loop over `ModelBuilder`.
- `optim.CacheRef`: use the name string `Problem.cached` returns.
- `Result.__getitem__`: use `Result.x`.
- `viewer.RobotView`, `viewer.robot_layers`: use `RigidMeshLayer`, `robot_skeleton`, `FramesLayer` and
  `SpheresLayer`.
- `viewer.skeleton_spec`, `viewer.spread_offsets`: use `viewer.robot_skeleton`.
- `Layer.times`, `Layer.root`: use `Viewer.add(..., times=)`.
- The module paths `kinematics.state`, `kinematics.jacobian`, `dynamics.state`, `io.parsers`, `lie.types`,
  `lie.alignment`, `spatial.motion`, `spatial.force`, `spatial.inertia`, `spatial.mesh` and their neighbours: use
  the package (`better_robot.kinematics`, ...).
- `Data` (`oMi`, `oMf`, `liMi`, `M`, `nle`, `J`, `dJ`, `Ag`, `hg`, `com`, ...) and `update_frame_placements`: use
  `forward_kinematics` (`KinematicState.joint_poses`, `frame_poses`) and the `kinematics` and `dynamics` functions
  (`mass_matrix`, `inverse_dynamics`, `joint_jacobian`, `dynamics.centroidal_map`, `center_of_mass`).
- `compute_joint_jacobians`, `get_joint_jacobian`, `get_frame_jacobian`, `compute_joint_jacobians_time_variation`,
  `get_*_jacobian_time_variation`, `kinematics.ReferenceFrame`: use `joint_jacobian`, `frame_jacobian` and the
  `kinematics.*_jacobian_time_variation` functions with `reference="world"`, `"local"` or `"local_world_aligned"`.
- `rnea`, `aba`, `crba`, `ccrba`, `CCRBAResult`, `compute_centroidal_map`, `compute_centroidal_momentum`, `nle`,
  `bias_forces`: use `inverse_dynamics`, `forward_dynamics`, `mass_matrix`, `dynamics.centroidal_map` and
  `dynamics.centroidal_momentum`; the bias forces are `inverse_dynamics` with zero acceleration.
- `compute_minverse`, `compute_coriolis_matrix`, the dynamics derivative helpers (`compute_rnea_derivatives`,
  `compute_aba_derivatives`, `compute_crba_derivatives`, centroidal derivatives): use `torch.linalg.inv` on
  `mass_matrix` and differentiate the dynamics functions with autograd.
- `integrate_q`, `semi_implicit_euler`, `symplectic_euler`, `rk4` and `dynamics.integrators`: use
  `kinematics.integrate`; physics integration belongs to the caller's simulation.
- `better_robot.dynamics.action`, `retarget`, `RetargetCostConfig`, `better_robot.tasks`: no replacement; build
  the task as an `optim.Problem` from `terms`.
- `solve_trajopt` and its parameterizations (`KnotTrajectory`, `BSplineTrajectory`,
  `TrajectoryParameterization`): use an `optim.Problem` with a time variable (`Problem(..., time=("q",))`) and
  `optim.solve`.
- `CostStack`, `CostItem`, `CostKind`, `LeastSquaresProblem`, `ResidualItem`, `ObjectiveItem`, `ObjectiveTerm`,
  `register_residual`, `get_residual`, `registered_residuals`, `better_robot.costs`: use `optim.Problem` with
  `Problem.residual` and `Problem.scalar` terms.
- The residual classes (`Residual`, `SmoothnessResidual`, `AccelerationResidual`, `JerkResidual`,
  `RestResidual`, `NullspaceResidual`, `YoshikawaResidual`, `SelfCollisionResidual`, `WorldCollisionResidual`,
  `JointAccelLimit`, `ContactConsistencyResidual`, `PointProjectionResidual`, the scene-SDF and masked-Chamfer
  residuals, `ScalarCost`): use plain functions as `optim` terms and the `terms` factories (`rest`,
  `finite_difference`, `self_clearance`, `velocity_limit`, ...).
- Residual weights and gates (`weight` objects `ScaleWeight`, `DiagonalWeight`, `Weight`, `row_weight`,
  `residual.enabled`, reductions, activity masks): use the term `weight=` and `Problem.with_(weights=...)`.
- `ResidualState`, `residual_jacobian`, `ResidualSpec`, `jacobian_spec`, `JacobianStrategy`, provider `inputs` and
  `reads` declarations: use `TermFn(fn, reads, *, jacobian=None)`; without `jacobian` the term is differentiated
  by autograd.
- `Variable`, `SO3Variable`, `SE3Variable`, `RobotVariable`, `VarSpec`, `Values`, `detach_values`, `Bounds`,
  `SO3Manifold`, `SE3Manifold`, `RobotConfig`, `Node`, `RobotState`, `EvaluationContext`, `Provider`,
  `RobotStateProvider`: use named tensors in `optim.Problem` with `manifolds=` (`terms.RobotManifold`) and
  `bounds=`, and `Problem.cached` for shared work.
- `Optimizer`, `OptimizationResult`, `OptimizerInfo`, `OptimizerConfig`, `SolverState`, `SolverStatus`, `LMState`,
  `LMStatus`, `FirstOrderResult`, `run_first_order`, `TorchOptimizer`, `Adam`, `AdamState`, `AdamStatus`,
  `LMThenLBFGS`, `MultiStageOptimizer`, `OptimizerStage`, `PhaseResult`, `run_phases`, `resume`: use
  `optim.solve` and `optim.run` with `Phase`s, which return a `Result`.
- `DampingStrategy`, `Constant`, `Adaptive`, `LevenbergMarquardt(block_step_limits=)`: use
  `LevenbergMarquardt(damping=...)`.
- `Cholesky`, `BandedCholesky`, `LSTSQ`, `LU`, `NormalCG`, `NormalOperator`, `BlockBandedMatrix`,
  `TemporalAnalysis`, `LinearizationReason`, `LinearSystemKind`, `ImplicitDiffConfig`, `linearization=`: use
  `Normal`, `BandedNormal` and `ArrowNormal`, chosen by `Problem(time=..., dense=...)`, and `optim.solve_layer`
  for implicit gradients.
- The import paths `optim.blocks`, `optim.kernels.*`, `optim.solvers.*`, `optim.structure`, `optim.providers`,
  `optim.autograd`, `optim.manifolds`, `optim.optimizers`, `optim.strategies`: use `better_robot.optim`.
- `IR_SCHEMA_VERSION`, `IRModel.schema_version`, `IRSchemaVersionError`: no replacement; the IR is an in-process
  value, so parse the source file again after an upgrade.
- `kinematics.chain`, `data_model.indexing`, `better_robot.utils`, `ExecutionBatch`, `flatten_execution_batch`,
  `ModelValues.execution_batch_shape`: no replacement; they had no caller.
- The Warp lanes (`use_warp`, the `warp` extra, the kernel modules, `kinematics._warp_bridge`) and
  `forward_kinematics(use_compile=)`: use `torch.compile` on the library functions; torch is the one
  implementation.
- `Visualizer`, `SkeletonMode`, `URDFMeshMode`, `TrajectoryPlayer`: use `Viewer` with `robot_skeleton`,
  `RigidMeshLayer` and its `Timeline`.
- The development dependencies `hypothesis`, `pyperf`, `mypy`, `jaxtyping`, `pytest-cov`, `pytest-xdist`,
  `matplotlib` and the GitHub Actions workflow: no replacement; nothing used them, and the test tiers run by local
  command and hooks.
- The `bench` pytest marker, the `tests/bench/` micro-benchmarks and `pytest-benchmark`, which ran them: use
  `benchmarks/run.py`, whose counter rows the full tier gates.

### Fixed
- The MJCF parser places capsule and cylinder `fromto` geometry as MuJoCo's compiler does.
- URDF inertias honour `<inertial><origin rpy>`.
- Zero-size batches run through forward kinematics, the dynamics functions and every solver.
- `Viewer.url` reports the bound port, not the requested one.
- `so3.to_euler` is exact at pitch ±π/2.
- `robot_skeleton` draws the markers of axis-aligned revolute and prismatic joints along their axes.
- `LBFGS` with bounds reaches the optimum of bound-heavy problems (a nonnegative force fit stopped 0.4-4 % above
  it, now within 0.01 %), and a problem whose plain gradient step finds no decrease is done.
- `robot_skeleton` draws a joint kind defined outside the library as a sphere instead of raising `KeyError`.

## v0.2.0 (2026-04-11)

The first stable release of the PyTorch-native BetterRobot stack. Its named stubs (`compute_minverse`,
`compute_coriolis_matrix`, the higher-order integrators, analytic derivatives, `JerkResidual`, `YoshikawaResidual`,
`NullspaceResidual`, `SelfCollisionResidual`, `WorldCollisionResidual`, `JointAccelLimit`, `retarget`, the Warp
kernels) raised on call.

### Added
- **Pinocchio-style `Model` / `Data` architecture.** Frozen `Model`,
  mutable `Data`, polymorphic `.to()`.
- **Universal joint system.** Revolute (R{X,Y,Z}, unaligned, unbounded),
  prismatic, spherical, free-flyer, fixed, helical, planar, mimic,
  composite. Single code path for fixed and floating base — a
  floating-base robot is one whose root joint is `JointFreeFlyer`.
- **Batched-by-default FK.** `forward_kinematics(model, q: (B..., nq))`
  runs over any batch shape with no Python loops.
- **Analytic Jacobians.** All built-in residuals have a `.jacobian()`
  method; `JacobianStrategy.AUTO` prefers analytic, falls back to
  central finite differences.
- **Unified solver stack.** `LeastSquaresProblem` + `CostStack` + `Optimizer`
  serves IK and trajectory optimisation through the same substrate.
  Pluggable optimisers (LM / GN / Adam / L-BFGS / multi-stage), linear
  selectable solvers (Cholesky / LSTSQ), robust kernels
  (L2 / Huber / Cauchy / Tukey), and damping strategies (Constant /
  Adaptive).
- **Featherstone dynamics.** RNEA, ABA, CRBA, CCRBA, centroidal momentum,
  centre of mass, and autograd-derived RNEA, ABA, and CRBA derivative helpers. Three-layer
  Crocoddyl-style action models for future optimal-control work.
- **Trajectory optimisation.** `solve_trajopt` with knot parameterisation;
  manifold-aware `Trajectory.resample`.
- **URDF + MJCF parsers.** `br.load(path)` dispatches by suffix;
  `free_flyer=True` adds a free-flyer root. Programmatic `ModelBuilder`
  for robots not described by a file.
- **Asset resolution.** `AssetResolver` Protocol with `Filesystem`,
  `Package`, `Composite`, and `CachedDownload` resolvers; mesh path
  logic lives in one place.
- **Viewer.** `Visualizer` with viser backend, `SkeletonMode`,
  `URDFMeshMode`, draggable IK target gizmos, frame-axes / grid /
  force-vector overlays, minimal `TrajectoryPlayer`.
- **Public API.** 26 symbols at `better_robot.__init__`: `Model`, `Data`, `Frame`, `Joint`, `Body`, `load`,
  `ModelBuilder`, `SE3`, `forward_kinematics`, `update_frame_placements`, `compute_joint_jacobians`,
  `get_joint_jacobian`, `get_frame_jacobian`, `JacobianStrategy`, `rnea`, `aba`, `crba`, `center_of_mass`,
  `compute_centroidal_map`, `register_residual`, `CostStack`, `LeastSquaresProblem`, `solve_ik`, `solve_trajopt`,
  `retarget`, `Trajectory`.
