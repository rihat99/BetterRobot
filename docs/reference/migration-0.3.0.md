# Migrating to 0.3.0

Replace each name in the left column by the one in the middle column. Version 0.3.0 keeps one name for each
idea: one flat `Model` record, plain tensors and functions instead of the Lie and spatial classes, one builder, one
loader taking a path, full words in names. Where the middle column is `—`, nothing replaces the name and the note
says why and what to call instead.

Names are written relative to `better_robot`: `lie.SE3` is `better_robot.lie.SE3`. A name without a module
(`SE3`, `Model.fk`) is gone from every module that exported it. `tests/contract/test_changelog.py` checks that every
public name of the 0.2 line that 0.3.0 lacks, or whose signature changed in a way that can break a caller, is on
this page, and that every name in the middle column exists. The
[changelog](changelog.md) lists what 0.3.0 adds, and the names removed between the 0.2.0 release and the last state
of the 0.2 line.

## Root `better_robot`

The root exports the twelve packages `exceptions`, `lie`, `spatial`, `model`, `kinematics`, `dynamics`,
`collision`, `io`, `optim`, `terms`, `viewer` and `testing`, and the names `load`, `Model`, `ModelBuilder`,
`JointFreeFlyer`, `forward_kinematics`, `KinematicState`, `frame_jacobian`, `joint_jacobian`, `inverse_dynamics`,
`mass_matrix`, `forward_dynamics`, `center_of_mass`, `solve_ik` and `Viewer`.

| 0.2 | 0.3.0 | note |
|---|---|---|
| `ModelStructure`, `ModelValues` | `Model` | every field lives on `Model`: `model.values.x` and `model.structure.x` are `model.x` |
| `Frame` | `Model.frame_parent_joint_ids`, `Model.frame_placements` | frame `i` is entry `i` of each field |
| `Body` | `Model.body_names`, `Model.body_inertias` | nothing constructed it |
| `Joint` | — | an enum nothing read; the joint objects are `Model.joints` |
| `FrameType` | — | every frame has one type |
| `JOINT_DIMENSIONS` | `JointModel` | each joint object carries its `nq` and `nv` |
| `Trajectory` | — | no reader in the library; keep clips in your own type |
| `SE3` | `lie.se3` | functions on `(..., 7)` pose tensors `[tx, ty, tz, qx, qy, qz, qw]` |
| `forward_kinematics_state` | `forward_kinematics` | same arguments |
| `frame_jacobian_time_variation`, `joint_jacobian_time_variation` | `kinematics.frame_jacobian_time_variation`, `kinematics.joint_jacobian_time_variation` | they leave the root |
| `centroidal_map`, `centroidal_momentum` | `dynamics.centroidal_map`, `dynamics.centroidal_momentum` | they leave the root |
| `data_model` | `model` | the package of `Model`, the joint classes and the geometry records |

## `Model`

`Model` is one frozen record: static fields hold the topology, tensor fields the differentiable parameters.

| 0.2 | 0.3.0 | note |
|---|---|---|
| `Model.structure`, `Model.values` | `Model` | the fields are flat on `Model` |
| `Model.fk` | `forward_kinematics` | `forward_kinematics(model, q)` |
| `Model.meta` | `io.load_geometry` | the shapes are a `Geometry` value beside the model: `io.load_geometry(path, model)` |
| `Model.lower_pos_limit`, `Model.upper_pos_limit` | `Model.lower_position_limit`, `Model.upper_position_limit` | full words |
| `Model.joint_models` | `Model.joints` | the joint objects |
| `Model.parents` | `Model.parent_joint_ids` | an integer index field ends in `_ids` |
| `Model.mimic_source` | `Model.mimic_source_ids` | |
| `Model.mimic_multiplier`, `Model.mimic_offset` | `Model.mimic_multipliers`, `Model.mimic_offsets` | per-joint fields are plural |
| `Model.frames` | `Model.frame_parent_joint_ids`, `Model.frame_placements` | frame `i` is entry `i` of each |
| `Model.joint_name_to_id`, `Model.body_name_to_id`, `Model.frame_name_to_id` | `Model.joint_id`, `Model.body_id`, `Model.frame_id` | each takes a name or an index |
| `Model.get_subtree`, `Model.get_support` | `kinematics.subtree`, `kinematics.support` | `subtree(model, joint)`, `support(model, joint)` |
| `Model.children`, `Model.subtrees`, `Model.supports`, `Model.topo_order` | `kinematics.subtree`, `kinematics.support` | joints are stored parents first, so `range(model.njoints)` is a topological order |
| `Model.expand_configuration` | `kinematics.full_configuration` | `full_configuration(model, q)` fills in the mimic joints |
| `Model.has_mimic`, `Model.q_expansion`, `Model.q_offset`, `Model.v_expansion` | `kinematics.full_configuration` | the mimic expansion is a private plan of the model |
| `Model.integrate`, `Model.difference` | `kinematics.integrate`, `kinematics.difference` | `integrate(model, q, v)`, `difference(model, q0, q1)` |
| `Model.body_inertia`, `Model.spatial_inertias` | `Model.body_inertias`, `spatial.inertia_matrix` | `model.body_inertias[..., i, :]` is body `i`'s packed `(10,)` inertia; `inertia_matrix(model.body_inertias)` gives the 6x6 matrices |
| `Model.rotor_inertia`, `Model.friction`, `Model.reference_configurations` | — | always zero or never read; motor inertia is `Model.armature`, damping `Model.damping` |
| `Model.random_configuration(generator=None)` | `Model.random_configuration(*, generator)` | the generator is required |
| `Model.with_values(joint_placements, body_inertias, frame_placements, armature)` | `Model.with_values(**fields)` | every tensor field of `Model`, by keyword, each checked for shape |
| `Model.__eq__`, `Model.__hash__` by value | `Model.structure_key` | models compare by identity; compare structures by `structure_key` |

## Joints

A joint's kind is its class. The classes are exported from `better_robot.model`.

| 0.2 | 0.3.0 | note |
|---|---|---|
| `JointModel` (a `Protocol`) | `JointModel` | an abstract base: `nq`, `nv`, `manifold`, `placement`, `motion_subspace`, `neutral`, `random` |
| `JointRX`, `JointRY`, `JointRZ`, `JointRevoluteUnaligned` | `JointRevolute` | `JointRevolute(axis=(1.0, 0.0, 0.0))`; the axis is a unit 3-vector |
| `JointPX`, `JointPY`, `JointPZ`, `JointPrismaticUnaligned` | `JointPrismatic` | `JointPrismatic(axis=...)` |
| `JointUniverse` | `JointFixed` | joint 0 is a `JointFixed` named `"universe"` |
| `JointComposite`, `JointKind` | — | nothing called them |
| `JointModel.kind` | `JointModel` | test the class: `isinstance(joint, JointRevolute)` |
| `joint_placement`, `joint_motion_subspace`, `JointModel.random_configuration` | `JointModel.placement`, `JointModel.motion_subspace`, `JointModel.random` | `random(generator, *, lower, upper)` |
| `joint_velocity` | `JointModel.motion_subspace` | the joint velocity is the motion subspace times `v` |
| `JointModel.integrate`, `JointModel.difference` | `kinematics.integrate`, `kinematics.difference` | written once per manifold, for the whole configuration |
| `joint_bias_acceleration`, `joint_motion_subspace_derivative` | — | hooks nothing called |
| kind strings `"revolute_rx"`, `"revolute_ry"`, `"revolute_rz"`, `"revolute_unaligned"`, `"prismatic_px"`, `"prismatic_py"`, `"prismatic_pz"`, `"prismatic_unaligned"`, `"revolute_unbounded"`, `"universe"`, `"composite"`, `"ball"`, `"free"`, `"floating"`, `"world"` | `JointRevolute`, `JointPrismatic`, `JointRevoluteUnbounded`, `JointFixed`, `JointSpherical`, `JointFreeFlyer` | the IR and the builder take joint objects |

## `lie`

| 0.2 | 0.3.0 | note |
|---|---|---|
| `lie.SE3`, `lie.SO3`, `lie.Pose`, `spatial.SE3`, `spatial.SO3`, `spatial.Pose` | `lie.se3`, `lie.so3` | functions on `(..., 7)` poses and `(..., 4)` quaternions `[qx, qy, qz, qw]` |
| `se3.adjoint_inv` | `se3.adjoint_inverse` | |
| `se3.from_translation(axis, disp)` | `se3.from_translation(translation)` | pass `axis * disp` |
| `se3.apply_base` | `se3.compose` | `se3.compose(base[..., None, :], poses)` broadcasts the base over the poses |
| `so3.adjoint` | `so3.to_matrix` | the same function under its one name |
| `tangents.right_jacobian_inv_so3`, `tangents.left_jacobian_inv_so3`, `tangents.right_jacobian_inv_se3`, `tangents.left_jacobian_inv_se3` | `tangents.right_jacobian_inverse_so3`, `tangents.left_jacobian_inverse_so3`, `tangents.right_jacobian_inverse_se3`, `tangents.left_jacobian_inverse_se3` | full words |

## `spatial`

Motions and forces are `(..., 6)` tensors, linear part first. An inertia is a packed `(..., 10)` tensor
`[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`: the centre of mass and the rotational inertia about it.

| 0.2 | 0.3.0 | note |
|---|---|---|
| `spatial.Motion`, `spatial.Force` | `(..., 6)` tensors | `.linear` is `x[..., :3]`, `.angular` is `x[..., 3:]`, `compose` and `zero` are `+` and `torch.zeros` |
| `spatial.Inertia` | `(..., 10)` tensors | `.mass` is `inertia[..., 0]`, `.com` is `inertia[..., 1:4]`, `add` and `zero` are `+` and `torch.zeros` |
| `spatial.Symmetric3` | `(..., 3, 3)` tensors | `from_matrix` and `to_matrix` are the identity |
| `Motion.cross_motion` | `spatial.cross_motion` | |
| `Motion.cross_force`, `Force.cross_motion` | `spatial.cross_force` | |
| `Motion.se3_action`, `Force.se3_action`, `Inertia.se3_action` | `spatial.transform_motion`, `spatial.transform_force`, `spatial.transform_inertia` | the pose comes first: `transform_motion(pose, twist)` |
| `Inertia.inertia_matrix` | `spatial.rotational_inertia` | the 3x3 about the centre of mass; the 6x6 is `spatial.inertia_matrix` |
| `Inertia.apply` | `spatial.inertia_matrix` | `inertia_matrix(inertia) @ twist[..., None]` |
| `Inertia.from_sphere`, `Inertia.from_box`, `Inertia.from_capsule`, `Inertia.from_ellipsoid` | `spatial.inertia_of_sphere`, `spatial.inertia_of_box`, `spatial.inertia_of_capsule`, `spatial.inertia_of_ellipsoid` | they return tensors; `inertia_of_ellipsoid(mass, *, semi_axes)` |
| `Inertia.from_mesh` | `spatial.inertia_of_mesh` | `from_mesh(...).data` is `inertia_of_mesh(...)`; it scans the mesh only with `check_mesh=True` |
| `Inertia.from_mass_com_sym3`, `Inertia.from_mass_com_matrix` | `spatial.inertia_from_mass_com` | `inertia_from_mass_com(mass, com, *, rotational_inertia)` with a `(..., 3, 3)` matrix |

## `kinematics`

| 0.2 | 0.3.0 | note |
|---|---|---|
| `KinematicState.joint_pose`, `KinematicState.frame_pose` | `KinematicState.joint_poses`, `KinematicState.frame_poses` | plural: one entry per joint or frame |
| `KinematicState.joint_rotation`, `KinematicState.joint_translation` | `KinematicState.joint_rotations`, `KinematicState.joint_translations` | |
| `KinematicState.frame_rotation`, `KinematicState.frame_translation` | `KinematicState.frame_rotations`, `KinematicState.frame_translations` | |
| `KinematicState.structure` | `KinematicState.model` | |
| `KinematicState.joint_transform`, `KinematicState.frame_transform` | `KinematicState.joint_transforms`, `KinematicState.frame_transforms` | |
| `KinematicState(structure, q, joint_transform, frame_transform)` | `KinematicState(*, model, q, joint_transforms)` | `forward_kinematics` builds states; the constructor is keyword-only; `frame_transforms` is computed from the joints on first use |
| `KinematicState.frame_transform`, `KinematicState.frame_pose` read under another grad mode than the one of `forward_kinematics` | `KinematicState.frame_transforms`, `KinematicState.frame_poses` | computed on first read under the grad mode of that read: under `torch.no_grad` or `torch.inference_mode` no graph; with grad enabled on a state made under `torch.no_grad`, a graph to the model tensors that require grad |
| `forward_kinematics_state(model, q, *, check_quaternion_norm)` | `forward_kinematics(model, q, *, check_quaternion_norm)` | the flag checks every quaternion joint |

## `dynamics` and `collision`

| 0.2 | 0.3.0 | note |
|---|---|---|
| `inverse_dynamics(model, state, *, v, a, fext=None)` | `inverse_dynamics(model, state, *, v, a, external_forces=None)` | full word |
| `forward_dynamics(model, state, *, v, tau, fext=None)` | `forward_dynamics(model, state, *, v, tau, external_forces=None)` | |
| `forward_dynamics` raising torch's `_LinAlgError` for a mass matrix that is not positive definite | `forward_dynamics` | NaN accelerations for that batch element only; test `ddq.isnan()` where the error was caught |
| `StateMultibody` | — | nothing called it |
| `RobotSpheres(frame_ids, offsets, radii)`, `RobotCapsules(frame_ids, starts, ends, radii)` | `collision.RobotSpheres`, `collision.RobotCapsules` | the fields are keyword-only: `frame_ids=`, `offsets=` |

## `exceptions`

| 0.2 | 0.3.0 | note |
|---|---|---|
| `exceptions.BackendNotAvailableError` | `exceptions.OptionalDependencyError` | also an `ImportError`; its message carries the install line |
| `exceptions.ModelInconsistencyError`, `io.IRError` | `exceptions.ModelError` | also a `ValueError` |
| `UnknownNameError(message)` | `exceptions.UnknownNameError(kind, name, *, valid)` | it writes the message from the kind, the name and the valid names; catching it is unchanged |

## `io` and `ModelBuilder`

`io` exports `load`, `parse`, `parse_urdf`, `parse_mjcf`, `build_model`, `build_geometry`, `load_geometry`,
`ModelBuilder`, `IRModel`, `IRBody`, `IRJoint`, `IRFrame` and `IRShape`.

| 0.2 | 0.3.0 | note |
|---|---|---|
| `load(source, *, format="auto", root_joint, free_flyer=False, preserve_joint_order=False, device, dtype)` | `load(path, *, format=None, root_joint=None, device=None, dtype=torch.float32)` | a path only; a builder's IR goes through `io.build_model` |
| `free_flyer=True` | `root_joint=JointFreeFlyer()` | one root parameter |
| `preserve_joint_order=` of `load` and `io.build_model` | — | one joint order: parents first, the file's order kept wherever it already is |
| `format="builder"` | `io.build_model` | `build_model(builder.finalize())` |
| `parse(source, *, format="auto")` | `parse(path, *, format=None)` | |
| `io.register_parser` | `io.build_model` | a new format is a function returning `IRModel`: `build_model(my_parser(path))` |
| `io.AssetResolver`, `io.FilesystemResolver`, `io.PackageResolver`, `io.CompositeResolver`, `io.CachedDownloadResolver` | `viewer.RobotVisual` | mesh URIs stay strings; `RobotVisual(model, geometry=, root=)` reads them under `root`, the description file |
| `io.IRGeom`, `IRBody.visual_geoms`, `IRBody.collision_geoms` | `io.IRShape`, `IRModel.shapes` | `IRShape(body, shape, collision)` with a typed `Shape` |
| `IRModel.meta` | `io.build_geometry` | the geometry is built from the IR: `build_geometry(ir, model)` |
| `IRJoint.kind`, `IRJoint.axis`, `IRJoint.pitch`, `IRJoint.joint_model` | `IRJoint.joint` | the joint object, such as `JointRevolute(axis=...)` |
| `IRJoint.friction` | — | `Model` has no friction to fill |
| `IRFrame.frame_type` | — | every frame has one type |
| `IRModel(name, bodies, joints, ...)`, `IRBody(name, ...)`, `IRJoint(name, parent_body, child_body, ...)`, `IRFrame(name, parent_body, placement)` | `name=`, `bodies=`, `joints=`, `parent_body=`, `child_body=`, `joint=`, `placement=` | keyword-only fields; the fields that left are in the rows above |
| `ModelBuilder.add_revolute_x`, `ModelBuilder.add_revolute_y`, `ModelBuilder.add_revolute_z` | `ModelBuilder.add_revolute` | `add_revolute(name, *, parent, child, axis=(1.0, 0.0, 0.0))` |
| `ModelBuilder.add_prismatic_x`, `ModelBuilder.add_prismatic_y`, `ModelBuilder.add_prismatic_z` | `ModelBuilder.add_prismatic` | `axis=` as above |
| `ModelBuilder.add_frame(name, *, parent_body, placement, frame_type)` | `ModelBuilder.add_frame(name, *, parent, placement)` | |
| `ModelBuilder.add_collision_geom(body, kind, *, params, origin)` | `ModelBuilder.add_shape(body, shape, *, collision=True)` | `shape` is a `model.Sphere`, `model.Box`, `model.Cylinder`, `model.Capsule` or `model.Mesh` |
| `ModelBuilder.add_joint(name, *, kind="...")` | `ModelBuilder.add_joint(name, *, kind=JointRevolute(...))` | a joint object, not a string |
| `build_kinematic_tree_model`, `build_kinematic_tree_body` | `ModelBuilder` | write the loop with the builder |
| frame names `"body_<link>"` | frame names `"<link>"` | every body's frame carries the body's name |

## `optim`

| 0.2 | 0.3.0 | note |
|---|---|---|
| `optim.CacheRef`, `Problem.cached(...)` returning one | `Problem.cached` | `problem.cached(...)` returns the name string a term reads |
| `TermFn(fn, reads, jac, window)`, `TermFn.jac` | `TermFn(fn, reads, *, jacobian, window)`, `TermFn.jacobian` | |
| `Problem.residual(..., jac=)` | `Problem.residual(..., jacobian=)` | |
| `Problem.with_(*, weights, free)` | `Problem.with_(*, data, weights, free)` | a problem's data is read-only; `with_(data=...)` replaces entries |
| `Phase(solver, iters)`, `Phase.iters`, `Result.iters` | `Phase(solver, iterations)`, `Phase.iterations`, `Result.iterations` | |
| `Phase(on_start=...)` returning a dict or `None` | `Phase.on_start` | returns `(x, data)` |
| `Result.__getitem__` | `Result.x` | `result.x["q"]` |
| `solve(..., iters=)`, `solve_layer(..., iters=)`, `solve_constrained(..., iters=)` | `iterations=` | |
| `LevenbergMarquardt(gtol, ftol, xtol)`, `GaussNewton(gtol, xtol)`, `LBFGS(gtol)` | `gradient_tolerance=`, `loss_tolerance=`, `step_tolerance=` | each solver keeps the subset it had |
| `TorchOptim(optimizer_cls, *, scheduler, **kwargs)` | `TorchOptim(factory, *, scheduler=None)` | `factory(params)` returns a `torch.optim.Optimizer` |
| `State.it` | `State.iteration` | a step passed as `Phase(step=)` or `solve(..., step=)`, and every step inside `solve_ik`, reads 0 here; `run` keeps the count |
| `State(x, loss, done, it=0, extra)` | `State(x, loss, *, done, solver, iteration=0, extra)` | `done` is keyword-only; `solver` is the class of the solver that made the state, and a warm start needs a solver of that class |
| `Result(x, loss, done, iters, ...)` | `Result(x, loss, *, done, iterations, ...)` | `done` and the fields after it are keyword-only |
| `Solver`, `Manifold` as `Protocol`s | `optim.Solver`, `optim.Manifold` | abstract bases: subclass them |
| `Normal(H, g, lower, upper)`, `BandedNormal(diagonal, upper, g, layout, ...)`, `ArrowNormal(band, border, corner, g, ...)` | `g=`, `layout=`, `corner=` | keyword-only after the matrix blocks |

## `terms`

| 0.2 | 0.3.0 | note |
|---|---|---|
| `solve_ik(..., iters=100)` | `solve_ik(..., iterations=100)` | also exported from the root; every step runs (no early exit), and the default is now 50 |
| `solve_ik(..., restarts=k, generator=g)` | `solve_ik(..., seeds=k + 1, generator=g)` | the starts run as one batch for `iterations` steps and the lowest per target is kept; `seeds=32, iterations=16` is the benchmark's multi-start call |
| `random_start(model, q0, *, generator=None)` | `random_start(model, q0, *, generator)` | the generator is required |
| `frame_pose(..., state=CacheRef(...))`, `frame_position(..., state=CacheRef(...))`, `self_clearance(..., state=CacheRef(...))`, `sphere_clearance(..., state=CacheRef(...))` | `state=` | the name string of the cached state |
| `sphere_clearance(..., hinge=True)`, `self_clearance(..., hinge=True)`: residual `max(margin - distance, 0)` | `sphere_clearance`, `self_clearance` | the hinge is C1, `gap² / (2 margin)` inside the margin band: raise the weight to keep the old push near the margin |
| `sphere_clearance(spheres, distance_fn, ...)` with a `distance_fn` whose entry `s` reads other centres, or that `torch.func` cannot transform | `optim.TermFn(fn=term.fn, reads=term.reads)` of that term | the analytic Jacobian needs both (record [0051](../contracts/decisions/0051-distance-functions-for-the-analytic-jacobian.md)) |

## `viewer`

| 0.2 | 0.3.0 | note |
|---|---|---|
| `viewer.RobotView`, `viewer.robot_layers` | `viewer.RobotVisual` | one layer: `RobotVisual(model, q, geometry=, root=, spheres=)` with `style=`, `collision=` and `frames=`; `update(q)` shows new configurations |
| `viewer.skeleton_spec`, `viewer.spread_offsets` | `viewer.SkeletonRecipe` | the skeleton is laid out inside `RobotVisual`; a `SkeletonRecipe` (`RobotVisual(maximal=, minimal=)`) sets how |
| `viewer.robot_skeleton(model, joint_poses, ...)` | `viewer.RobotVisual` | `RobotVisual(model, q, style="maximal")` or `"minimal"`; it runs the forward kinematics itself. The torque colours (`tau=`) and `end_points=` have no replacement |
| `robot_meshes(model, *, collision, default_color)` | `viewer.RobotVisual`, `viewer.load_meshes` | `RobotVisual(model, q, geometry=io.load_geometry(path, model), root=path)` draws the display shapes; `collision=True` overlays the collision shapes; `load_meshes(geometry, root=path, collision=)` returns them as `MeshSpec` arrays |
| `Layer.times`, `Layer.root` | `Viewer.add(..., times=)` | the viewer keeps what each `add` set; `Layer.elements` is read-only |
| `Viewer.add(layer, group)`, `Viewer.add_plot(..., group=)` | `Viewer.add(layers, *, tags)`, `Viewer.add_plot(..., tags=)` | one layer or an iterable of layers, returned as given; a layer's first tag is its scene parent `/<tag>/<name>`; `offset=` is `pose=` |
| `Viewer(*, host, port, fps, theme=None, server)` | `Viewer(*, host, port, fps, theme=LIGHT, ground=True, server)` | the viewer adds a `GroundLayer` under the tag `ground`; pass `ground=False` to add your own; a `server` passed in comes without `host` and `port` (they were ignored) |
| `Viewer.remove_group(group)`, `Viewer.set_offset(group, offset)`, `Viewer.set_visible(layer, visible)` | `Viewer.remove(target)`, `Viewer.set_pose(tag, pose)`, `Viewer.set_visible(target, visible)` | `remove` and `set_visible` take a layer or a tag; `set_pose` takes the offset `(3,)` or a pose `(7,)` `[x, y, z, qx, qy, qz, qw]`, which also turns the tag |
| `Viewer.play(block=True)` | `Viewer.serve()` | plays and serves the page until interrupted; `viewer.timeline.play()` plays without blocking |
| `Viewer.rebuild(layer)` | `Layer.redraw`, `Viewer.replace` | `layer.redraw()` pushes the current step after an in-place edit; `replace(old, new)` swaps in a new layer |
| `Viewer.aim(*, steps, azimuth, elevation)` | `Viewer.aim(*, steps, azimuth, elevation, fov)` | frames the boxes of the layers' `Layer.bounds` over at most 32 samples of each layer (32 of `steps` when given); until it is called the viewer aims after each `add` |
| `world_points` of `Layer` and every layer | `Layer.bounds` | the world box `(lo, hi)` of a step, which `Viewer.aim` and `Viewer.record` frame |
| `Layer(name, nsteps)`, `Layer.listed`, `GroundLayer.listed` | `Layer(name, nsteps, *, tags=)` | the sidebar's Layers section has one checkbox per tag once there are two; the ground is never listed |
| `Layer.build_gui(gui)`, `PlotPanel.build_gui`, `ImagePanel.build_gui` | `Layer.build_gui(sidebar)`, `Sidebar` | a layer adds its controls through the viewer's `Sidebar`; viser's GUI API is `sidebar.gui` |
| `GroundLayer(..., cell_color=, section_color=)` | `GroundLayer(..., cell_color=None, section_color=None)` | the grid takes `Theme.grid` unless given |
| `GroundLayer(size=10.0)` | `GroundLayer(size=None)` | the default grid has no edges and fades out; pass `size=` for a square |
| `Timeline.build_gui(gui)` | `Timeline.build_gui(sidebar)` | the viewer builds the Timeline section through its `Sidebar` |
| `Theme.brand_color` | `Theme.accent` | the colour of the GUI and the tools |
| `Theme.grid_cell`, `Theme.grid_section` | `Theme.grid` | a (cell, section) pair |
| `Theme.robot`, `Theme.human` | `Theme.hue` | a robot or body takes `theme.hue(index)`, its parts `theme.shade(hue, role)` |
| `Theme.joint`, `Theme.link`, `Theme.torque_ramp`, `Theme.over_limit` | `Theme.shade` | a skeleton takes its robot's hue and `theme.shade(hue, role)` for joints, drum faces, bodies and sticks; the torque colours went with `robot_skeleton(tau=)` |
| `Theme.point`, `Theme.force`, `Theme.torque`, `Theme.contact`, `Theme.target` | — | no layer read the colours, and an application passes its own |
| `Theme.control_width` | `Theme.sidebar_width` | viser's `"small"`, `"medium"` (the default) or `"large"`; `Viewer.set_theme` keeps it |
| `viewer.theme.colormap(values, *stops)` | — | nothing called it once the torque colours went; an application maps values to colours with numpy |
| `from better_robot.viewer.layers import ArrowsLayer, CameraLayer, FramesLayer, GroundLayer, PointsLayer, SkeletonLayer, SkinnedMeshLayer` | `from better_robot.viewer import ...` | `viewer.layers` re-exported seven layers, a second path to them; `better_robot.viewer` holds every layer (`SkeletonLayer` is `RobotVisual`, below) |
| `ArrowsLayer.vectors` (the given vectors times `scale`) | `ArrowsLayer(name, origins, *, vectors, scale)` | the attribute `vectors` holds the vectors as given and `scale` applies at each push: edit either and `redraw()` |
| `ArrowsLayer(..., head_size=, line_width=, shaft_radius=None, head_radius=None)`, `ArrowsLayer.HEAD_FRACTION` | `ArrowsLayer(..., shaft_radius=0.006, head_radius=0.016, head_length=0.04)` | viser's solid arrows: no line shafts, the head keeps its size, so an arrow shorter than `head_length` is all head |
| `CameraLayer(name, poses, *, fov, aspect, ...)` | `CameraLayer(name, poses, *, intrinsics, size, ...)` | the frustum from the `(3, 3)` matrix `K` of a `(W, H)` image, principal point included; `color=None` takes `Theme.text` |
| `viewer.layers.skinned.CameraLayer` | `viewer.CameraLayer` | the same class; `viewer.layers.skinned` re-exported it, a second path to it |
| `Timeline(nsteps=1, fps=30.0, loop=True)` | `Timeline(nsteps, *, fps=30.0, loop=True)` | |
| `SkeletonLayer(name, joints, parents, ...)` | `viewer.RobotVisual` | a model's skeleton is `RobotVisual(model, q, style="minimal")` or `"maximal"`; joint positions without a model are a `SpheresLayer` or a `PointsLayer` |
| `ArrowsLayer(name, origins, vectors, ...)`, `MeshLayer(name, vertices, faces, ...)`, `SpheresLayer(name, centers, radii, ...)` | `vectors=`, `faces=`, `radii=` | keyword-only after the second argument |
| `RigidMeshLayer(name, meshes, body_poses, ...)`, `SkinnedMeshLayer(name, vertices, faces, weights, bind_poses, bone_poses, ...)` | `body_poses=`, `faces=`, `weights=`, `bind_poses=`, `bone_poses=` | keyword-only |
| `MeshSpec(vertices, faces, body, origin, color, trimesh)`, `Theme(dark, brand_color, ...)` | `body=`, `origin=`, `dark=`, `accent=` | keyword-only fields |
| `PoseGizmo(viewer, name, pose, **options)` (in `viewer.tools`) | `pose=`, `on_change=`, `scale=`, `translation=`, `on_top=` | named options only; `pose` is `(7,)`; `translation=False` replaces `disable_sliders=True, disable_axes=True`, `on_top=True` replaces `depth_test=False` |
| `CameraFrustum(position, wxyz, fov, aspect)` (in `viewer.tools`) | `CameraFrustum(pose, *, fov, aspect)` | `pose (7,)` is `[x, y, z, qx, qy, qz, qw]`, quaternion scalar last; `CameraFrustum.from_camera` reorders viser's `wxyz` |
| `ray_hit`, `nearest_on_ray`, `vertices_in_sphere`, `vertices_in_rect` (in `viewer.tools`) | `vertices=`, `direction=`, `radius=`, `rect_min=` | keyword-only after the first two arguments |
| click callback event with `screen_pos ((x, y),)` and `event_type` | viser's click event | `Viewer.on_pointer("click", ...)` passes it: `screen_pos` is `(x, y)`, no `event_type` |
| rect-select callback event with `screen_pos` | viser's rect-select event | its corners are `screen_min` and `screen_max` |
| `Viewer.record(path, *, steps, size=(960, 540), fps, gpu=False, camera)` | `Viewer.record(path, *, size=(1920, 1080), fps, steps, camera, fov, gpu=None, bitrate)` | `gpu` is the `nvidia-smi` index of the GPU to draw on (`gpu=True` used whichever GPU EGL chose); the call returns the path it wrote |
| `Viewer.record` to `.gif`, to `name.png` frames or to a directory of frames | `Viewer.record` to `.mp4`, `.webm` or one `.png` still | a still per call: `record("a.png", steps=[k])`; ffmpeg turns a video into a GIF or frames |
| `viewer.capture`: `HeadlessClient`, `write_frames`, `find_chrome`, `find_ffmpeg` | `Viewer.record` | the browser path is private (`src/better_robot/viewer/record.py`) |

## Module paths

Import from the package; the modules under it are private.

| 0.2 | 0.3.0 | note |
|---|---|---|
| `better_robot.data_model`, `better_robot.data_model.joint_models` | `better_robot.model` | |
| `better_robot.kinematics.state`, `better_robot.kinematics.jacobian` | `better_robot.kinematics` | |
| `better_robot.dynamics.state` | `better_robot.dynamics` | |
| `better_robot.dynamics.state_manifold` | — | `StateMultibody` had no caller |
| `better_robot.io.parsers`, `better_robot.io.builders`, `better_robot.io.ir` | `better_robot.io` | `io.parse_urdf`, `io.parse_mjcf`, `io.ModelBuilder`, the IR records |
| `better_robot.io.assets` | — | the asset resolvers are gone (see `io` above) |
| `better_robot.lie.types` | `better_robot.lie` | the classes are gone (see `lie` above) |
| `better_robot.lie.alignment` | `better_robot.lie` | `lie.umeyama` keeps its name |
| `better_robot.spatial.motion`, `better_robot.spatial.force`, `better_robot.spatial.inertia`, `better_robot.spatial.symmetric3`, `better_robot.spatial.ops`, `better_robot.spatial.mesh` | `better_robot.spatial` | |

## Behaviour under the same names

| 0.2 | 0.3.0 | note |
|---|---|---|
| joints ordered depth-first | joints ordered parents first, keeping the file's order wherever it already is | joint ids change for some robots (the G1); look joints up by name |
| joint 0 a `JointUniverse` | joint 0 a `JointFixed` named `"universe"` | joint 1 is the root joint |
| `check_quaternion_norm=True` checks joint 1 | `check_quaternion_norm=True` checks every quaternion joint | |
| `Model.integrate` promotes mixed dtypes | `kinematics.integrate` raises `exceptions.DtypeMismatchError` | no silent promotion |
| `q_neutral` as written in the file | `Model.q_neutral` clamped into the limits at build | |
| `Model` equal by value | `Model` equal by identity | `structure_key` compares structures by value |
| MJCF parsed by hand | `io.parse_mjcf` compiles through MuJoCo | needs `better-robot[mjcf]`; the loader reads the passive rigid tree |
| a second `Viewer.on_pointer` replaces the callback | `Viewer.clear_pointer` first | a second registration raises `exceptions.ArgumentError` |
| an unknown group in `Viewer.set_offset`, `Viewer.remove_group` raises `KeyError` | `exceptions.UnknownNameError` | a `KeyError` subclass |
| the same layer added twice, or a failing layer half-added | `exceptions.ArgumentError` | a failing layer leaves the viewer as it was |
| a raising `PoseGizmo` callback is printed | the gizmo stops | `wait()` raises the callback's error |

## Install extras and environment

| 0.2 | 0.3.0 | note |
|---|---|---|
| `better-robot[io-mjcf]` | `better-robot[mjcf]` | MuJoCo compiles MJCF |
| `better-robot[dev]` | the dev dependency group | `uv sync --group dev`; `test-oracles` holds the parity oracles |
| `better-robot[viewer]` with `viser>=1.0.18` | `better-robot[viewer]` with `viser>=1.0.30` and `pillow>=10` | |
| `better-robot[viewer]` with `viser>=1.0.30` | `better-robot[viewer]` with `viser>=1.1.1,<1.2` | `Viewer.record` drives the client of viser 1.1 |
| `$FFMPEG` | `$BETTER_ROBOT_FFMPEG` | the ffmpeg that `Viewer.record` writes an `.mp4` with, else the one on `PATH` |
| — | `better-robot[testing]` | pytest, for `better_robot.testing`'s pytest plugin |
| Python 3.10 and 3.11 | Python 3.12 or newer | |
