# P1 toolbox — BR public API against a general robotics toolbox

Scope: the whole public surface of BetterRobot (BR) at worktree head `23070f4`, compared with what a
Pinocchio / Drake / MuJoCo / OpenSim user expects. Every claim below was read from code or run with
`uv run python -c ...` (CPU). Snippet inputs: `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/p1tb/a.urdf`.
Budget note: the review used about 50k tokens, so the theme coverage is broad but shallow in places. Items
marked *unverified* were not run.

## Current state (facts, with paths)

- Top level (`src/better_robot/__init__.py:44-71`): `Model, ModelStructure, ModelValues, Frame, Joint, Body, load,
  ModelBuilder, SE3`, FK + 4 Jacobian functions, 6 dynamics functions. Small and coherent.
- `io` (`src/better_robot/io/__init__.py`): `load, build_model, register_parser`, IR dataclasses
  (`IRModel, IRBody, IRJoint, IRFrame, IRGeom, IRError`), `ModelBuilder`, and 5 asset resolvers. Parsers: URDF, MJCF,
  programmatic (`io/parsers/`). There is no public "file to IR" function: `load` goes straight to `Model`.
- `data_model`: frozen `Model` with about 60 public attributes, including `integrate`, `difference`,
  `random_configuration`, `fk`, `get_subtree/get_support`, `with_values`, `to`, `armature/damping/friction`,
  `lower_pos_limit/upper_pos_limit/velocity_limit/effort_limit`, and the mimic tools (`expand_configuration`,
  `data_model/reduced_coordinates.py`). Joint models: fixed, revolute, prismatic, helical, planar, spherical,
  translation, free_flyer, composite (`data_model/joint_models/`).
- `kinematics`: `forward_kinematics_state(model, q) -> KinematicState` (fields `q, joint_transform, frame_transform`),
  then `frame_jacobian(model, state, *, frame, reference)` and `joint_jacobian(model, state, *, joint, reference)`.
- `dynamics`: `inverse_dynamics, forward_dynamics, mass_matrix, centroidal_map, centroidal_momentum, center_of_mass`,
  all called as `(model, state, *, v=..., ...)`. `dynamics/state_manifold.py:17 StateMultibody` exists but is not exported.
- `terms`: `frame_pose(model, frame, *, target)`, `frame_position`, `joint_limits`, `rest`, `finite_difference(model, *,
  order, dt)`, `self_clearance(spheres, pairs, *, margin)`, `sphere_clearance`, `ik_problem`, `solve_ik`, `random_start`.
- `optim`: `Problem`, `solve(problem, solver, *, iters)`, `run(phases)`, `solve_layer`, `solve_constrained` (penalty /
  augmented-Lagrangian outer loop), LM / GN / LBFGS / TorchOptim, robust kernels, `check_jacobian`.
- `exceptions.py`: `BetterRobotError` with `ModelInconsistencyError, DeviceMismatchError, DtypeMismatchError,
  QuaternionNormError, ShapeError, BackendNotAvailableError`. `IRError` inherits both `BetterRobotError` and `ValueError`.
- Missing entirely (grep over `src/better_robot`, no hits): splines, retiming, trajectory or time-series types, controllers,
  simulation stepping, model composition or joint locking, calibration, writers or exporters (`to_urdf`, save),
  `logging.getLogger`, `warnings.warn`, and SRDF / SDF / USD parsers.
- `docs/reference/roadmap.md:64-80` already defers optimizer extensions, dynamics derivatives, B-splines and benchmarks.
  It says nothing about composition, controllers, simulation, exporters or naming.

## How references evolved (with sources)

- **Pinocchio, composition and reduction.** It keeps topology frozen and edits by building a new model:
  `appendModel(modelA, modelB, frameInModelA, aMb)` and `buildReducedModel` (lock joints)
  (`references/kin_dyn/pinocchio/include/pinocchio/algorithm/model.hpp:28,51`). Geometry lives in a separate
  `GeometryModel`. SRDF supplies reference configurations and disabled collision pairs
  (`references/kin_dyn/pinocchio/include/pinocchio/parsers/srdf.hpp`). Mimic support arrived late (3.x,
  `multibody/joint/joint-mimic.hpp`), after years of users writing it by hand, which is the same path BR took with
  `reduced_coordinates.py`.
- **Drake, immutable after `Finalize()`.** The digest says so explicitly (`references/design/drake.md:171,197`). Users
  kept asking to edit models, and Drake later added `MultibodyPlant::RemoveJoint` before Finalize
  (`references/sim/drake/multibody/plant/multibody_plant.h`). Drake ships first-class trajectory types
  (`references/sim/drake/common/trajectories/{bspline_trajectory,piecewise_polynomial,...}.h`) and controllers as
  separate systems (`references/sim/drake/systems/controllers/{inverse_dynamics_controller,pid_controlled_system}.h`).
  Integration lives outside the plant.
- **MuJoCo, from XML-only to a spec API.** For years the only way to edit a model was to change the XML. Version 3.2
  introduced `mjSpec`, a procedural edit / attach / compile API
  (https://mujoco.readthedocs.io/en/stable/programming/modeledit.html; typings at
  `references/sim/mjlab/typings/mujoco/_specs.pyi`). mjlab builds its whole entity and scene composition on it
  (`references/design/mjlab.md:9-34`). The lesson: an editable intermediate layer ends up as the composition API.
- **BEST_PRACTICES digest.** "Don't mix integration with dynamics" (`references/design/BEST_PRACTICES.md:181`), and
  "single source of truth for indexing" (`:73`). Both favour new bricks (sim, control) on top of the frozen model,
  not inside it.

## Findings (ranked, F1..Fn: severity, evidence, consequence)

**F1 — high (fix before merge): three names for one SE(3) quantity, and two names for one field.**
Evidence: `terms.frame_pose` and `lie.Pose` say *pose*; `Model.joint_placements`, `Model.with_values(frame_placements=)`
and `Frame.joint_placement` say *placement*; `KinematicState.joint_transform` and `.frame_transform` say *transform*.
`Frame.joint_placement` and the `with_values` keyword `frame_placements` are the same quantity under two names.
Consequence: users guess attribute names, and every new brick (trajectory, contact, viewer layer) adds a fourth
spelling. A rename is only cheap before external users exist.

**F2 — high: identifier arguments behave differently across modules.**
Evidence (run): `joint_jacobian(m, s, joint='j1')` raises a raw `TypeError: new(): invalid data type 'str'`, and
`m.get_subtree('j1')` raises `TypeError: tuple indices must be integers`. `frame_jacobian(frame=)`, `Model.joint_id(joint)`
and `frame_id(frame)` all accept `int | str`. `get_subtree(joint_id: int)` uses the parameter name `joint_id`, while
`joint_id()` itself takes `joint`. Argument kind also varies: `frame` is positional in `terms.frame_pose(model, frame, ...)`
but keyword-only in `kinematics.frame_jacobian(..., *, frame)`, and `pairs` is positional in
`terms.self_clearance(spheres, pairs, ...)` but keyword-only in `collision.self_distances(spheres, state, *, pairs)`.
Consequence: you cannot carry call-site habits from one module to the next, and the errors are unhelpful.

**F3 — high: the parsers silently drop joint dynamics parameters.**
Evidence (run): a URDF joint with `<dynamics damping="0.5" friction="0.2"/>` loads with `model.damping == [0.]` and
`model.friction == [0.]`. `IRJoint` has no fields for damping, friction or armature
(`io/ir.py`; fields `name..joint_model`), and the parsers never mention them (`grep -c` gives 0 in
`io/parsers/{urdf,mjcf}.py`). `build_model.py:883-885` fills them with defaults. The IR also says `lower/upper` where
`Model` says `lower_pos_limit/upper_pos_limit`.
Consequence: `forward_dynamics` (and anything a simulation brick builds later) is wrong for real robots, with no warning.

**F4 — medium: the exception hierarchy is mostly bypassed.**
Evidence: raise statements in `src/` are 83 `ValueError`, 20 `TypeError`, 9 `FileNotFoundError`, 5 `KeyError`, against 14
`ShapeError` and 6 `ModelInconsistencyError`. `BackendNotAvailableError` exists, yet torch is the only backend.
`IRError(BetterRobotError, ValueError)` is the right pattern, but only `io` uses it.
Consequence: `except BetterRobotError` misses most user mistakes, so downstream (BH, BVR) cannot catch "our" errors
reliably.

**F5 — medium: no model editing or composition.**
Evidence: `Model` is frozen. `with_values` changes tensors only. `ModelBuilder` builds from scratch. There is no
`append_model`, no lock-joints / reduced model, no add-frame-to-loaded-model, and no public `parse(path) -> IRModel` (the
IR exists but `load` hides it).
Consequence: you cannot attach a gripper to an arm, add a tool or marker frame, mount BH hands on a body, or lock a
joint for a sub-problem without hand-editing a URDF. Drake and MuJoCo both had to retrofit this (see above).

**F6 — medium: "batched by default" breaks at sampling.**
Evidence (run): `Model.random_configuration()` returns shape `(nq,)` and takes no `batch_shape`.
`terms.random_start(model, q0, generator)` is a second sampler under a different name, in a different package. The
default dtype is float32 (`io.build_model(dtype=torch.float32)`), and the user-facing docs do not state it.

**F7 — medium: no trajectory or time-series type.**
Evidence: trajectories are raw `(T, nq)` tensors named `"q"` inside a `Problem`, and `dt` is a float argument of
`finite_difference`. `StateMultibody` (q, v manifold) is internal. There are no time stamps, no interpolation
(`Model.integrate` and `difference` exist but are not wired to time), no splines and no retiming.
Consequence: the viewer timeline, BH motion data, BVR fits and trajectory optimization each invent their own
container. This is the most likely source of future duplication across the three repos.

**F8 — medium: no control or simulation hook.**
Evidence: there is no integrator step (for example semi-implicit Euler from `forward_dynamics` plus `Model.integrate`),
no PD or computed-torque helper, and no contact. The good news: following `BEST_PRACTICES.md:181`, these can be new
top-level bricks without touching the core. They do need F7's state and time-series type first.

**F9 — low-medium: the `io` surface is uneven.**
Evidence: `load(format=...)` shadows the `format` builtin. There are two ways to request a floating base:
`load(free_flyer=True)` and `load(root_joint=JointFreeFlyer())` (`io/__init__.py:66,86`). There is no SRDF (named
configurations, disabled collision pairs), no SDF or USD reader, and no writer at all. No round-trip means BH or
retarget outputs cannot be exported for MuJoCo or Isaac users.

**F10 — low: two FK entry points.**
`Model.fk` is shorthand for `forward_kinematics_state`, but there is no matching shorthand for anything else. Fine if
it is documented as the only method shortcut. Otherwise it invites `Model.jacobian`, `Model.rnea`, and so on.

**F11 — low: no logging, progress hooks or public testing helpers.**
Evidence: there are zero loggers and zero warnings in `src/`. Solver progress is visible only in `Result`.
`optim.check_jacobian` is the only public testing utility. *Unverified*: whether BH re-implements SE3 or model
asserts in its tests.

## Roadmap (table: brick | why | prerequisite | when)

| Brick | Why (which user) | Prerequisite in the architecture | When |
|---|---|---|---|
| One SE(3) vocabulary (for example *placement* = constant model offset, *pose* = computed world SE(3)) applied to `Frame`, `KinematicState`, `with_values`, `terms` | every user; stops name drift before BH and BVR pin to it | owner choice of words (Q1) | now |
| `int \| str` for every joint, frame or body argument, resolved once at the public boundary via `model.joint_id()` and friends; uniform keyword-only after `(model, state)` | every user | none | now |
| Parse URDF `<dynamics>` and MJCF `armature`, `damping`, `frictionloss` into new optional `IRJoint` fields | dynamics and sim users | additive IR fields with zero defaults | now |
| `io.parse(source, format=) -> IRModel` public; `load = build_model(parse(...))` | composition, BH, tooling | none (the IR already exists) | now |
| `random_configuration(batch_shape=(), generator=None)` | sampling-based planners, IK restarts | none | now |
| Error subclasses with dual inheritance (`UnknownNameError(BetterRobotError, KeyError)`, `InvalidArgumentError(BetterRobotError, ValueError)`), plus a contract test that public wrappers raise only `BetterRobotError` | downstream libraries (BH, BVR) | none; existing `except ValueError` stays valid | now (types), next (sweep the 83 raises) |
| IR-level editing: `ir_attach(ir_a, ir_b, at_frame, placement)`, `ir_lock_joints(ir, names, q)`, `ir_add_frame`; `Model` stays frozen | gripper on arm, BH hands, tool frames | `io.parse` public (above) | next |
| `Trajectory` container: `t: (T,)`, `q: (..., T, nq)`, optional `v`, manifold interpolation via `Model.integrate/difference`, `.to()`, slicing | viewer timeline, BH motion, BVR, trajopt | F1 naming; export `StateMultibody` or a similar `(q, v)` state | next |
| SRDF parser (named poses, disabled pairs feeding `self_collision_pairs`) | collision and IK users | `io.parse` and IR meta | next |
| Manifold B-splines plus retiming (TOPP-RA style) | planners, robot playback | `Trajectory`; roadmap's deferred B-spline contract | later |
| `better_robot.control`: PD, computed torque (`inverse_dynamics`), QP whole-body control on `solve_constrained` | control researchers | `Trajectory`, state type; F3 fixed | later |
| `better_robot.sim`: `step(model, state, tau, dt)` semi-implicit Euler, contact later | RL and sim users | F3, F7, control; integrators kept out of dynamics | later |
| Writers: IR to URDF / MJCF; lazy optional USD / SDF readers | export to MuJoCo or Isaac, BH asset export | IR is the single interchange layer | later |
| Solver progress callback or logger hook; `better_robot.testing` (assert_se3_close, random-tree model factory) | long solves; BH and BVR tests | none | maybe |
| Calibration (kinematic parameter identification via `with_values` plus optim) | real-robot users | `with_values` differentiable (it is) | maybe |

## Prep changes recommended now (each ≤ 1 day, non-breaking)

1. **Name resolution at the boundary.** Make `joint_jacobian`, `joint_jacobian_time_variation` and `get_subtree` accept
   `int | str` through `model.joint_id()`. Add a parametrized contract test over every public function that takes a
   `joint`, `frame` or `body` argument (int, str, unknown name gives a `BetterRobotError` subclass).
2. **Joint dynamics in IR.** Add `damping`, `friction` and `armature` (default 0.0) to `IRJoint`, read URDF `<dynamics>`
   and MJCF `damping`, `frictionloss` and `armature`, and pass them through `build_model`. Add one test with the URDF above.
3. **`io.parse`.** Expose the existing parser dispatch as `parse(source, *, format) -> IRModel`. `load` keeps its
   signature.
4. **`random_configuration(batch_shape=(), generator=None)`.** Additive keyword. Optionally have `terms.random_start`
   call it.
5. **Error types.** Add `UnknownNameError(BetterRobotError, KeyError)` and `InvalidArgumentError(BetterRobotError,
   ValueError)`. Use them in the name lookups and the `io` boundary first, and sweep the rest next. Decide whether
   `BackendNotAvailableError` stays (Q4).
6. **Naming glossary.** Write the chosen SE(3), limit and identifier vocabulary into `docs/reference/glossary.md` and
   `docs/contracts/`, so the rename (or its deferral) is an explicit decision. Add a `Trajectory`/state placeholder
   entry to `docs/reference/roadmap.md` so the viewer, BH and BVR do not each build their own.

## Open questions for the owner

1. **SE(3) vocabulary.** Rename before merge (breaking but pre-release), or keep current names and only document them?
   Recommended: rename now, to *placement* for model constants and *pose* for computed world values, and retire
   *transform*.
2. **Argument style.** Should identifiers (`frame`, `joint`, `pairs`) always be keyword-only after `(model, state)`,
   including in `terms`? Recommended: yes.
3. **Where composition lives.** On `IRModel` (keeps `Model` frozen, mirrors MuJoCo `mjSpec`), or as
   `Model -> Model` functions like Pinocchio `appendModel`? Recommended: IR.
4. **`BackendNotAvailableError`.** Keep it as a reserved hook for a future kernel brick, or remove it?
5. **Ownership of the shared `Trajectory` type.** Does it belong in BR (used by BH and BVR), and is it in scope for the
   next task?
