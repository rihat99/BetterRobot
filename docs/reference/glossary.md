# Glossary

One vocabulary for BetterRobot and the libraries built on it. Each term has one meaning here; where a word still
carries two, its row says so. BetterHuman's glossary links this page and adds only its own words.

Every word of every public name, method, field and parameter is on this page: a term in the bold first column of a
table, or an [everyday word](#everyday-words) at the end. A term may span several words of a name (`free flyer`
spells `add_free_flyer_root`), and a plural matches its singular.
`tests/contract/test_vocabulary.py::test_glossary_words` checks it; a change that adds a word adds it here.

## How names are built

- **Plural for an axis over named things.** A tensor name is plural exactly when its shape has an axis over joints,
  frames, bodies, vertices or other named things: `state.frame_poses (B..., nframes, 7)`, and
  `state.frame_pose_of("hand") (B..., 7)` for one frame. Batch axes and component axes never make a name plural:
  `q (B..., nq)`, `lower_position_limit (nq,)`.
- **`_ids` for integer indices.** A field of integer indices ends in `_ids` (`parent_joint_ids`,
  `frame_parent_joint_ids`, `mimic_source_ids`), and a field that ends in `_ids` holds integers. A bare plural holds
  objects or values (`joints`, `joint_placements`). `nqs`, `nvs`, `idx_qs` and `idx_vs` give the width and start of
  each joint's slice of `q` and `v`; they are not ids.
- **Transform and affine.** A *transform* is a rigid `[R | p]` matrix `(..., 3, 4)`. An *affine* is a `(..., 3, 4)`
  that may also scale or shear; BetterRobot has none, BetterHuman's skinning uses them (`skin_affines`).
  *Placement* and *pose* are 7-vectors ([C-NAME-5](../contracts/01-naming.md)).
- **`forward_kinematics` and `forward`.** BetterRobot's `forward_kinematics(model, q)` computes kinematics only.
  BetterHuman's `forward` computes kinematics plus skinning. Two computations, two names.
- **`to`.** BetterRobot moves a model with `Model.to(device=, dtype=)`. BetterHuman chooses dtype and device once, at
  load, and has no `to`: a human model moves by loading it again (BetterHuman decision record 0005).
- **Acronyms.** Capitals inside CapWords (`IRModel`, `LBFGS`), lower case inside snake_case (`parse_urdf`,
  `solve_ik`).

## Robot model

| Term | Meaning |
|---|---|
| **BetterRobot** | This library, `better_robot`: batched robot kinematics, dynamics and nonlinear least squares in PyTorch. |
| **configuration, `q`** | The numbers that place every joint, `(B..., nq)`: one slice per joint, in joint order. In `lie.so3`, `q` is a unit quaternion instead. |
| **generalized velocity, `v`** | The tangent of `q`, `(B..., nv)`: a rate of motion in the configuration space, one slice per joint. It is never added to `q` (see **retraction**). In `lie.se3.exp`, `v` is an se(3) tangent. |
| **generalized acceleration, `a`** | The time derivative of `v`, `(B..., nv)`. |
| **generalized force, `tau`** | τ, the force or torque at each coordinate of `v`, `(B..., nv)`. |
| **`nq`, `nv`** | The widths of `q` and `v`. A quaternion stores four numbers for three degrees of freedom, so `nq` can exceed `nv`. |
| **`nqs`, `nvs`, `idx_qs`, `idx_vs`** | The width and the start (`idx`, index) of each joint's slice of `q` and `v`. |
| **full layout, `_full`** | The coordinates of every joint, mimic joints included: `nq_full`, `idx_qs_full`, and `kinematics.full_configuration(model, q)`, which fills them in. |
| **degree of freedom** | One independent direction in which the robot can move. |
| **joint** | The rule that lets one part of the robot move relative to its parent. Joint 0 is the universe. |
| **joint kind, kind** | The class of a joint, a `JointModel` (`JointRevolute`, `JointFreeFlyer`, ...): its coordinates, their manifold and the motion they make. |
| **universe** | Joint 0: fixed, identity placement, parent `-1`, no coordinates. Body 0 and frame 0 carry its name. |
| **root** | Joint 1, the joint that holds the root body under the universe; a free flyer there floats the base. In `testing` and the viewer, the directory or scene path that other paths are relative to. |
| **body, link** | A rigid part with mass and inertia; URDF calls it a link. Every body has a frame of its own name. |
| **frame** | A named coordinate system that moves with a joint: each body's own frame, a tool, a sensor or a user point. Always a robot frame, never an animation frame ([C-NAME-4](../contracts/01-naming.md)). |
| **parent, child** | In the kinematic tree, the joint or body one step towards the universe, and one step away from it. |
| **kinematic tree** | The parent-to-child structure that connects the robot's joints and bodies. |
| **subtree, support** | `kinematics.subtree(model, joint)` is the joint and every joint below it; `kinematics.support(model, joint)` the joints from the universe down to it. |
| **fixed base** | A robot whose root does not move relative to the world. |
| **floating base** | A robot whose root is a free-flyer joint. |
| **mimic joint, mimic** | A joint whose coordinate follows another's: `q_full[j] = multiplier * q[source] + offset`. It has no slice of `q` or `v` and appears only in the full layout. |
| **neutral configuration, neutral** | The configuration that puts every joint at its identity placement, `Model.q_neutral (nq,)`; `JointModel.neutral()` for one kind. |
| **position limit** | `lower_position_limit` and `upper_position_limit`, `(nq,)`: the range of each coordinate of `q`. |
| **velocity limit, effort limit** | The bounds on the speed and on the force or torque of each coordinate of `v`, `(nv,)`. |
| **armature** | The rotor inertia added to the diagonal of the mass matrix, `(nv,)`, as MJCF defines it. |
| **damping** | Two meanings. On `Model`, the viscous joint friction of each coordinate of `v`, `(nv,)`. In `optim`, the Levenberg-Marquardt knob that regularizes the linear system and sets how cautious a step is. |
| **gravity** | `Model.gravity (6,)`: the world's gravity as a spatial acceleration, `[linear, angular]`. |
| **inertia** | The packed rigid-body inertia `(..., 10)`: mass, center of mass and rotational inertia, in the layout of the [coupling standard](../contracts/coupling-standard.md). `Model.body_inertias (I..., nbodies, 10)`. |
| **rotational inertia** | The 3x3 inertia matrix about the center of mass, `(..., 3, 3)` (`spatial.rotational_inertia`). |
| **`Model`** | One flat frozen record: static fields (names, joints, parent joint ids, index tables) and tensor fields (placements, inertias, limits, gravity, `q_neutral`). Treat its contents as read-only. |
| **structure key** | `Model.structure_key`: a hashable value of the static fields. Two models with equal keys share every plan and index table. |
| **`KinematicState`** | The frozen result of `forward_kinematics(model, q)`: world transforms of every joint and frame, `(B..., n, 3, 4)`, and the poses, rotations, translations and Jacobian columns read from them. Jacobians and the dynamics algorithms take it; each query makes its own. |
| **`ModelBuilder`** | Builds a model in code: `add_body`, `add_joint`, `add_frame` and `add_shape` by name, then `finalize` returns the `IRModel`. |
| **id, `_ids`** | The integer index of a joint, frame or body (`Model.joint_id`, `frame_ids`). Every public boundary takes the id or the name. |

## Joint kinds

The `q` and `v` layout of each kind is in the [coupling standard](../contracts/coupling-standard.md).

| Term | Meaning |
|---|---|
| **fixed** | No motion: the child is rigidly placed in the parent (`JointFixed`). |
| **revolute** | Rotation about an axis by an angle (`JointRevolute`). |
| **unbounded, continuous** | A revolute joint without limits, `q = [cos, sin]` of its angle (`JointRevoluteUnbounded`; URDF's `continuous`). |
| **prismatic** | Translation along an axis (`JointPrismatic`). |
| **helical** | Rotation about an axis with translation along it, `pitch` metres per radian: a screw (`JointHelical`). |
| **spherical** | Rotation in 3D: `q` a unit quaternion, `v` the angular velocity in the child frame (`JointSpherical`). |
| **planar** | Translation in a plane and rotation about its normal (`JointPlanar`). |
| **translation** | Translation in 3D, `q = [x, y, z]` (`JointTranslation`). Elsewhere, the position part `p` of a pose, `(..., 3)` (`frame_translations`, `se3.from_translation`). |
| **free flyer** | Free rigid motion: `q` a pose `(7,)`, `v` a twist `(6,)` in the child frame. The root joint of a floating base. |
| **motion subspace** | `S (B..., 6, nv)` of a joint kind: the child's twist in its own frame is `S @ v`. |

## Poses and motion

| Term | Meaning |
|---|---|
| **pose** | A position and an orientation together, computed in the world frame, `(..., 7)`: `[tx, ty, tz, qx, qy, qz, qw]`. |
| **placement** | A pose expressed in the parent frame, as a 7-vector: the model constants `joint_placements` and `frame_placements`, and a joint kind's q-dependent `JointModel.placement(q)`. |
| **transform** | A world pose written as a rigid `(..., 3, 4)` matrix `[R | p]` (`frame_transforms`). As a verb, re-expressing a quantity in another frame (`spatial.transform_force`, `transform_motion`, `transform_inertia`). |
| **affine** | A `(..., 3, 4)` matrix that may scale or shear as well as rotate and translate. BetterRobot has none; BetterHuman's skinning uses them. |
| **world** | The fixed frame that every pose is expressed in (`world_geometry`). |
| **rotation** | A 3D rotation: a unit quaternion `(..., 4)` in poses and `lie.so3`, a matrix `R (..., 3, 3)` in `frame_rotations` and `box_rotation`. |
| **quaternion, rotation quaternion** | Four numbers `[qx, qy, qz, qw]`, scalar last, of unit norm: a 3D rotation without the singularities of three angles. |
| **quaternion double cover** | A quaternion and its negation represent the same rotation. |
| **Euler angles, Euler** | `[roll, pitch, yaw]`, the rotation `Rz(yaw) Ry(pitch) Rx(roll)` (`so3.from_euler`, `so3.to_euler`). |
| **SO(3)** | The group of 3D rotations; `lie.so3` works on it. |
| **SE(3)** | The group of rigid 3D poses, a rotation plus a translation; `lie.se3` works on it. |
| **Lie group, Lie** | A manifold that is also a group, here SO(3) and SE(3). `better_robot.lie` holds their operations as tensor functions. |
| **compose** | The group product `a b`: the pose of C in A is `X_AC = compose(X_AB, X_BC)` (`se3.compose`, `so3.compose`). |
| **act** | Apply a rotation or a pose to points `p (..., 3)` (`so3.act`, `se3.act`). |
| **exp, log** | The exponential map from a tangent to the group, and its inverse (`se3.exp (..., 6) -> (..., 7)`, `so3.log (..., 4) -> (..., 3)`). |
| **hat, vee** | `hat` writes a tangent as its matrix (skew-symmetric 3x3 for SO(3), 4x4 for SE(3)); `vee` reads the tangent back. |
| **left Jacobian, right Jacobian** | The Jacobians of the exponential map, `Jl(ξ) = Jr(-ξ)` (`lie.tangents`). |
| **slerp, sclerp** | Spherical linear interpolation of rotations, and screw linear interpolation of poses, at a fraction `t` in `[0, 1]`. |
| **Umeyama alignment, Umeyama** | The weighted least-squares similarity (rotation, translation and an optional scale) that maps one point set onto another (`lie.umeyama`). |
| **tangent** | A small local displacement from a point on a curved space. It lets an optimizer take ordinary vector steps on rotations and robot configurations. |
| **twist** | A six-number tangent of a rigid pose: linear motion followed by angular motion. |
| **wrench** | A six-number force quantity: force followed by torque. |
| **motion, force** | In spatial algebra, a motion vector is twist-like (a velocity or an acceleration) and a force vector wrench-like. `transform_motion` and `transform_force` change their frame. |
| **cross** | The spatial cross products: `cross_motion(twist, other) = ad(twist) @ other` and `cross_force(twist, wrench) = -ad(twist)^T @ wrench`. |
| **adjoint** | `Ad(T) (..., 6, 6)`: the matrix that re-expresses a twist given in frame B in frame A, for the pose `T` of B in A (`se3.adjoint`). |
| **spatial algebra, spatial** | Six-dimensional vector algebra of twists, wrenches and inertias; `better_robot.spatial`. |
| **retraction, retract, integrate** | Apply a tangent step and return to the manifold. `kinematics.integrate(model, q, v)` is the robot-configuration retraction; `Manifold.retract(x, delta)` the optimizer's. |
| **difference** | The tangent that moves one manifold value to another. `kinematics.difference(model, q0, q1)` returns it for robot configurations. |
| **manifold** | A space that is locally vector-like but may be curved globally, such as rotations or unit quaternions. An `optim.Manifold` tells a solver how to step a variable on one. |
| **Euclidean** | A flat space, where a step is an addition (`optim.Euclidean`; `kinematics.euclidean_columns`, the coordinates of joints that integrate by addition). |
| **closed manifold** | A triangle mesh without boundary: every edge is shared by exactly two faces (`spatial.validate_closed_manifold`). |

For a visual and mathematical introduction, see [Poses and twists](../lie-and-spatial.md).

## Kinematics and derivatives

| Term | Meaning |
|---|---|
| **kinematics** | The geometry of motion: where every frame is and how fast it moves, without forces (`better_robot.kinematics`). |
| **forward kinematics, FK** | Joint configuration in, poses out: `forward_kinematics(model, q)` returns a `KinematicState`. |
| **inverse kinematics, IK** | Desired frame poses in, a joint configuration out. IK is solved as an optimization problem because several joints and targets interact. |
| **seed** | One start of a multi-start solve. `solve_ik(seeds=)` solves every seed of every target in one batch and keeps the best per target; the first seed is `q0`, the others random configurations. |
| **Jacobian** | A table that predicts how outputs move when inputs change a little. A frame Jacobian `(B..., 6, nv)` maps joint velocity to frame twist. |
| **Jacobian time variation, time variation** | `dJ/dt` along `v`, `(B..., 6, nv)` (`frame_jacobian_time_variation`). |
| **analytic Jacobian** | A derivative block written from a mathematical formula. It may be exact or a documented local approximation; either way, it is compared with automatic differentiation or finite differences. |
| **automatic differentiation** | PyTorch applies the chain rule through the executed tensor operations. It gives derivatives of a differentiable forward calculation without a hand-written final formula. |
| **finite difference** | Re-run a function after small numerical input changes and estimate its derivative. Approximate and slow, but an independent check. `terms.finite_difference` differentiates a trajectory in time this way. |
| **reference** | The frame a Jacobian is expressed in: `"local"`, `"world"` or `"local_world_aligned"`, the default. |
| **LOCAL** | A frame twist measured at the frame origin and expressed in that frame's axes. |
| **WORLD** | A spatial twist expressed in world axes and shifted to the world origin. |
| **LOCAL_WORLD_ALIGNED, LWA** | A twist measured at the frame origin but expressed in world axes. BetterRobot's default frame-Jacobian convention. |
| **manipulability, threshold** | Yoshikawa's measure `sqrt(det(J Jᵀ))` of a frame's position Jacobian `J`: zero at a singular configuration, where the frame cannot move in some direction. `terms.manipulability(frame=, threshold=)` pushes only while it is below `threshold`. |

See [Kinematics and Jacobians](../kinematics.md) for when BetterRobot uses each
derivative method.

## Dynamics

| Term | Meaning |
|---|---|
| **dynamics** | Motion together with the forces that cause it (`better_robot.dynamics`). |
| **inverse dynamics** | Given motion, compute the generalized forces that produce it. |
| **forward dynamics** | Given generalized forces, compute the acceleration. |
| **RNEA** | Recursive Newton-Euler Algorithm, used for inverse dynamics. |
| **ABA** | Articulated-Body Algorithm, used for forward dynamics. |
| **CRBA** | Composite Rigid-Body Algorithm, used to compute the mass matrix. |
| **mass matrix** | `M(q) (B..., nv, nv)`, the joint-space inertia that relates acceleration to generalized force. |
| **bias force** | The velocity-dependent and gravity forces present before any applied generalized force. |
| **external force** | A wrench applied to a body from outside, `external_forces (B..., njoints, 6)`. |
| **center of mass, COM** | The mass-weighted average position of the robot's bodies. |
| **centroidal momentum** | The robot's total linear and angular momentum about its center of mass, `(B..., 6)`. |
| **centroidal map** | `A_g(q) (B..., 6, nv)`, the matrix with `h_g = A_g v`: centroidal momentum from velocity. |

## Optimization

| Term | Meaning |
|---|---|
| **term** | A plain function of named tensors (variables and data) that returns a residual `(B, ..., k)` the solver drives towards zero, or a scalar cost. Its Jacobian comes from autograd unless the term supplies `jacobian`. |
| **`Problem`** | The container of variables, data and weighted terms. `Problem.loss` feeds autograd optimizers; `Problem.linearize` builds the normal equations for the Gauss-Newton family. |
| **variable** | A value the optimizer may change, `(B, *event)`. One problem can hold several variables with different shapes and manifolds; `x` is the dictionary of all of them. |
| **data** | Named tensors the terms read and the solver never changes (`Problem(data=)`). |
| **residual** | The vector a least-squares term returns, `(B, ..., k)`. |
| **loss** | The scalar per batch element a solver minimizes, `(B,)`: the weighted squared residuals plus the scalar terms. |
| **state** | What one computation leaves for the next: a `KinematicState` after forward kinematics, an `optim.State` after a solver iteration. |
| **step** | In `optim`, the update one iteration applies. In the viewer, one entry of the timeline. |
| **rest configuration, rest** | The configuration a regularizing term pulls towards (`terms.rest(model, q_rest)`, `ik_problem(rest_weight=)`). |
| **`Phase`** | One stage of a staged solve: term weight overrides, frozen coordinates, an iteration budget and an optional start hook, run in sequence by `optim.run`. |
| **cached value** | What `Problem.cached(fn, name=...)` registers and names: a value computed once per evaluation (for example the kinematic state) that every term reading that name shares. |
| **window** | The number of consecutive knots a time-local term reads. A window of 2 couples knot `t` with `t + 1`. |
| **normal equations, normal** | `H delta = -g`, the linear system of one Gauss-Newton step, with `H` and `g` from the Jacobian and the weighted residuals (`optim.Normal`). In geometry, a normal is a unit vector perpendicular to a surface (`sphere_halfspace(normal=)`). |
| **banded normal, arrow normal** | The block-tridiagonal normal equations of a trajectory problem (`BandedNormal`), and the same bordered by variables shared over time (`ArrowNormal`). Both solve in time linear in the number of knots. |
| **least squares** | Choose variables that make the sum of squared residual entries small. This balances many errors at once. |
| **quadratic program, QP** | Minimise `½ xᵀH x + gᵀx` subject to linear rows `lower <= rows x <= upper`: the constrained form of one Gauss-Newton step (`optim.solve_qp`). |
| **constraint term** | A term a solver holds instead of minimizing: `Problem.equality` holds `fn = 0`, `Problem.inequality` holds `lower <= fn <= upper`, each entry of the output one row of every step. It is no part of the loss. |
| **sequential quadratic programming, sequential QP** | Solve a problem with constraint terms by a sequence of quadratic programs: each step minimizes the damped Gauss-Newton model of the loss under the constraint terms linearized at the iterate (`optim.SequentialQP`). |
| **gain** | The share of a row's distance to its bound that one step of `SequentialQP` or `track` may use, and the share of a violation that it repairs (`SequentialQP(gain=)`, `track(gain=)`); 1 is the plain linearization. |
| **merit** | What a step of `SequentialQP` must lower to be taken: the loss plus the summed violation of the constraint rows times a price, the largest multiplier of the step's program capped by two prices that redundant rows do not raise. |
| **step radius** | The largest entry a step of `SequentialQP` may have (`State.extra["radius"]`): none at first, doubled by a taken step, shortened by a refused one once the damping has reached 1. |
| **track** | Follow targets that move between calls: `optim.track` takes one step of differential inverse kinematics, one quadratic program of the damped Gauss-Newton model under the constraint terms and bounds, per call. |
| **Hessian** | The matrix `H` of a quadratic objective: for one Gauss-Newton step, the `H` of the normal equations with its damping. |
| **row** | One linear constraint of a quadratic program, `lower <= a · x <= upper`; an equality when both bounds are equal, one-sided when one is infinite. An elastic row (`solve_qp`) gives way once holding it would take a multiplier beyond the cap that `weight` sets. |
| **primal, dual** | Of a constrained problem: its variables `x`, and the multipliers of its rows. The primal residual is the largest distance of a row from where its multiplier holds it (its violation where the multiplier is zero), the dual residual the largest entry of `H x + g + rowsᵀ dual`. |
| **robust kernel, kernel** | A function `rho` of a squared residual norm that reduces the influence of large residuals, often to limit outliers. |
| **Huber, Cauchy, Tukey, Geman-McClure** | The robust kernels of `optim`. Huber is quadratic inside `delta` and linear outside; Cauchy (Lorentzian) grows logarithmically; Geman-McClure is bounded; Tukey's biweight is flat beyond its cutoff `c`, so larger residuals get no weight. |
| **gradient** | The local direction in which a scalar objective increases fastest. |
| **Gauss-Newton, GN** | A least-squares method that linearizes residuals and solves the resulting local quadratic problem. |
| **Levenberg-Marquardt, LM** | Gauss-Newton with adjustable damping. Low damping takes a bolder Newton-like step; high damping a more cautious gradient-like step. |
| **L-BFGS** | Limited-memory BFGS: a quasi-Newton method that keeps a short `history` of steps and gradients (`optim.LBFGS`). |
| **line search, Armijo** | Shrink a step along its direction until the loss decreases enough: the Armijo condition (`LBFGS(armijo=)`). `LBFGS(stop_search=False)` tries every step length instead of stopping at the first that every problem accepts. |
| **Cholesky factorization** | A fast way to solve a positive-definite symmetric linear system. |
| **bound** | A lower or upper limit on a Euclidean coordinate. |
| **convergence** | The solver met its stopping rule. A returned result can still hold a useful final candidate when it did not converge. |
| **warm start, warm** | Reuse a previous candidate or solver state as the starting point for a related solve. |
| **implicit differentiation** | Differentiate the final optimality equation instead of storing every solver iteration (`optim.solve_layer`). |
| **temporal structure** | A trajectory problem in which each residual touches only nearby time steps. |
| **block-banded matrix** | Storage for a temporal linear system whose nonzero blocks stay near the diagonal. |

See [Optimisation](../optim-and-terms.md) for the complete problem and solver model.

## Batching and tensors

| Term | Meaning |
|---|---|
| **PyTorch, torch** | The tensor library BetterRobot runs on, and its only implementation. |
| **event shape** | Trailing axes that describe one value, such as `(nq,)` or `(6, nv)`. |
| **batch shape, `B...`** | Any leading axes that represent independent evaluations. |
| **model batch, `I...`** | Optional leading axes of a model's tensor values (`joint_placements (I..., njoints, 7)`): robots with one topology. A state's batch is `broadcast(B..., I...)`. |
| **batching** | Evaluate many independent inputs in one tensor call instead of a Python loop. |
| **broadcasting** | Reuse a size-one or missing batch axis across a compatible larger batch, following PyTorch rules. |
| **device** | Where a tensor lives, such as the CPU or one CUDA GPU. |
| **dtype** | The tensor's numeric type, such as `torch.float32` or `torch.float64`. |
| **compute pass** | One complete operation over the robot, such as FK or RNEA. |
| **compiled** | A call turned by `torch.compile` into one graph of fused kernels, made on request by `kinematics.compiled`; eager, the default, runs each torch operator as it comes. |

## Geometry and collision

| Term | Meaning |
|---|---|
| **`Geometry`, geometry** | The shapes attached to a model's bodies: a value beside the model, not a field of it (`io.load_geometry`). |
| **shape** | Two meanings. A tensor's sizes (`ShapeError`, `batch_shape`). A primitive or mesh attached to a body (`Shape`, `AttachedShape`, `ModelBuilder.add_shape`). |
| **collision** | A shape used for distance queries, as opposed to one for display only (`add_shape(collision=False)`). |
| **self-collision, self** | Contact between two parts of the same robot (`self_collision_pairs`, `self_distances`, `terms.self_clearance`). |
| **signed distance, distance** | The distance between two shapes, negative when they overlap. |
| **signed distance field, SDF** | A voxel grid of signed distances to an obstacle (`collision.sphere_sdf`). |
| **clearance, margin, hinge** | The residual `margin - distance` between shapes: positive when they are closer than `margin`. With `hinge=True` it is zero for a clear configuration, `gap² / (2 margin)` for a gap `margin - distance` below `margin`, then `gap - margin / 2`: a hinge whose slope has no jump (`terms.self_clearance`, `terms.sphere_clearance`). `terms.joint_limits(margin=)` uses the same hinge from `margin` inside the limits. |
| **closest** | `terms.self_clearance(closest=k)` keeps the `k` sphere pairs with the least room (distance less margin) in each configuration, `terms.sphere_clearance(closest=k)` the `k` smallest sphere-obstacle distances: a fixed number of rows. |
| **capsule** | A cylinder with a hemisphere on each end: a segment from `start` to `end` with a `radius`. |
| **half-space** | The solid on one side of a plane, `{x : normal . x <= offset}`. |
| **semi-axes** | The three half-lengths of an ellipsoid along its axes (`spatial.inertia_of_ellipsoid`). |
| **origin** | A frame's or a grid's zero point `(3,)`. In a robot description's `<origin>` and the builder (`add_joint(origin=)`, `MeshSpec(origin=)`), a placement `(7,)`. |

## Robot descriptions

| Term | Meaning |
|---|---|
| **description** | A URDF or MJCF file, or `ModelBuilder` calls, that describes a robot. Names stay exactly as it gives them. |
| **URDF** | An XML robot-description format common in ROS tooling. |
| **MJCF** | MuJoCo's XML model format. |
| **intermediate representation, IR** | The parser-neutral model that URDF, MJCF and `ModelBuilder` produce before `build_model` (`IRModel`, `IRJoint`, ...). |
| **URI** | A mesh's address as the description wrote it: a path or `package://...` (`Mesh.uri`). |

## Viewer

| Term | Meaning |
|---|---|
| **`Viewer`** | The browser viewer on viser. `viewer.add(layer, tags=...)` adds a layer or an iterable of layers, `viewer.show(step)` draws one step, `viewer.serve()` serves the page until interrupted, `viewer.record(path)` draws the live scene in a headless Chromium into a video (`.mp4` or `.webm`) or a still (`.png`). |
| **tag** | A name a layer carries (`Layer.tags`, `Viewer.add(tags=)`); the first is its scene parent. `Viewer.remove`, `set_visible`, `set_pose` and `redraw` address every layer of a tag. |
| **sidebar** | The viewer's panel of controls in ordered sections (`Sidebar`, `Layer.build_gui(sidebar)`). |
| **bounds** | The box `(lo, hi)` of what a layer draws at a step, in its first tag's frame (`Layer.bounds`), which `Viewer.aim` frames. |
| **redraw** | Push a layer's current step again after its arrays changed (`Layer.redraw`, `Viewer.redraw`). |
| **serve** | Keep the viewer's page served, playing the timeline, until interrupted (`Viewer.serve`). |
| **hue** | A robot's one colour (`Theme.hues`, `Theme.hue(index)`); its parts take shades of it. |
| **shade** | A hue mixed towards black or white for one role of a part, such as a joint or a bone (`Theme.shade`). |
| **backdrop** | The page behind the scene: a vertical gradient (`Theme.backdrop`). |
| **accent** | The colour of the viewer's controls and tools (`Theme.accent`). |
| **intrinsics** | A pinhole camera's `(3, 3)` matrix `K`: the focal lengths `fx, fy` and the principal point `cx, cy`, in pixels (`CameraLayer(intrinsics=)`). |
| **`Layer`** | One drawable thing (a robot, frames, points, a skeleton) holding a whole clip. The viewer calls `layer.set_step(k)`; `layer.nsteps` is its clip length. `optim.solve_layer` uses the word as deep learning does: a solve inside an autograd graph. |
| **timeline** | The steps the viewer plays (`Timeline`). It is as long as the longest layer; a shorter layer holds its last step. |
| **skinning, skin, skinned mesh** | Linear blend skinning: each vertex of a skinned mesh moves with a weighted blend of bone poses (`SkinnedMeshLayer.skin(step)`). |
| **bind pose, bind** | The pose in which a skinned mesh's vertices are given (`SkinnedMeshLayer(bind_poses=)`); for a `Skin`, the configuration `bind (nq,)` of its rest surface. |
| **visual** | A robot as one layer of the viewer (`RobotVisual`): its meshes or a skeleton, with a collision overlay and frames. |
| **style** | How a robot visual draws its robot: `"meshes"`, `"minimal"` or `"medium"` (`RobotVisual(style=)`, `RobotVisual.set_style`). |
| **minimal, medium** | The two skeleton styles and their recipes: `MINIMAL` is a stick figure with one ball per cluster of joints, `MEDIUM` one glyph per joint on thin pale rods, in two tones. |
| **recipe** | The choices that lay out a skeleton (`SkeletonRecipe`): glyphs, bones, limit arcs, clusters. Its `twists=False` leaves out the glyph of a joint that only turns its limb about the limb's own line, which is not the six-number twist. |
| **glyph** | The shape that stands for one joint in a skeleton: a drum for a revolute joint, a sleeve on a rail for a prismatic one, a ball otherwise (`SkeletonRecipe(glyphs=)`). |
| **cluster** | Joints close enough to share one ball in the minimal skeleton (`SkeletonRecipe(cluster=)`, a fraction of the robot's size). |
| **trimesh** | The mesh library; `MeshSpec.trimesh` holds one of its meshes, with textures or vertex colours. |
| **bitrate** | Bits per second of an encoded video; `Viewer.record(bitrate=)` trades file size for detail. |
| **tool** | A helper on top of the viewer that holds callbacks and sits outside the step loop, unlike a layer: a gizmo or picking (`better_robot.viewer.tools`). |
| **gizmo** | A handle in the scene a person drags to move or turn a pose (`tools.PoseGizmo`). |
| **ray** | A half-line from an origin along a direction, as a click in the browser casts it into the scene (`tools.ray_hit`, `tools.nearest_on_ray`). |
| **frustum, aspect** | What a browser's camera sees: its pose, its vertical field of view and its aspect, the view's width over its height (`tools.CameraFrustum`); picking projects points with it. |

## Testing

| Term | Meaning |
|---|---|
| **fake server** | `testing.FakeServer`: a viser server without sockets, for tests of the viewer. |
| **dead citation** | A repository path cited in tracked text that does not resolve (`testing.dead_citations`). |
| **snippet** | A Python fence of a docs page; `testing.snippets.check_page` runs each once and compares its output. |
| **glossary** | This page as data: `testing.glossary.read()` returns its terms and its everyday words, from the copy that ships in the package, so a library built on BetterRobot checks its names against them and against its own page (`read(page)`). |
| **vocabulary** | The words a public name may use: a glossary's terms and everyday words together. `testing.glossary.spelling(name_words, vocabulary)` spells a name with them. |

## Abbreviations and symbols

| Term | Meaning |
|---|---|
| **`a`, `b`** | The two operands of `compose(a, b)`. As a suffix, the first and second object of a pair (`start_a`, `radius_b`). |
| **API** | Application programming interface: the public names (`testing.public_api`). |
| **`c`** | A robust kernel's scale (`Cauchy(c=)`, `Tukey(c=)`). |
| **`dim`** | Dimension: `tangent_dim` is the size of a tangent. |
| **`dt`** | A time step in seconds. |
| **`fn`** | Function (`TermFn`, `distance_fn`). |
| **`fov`** | A camera's field of view, in radians. |
| **`fps`** | Steps per second of playback ("frames per second"; the one place where *frame* means an image). |
| **`g`, `H`** | The gradient and the Gauss-Newton matrix of the normal equations. |
| **GPU** | Graphics processing unit (`Viewer.record(gpu=)`). |
| **GUI** | Graphical user interface: the viewer's sidebar controls (`Layer.build_gui`, `Sidebar.gui`). |
| **`init`** | Initialize: `Solver.init(problem, x)` returns the first state. |
| **I/O** | Input and output: reading robot descriptions (`better_robot.io`). |
| **`min`, `max`** | Minimum and maximum (`ArrowsLayer(min_length=)`, `tools.vertices_in_rect(rect_max=)`). |
| **`njoints`, `nframes`, `nbodies`, `nsteps`** | The numbers of joints (the universe included), frames, bodies and viewer steps. |
| **`omega`, `w`, `W`** | ω, a rotation vector `(..., 3)`: an SO(3) tangent, axis times angle. `W` is its skew-symmetric matrix `(..., 3, 3)`. `w` and `omega` name the same quantity. |
| **optim** | Optimization (`better_robot.optim`; `TorchOptim` runs a `torch.optim` optimizer). |
| **`p`** | Points `(..., 3)` that a rotation or a pose acts on. |
| **`q0`, `q1`, `q2`** | Numbered configurations: `difference(model, q0, q1)` goes from `q0` to `q1`, and `q0` starts a solve (`solve_ik(q0=)`). In `so3.slerp(q1, q2, t)`, two unit quaternions. |
| **`R`** | A rotation matrix `(..., 3, 3)`. |
| **rect** | Rectangle: the screen box a person drags in viser's rectangle selection (`tools.vertices_in_rect`). |
| **repo, src** | Repository and source directory (`testing.dead_citations(repo_root)`, `testing.imported_packages(src_root, package)`). |
| **`rho`** | ρ, a robust kernel's loss of a squared residual norm. |
| **`s`** | Seconds, as a unit suffix (`timeout_s`). |
| **spec** | Specification: what to draw (`MeshSpec`). |
| **`t`, `T1`, `T2`** | In `lie.se3`, a pose `(..., 7)`; `T1` and `T2` are the two poses `sclerp` interpolates. In `slerp` and `sclerp`, `t` is the fraction in `[0, 1]`. |
| **URL** | The viewer's web address (`Viewer.url`). |
| **`x`, `X`** | In `optim`, the variables `{name: (B, *event)}`. In `vee_se3`, `X` is the 4x4 matrix of an SE(3) tangent. |
| **`xi`** | ξ, an SE(3) tangent `(..., 6)` in `lie.tangents`; `se3.exp` names the same quantity `v`. |

## Everyday words

These words keep their ordinary English meaning in public names.

actual, add, aim, angle, apply, argument, arrow, at, atomic, attached, axes, axis, azimuth, band, bone, border, box,
build, button, by, cached, callback, camera, cell, center, change, check, checkbox, clear, click, client, color, column,
component, consistently, constrained, corner, create, cylinder, dark, decrease, delta, dense, density, dependency,
diagnostics, diagonal, direction, dispose, done, dropdown, element, elevation, ellipsoid, end, entry, equality, error,
estimate, evaluate, every, exact, exception, expected, extend, extent, extra, face, factory, field, fill, finalize,
flat, floor, folder, forever, format, fraction, free, from, generator, get, graph, grid, ground, growth, half, head,
height, high, history, hit, host, identity, ignore, image, imported, in, index, inequality, install, inverse, iteration,
keep, key, label, layout, length, light, limit, line, linearize, load, lock, loop, low, lower, masked, mass, match,
matrix, mesh, mismatch, mode, multiplier, name, nearest, new, no, norm, normalize, object, of, offset, old, on, opacity,
option, optional, order, orient, other, outer, output, package, page, pair, panel, parse, path, pause, penalty,
permutation, pitch, plane, play, playing, plot, point, pointer, port, position, predicted, project, public, radii,
radius, ramp, random, range, read, reason, record, remove, replace, resize, restrict, result, rigid, robot, role, run,
save, scalar, scale, scene, scheduler, search, section, segment, serializer, series, server, set, shading, shadow,
shaft, show, side, size, skeleton, skip, sleep, slice, slider, solution, solve, solver, source, spelling, sphere,
squared, start, stop, target, testing, text, theme, time, timeout, to, tolerance, top, unknown, up, update,
upper, valid, validate, value, vector, vertices, visible, voxel, wait, weight, width, with, word
