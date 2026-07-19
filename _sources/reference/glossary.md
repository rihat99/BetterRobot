# Glossary

This page defines the words used by the tutorials before giving their compact
mathematical names.

## Robot model

| Term | Meaning |
|---|---|
| **configuration, `q`** | The numbers that describe the robot's joint state. Its trailing size is `nq`. |
| **generalized velocity, `v`** | A direction and rate of motion in the configuration space. Its trailing size is `nv`. |
| **degree of freedom** | One independent direction in which the robot can move. A quaternion uses four stored numbers for three rotational degrees of freedom. |
| **joint** | The rule that allows one part of the robot to move relative to its parent. |
| **body** | A rigid part with mass and inertia. |
| **frame** | A named coordinate system attached to the robot. A frame may mark a joint, body, tool, sensor, or user point. |
| **kinematic tree** | The parent-to-child structure connecting the robot's joints and bodies. |
| **fixed base** | A robot whose root does not move relative to the world. |
| **floating base** | A robot whose root is a free-flyer joint with translation and rotation. |
| **`Model`** | Robot structure and tensor values shared across evaluations. Treat its contents as read-only. |
| **`ModelStructure`** | Immutable topology, index tables, and joint metadata. |
| **`ModelValues`** | Differentiable model tensors such as placements, inertias, limits, and gravity. |
| **`Data`** | Mutable results and caches for one evaluation. |

## Poses and motion

| Term | Meaning |
|---|---|
| **pose** | A position and orientation together. BetterRobot stores one as `[tx, ty, tz, qx, qy, qz, qw]`. |
| **rotation quaternion** | Four numbers `[qx, qy, qz, qw]` that represent a 3D rotation without the singularities of three-angle coordinates. The four numbers have unit norm, so they contain three independent degrees of freedom. |
| **quaternion double cover** | A quaternion and its negation represent the same physical rotation. |
| **SO(3)** | The mathematical space of all 3D rotations. |
| **SE(3)** | The mathematical space of all rigid 3D poses: a rotation plus a translation. |
| **tangent** | A small local displacement from a point on a curved space. It lets an optimizer take ordinary vector steps on rotations and robot configurations. |
| **twist** | A six-number tangent for a rigid pose: linear motion followed by angular motion. |
| **wrench** | A six-number force quantity: force followed by torque. |
| **retraction** | An operation that applies a tangent step and returns to the manifold. `Model.integrate` is the robot-configuration retraction. |
| **difference** | The local tangent that moves one manifold value to another. `Model.difference(q0, q1)` returns it for robot configurations. |
| **manifold** | A space that is locally vector-like but may be curved globally, such as rotations or unit quaternions. |

For a visual and mathematical introduction, see
{doc}`/concepts/lie_and_spatial`.

## Kinematics and derivatives

| Term | Meaning |
|---|---|
| **forward kinematics, FK** | Joint configuration in, poses out. FK answers where every joint and frame is. |
| **inverse kinematics, IK** | Desired frame poses in, a joint configuration out. IK is solved as an optimization problem because several joints and targets interact. |
| **Jacobian** | A table that predicts how outputs move when inputs change a little. A frame Jacobian maps joint velocity to frame twist. |
| **analytic Jacobian** | A derivative block written from a mathematical formula. It may be exact or a documented local approximation; either way, it must be compared with automatic differentiation or finite differences. |
| **automatic differentiation** | PyTorch applies the chain rule through the executed tensor operations. It gives derivatives of a differentiable forward calculation without a hand-written final formula. |
| **finite differences** | Re-run a function after small numerical input changes and estimate its derivative. It is approximate and slow but useful as an independent debugging check. |
| **LOCAL** | A frame twist measured at the frame origin and expressed in that frame's axes. |
| **WORLD** | A spatial twist expressed in world axes and shifted to the world origin. |
| **LOCAL_WORLD_ALIGNED, LWA** | A twist measured at the frame origin but expressed in world axes. This is BetterRobot's default frame-Jacobian convention. |

See {doc}`/concepts/kinematics_and_jacobians` for when BetterRobot uses each
derivative method.

## Optimization

| Term | Meaning |
|---|---|
| **variable** | A value the optimizer may change. One problem can contain several variables with different shapes and manifolds. |
| **residual** | A vector of errors that should approach zero, such as position error or distance from a rest pose. |
| **least squares** | Choose variables that make the sum of squared residual entries small. This balances many errors at once. |
| **weight** | A multiplier that makes one residual matter more or less than another. |
| **robust kernel** | A function that reduces the influence of large residual groups, often to limit the effect of outliers. |
| **gradient** | The local direction in which a scalar objective increases fastest. |
| **Gauss--Newton, GN** | A least-squares method that linearizes residuals and solves the resulting local quadratic problem. |
| **Levenberg--Marquardt, LM** | Gauss--Newton with adjustable damping. Low damping takes a bolder Newton-like step; high damping takes a more cautious gradient-like step. |
| **damping** | The LM knob that regularizes the local linear system and controls step caution. |
| **Cholesky factorization** | A fast way to solve a positive-definite symmetric linear system. |
| **bound** | A lower or upper limit on a Euclidean state coordinate. |
| **convergence** | The solver met its stopping rule. A returned result can still contain a useful final candidate when it did not converge. |
| **warm start** | Reuse a previous candidate or solver state as the starting point for a related solve. |
| **implicit differentiation** | Differentiate the final optimality equation instead of storing every solver iteration. BetterRobot offers this only for eligible generic dense solves. |
| **temporal structure** | A trajectory problem in which each residual touches only nearby time steps. |
| **block-banded matrix** | Storage for a temporal linear system whose nonzero blocks stay near the diagonal. |

See {doc}`/concepts/residuals_costs_and_solvers` for the complete problem and
solver model.

## Batching and tensors

| Term | Meaning |
|---|---|
| **event shape** | Trailing axes that describe one value, such as `(nq,)` or `(6, nv)`. |
| **batch shape, `B...`** | Any leading axes that represent independent evaluations. |
| **batching** | Evaluate many independent inputs in one tensor call instead of a Python loop. |
| **broadcasting** | Reuse a size-one or missing batch axis across a compatible larger batch, following PyTorch rules. |
| **device** | Where a tensor lives, such as CPU or a particular CUDA GPU. |
| **dtype** | The tensor's numeric representation, such as `torch.float32` or `torch.float64`. |
| **compute pass** | One complete operation over the robot, such as FK or RNEA. |

## Dynamics

| Term | Meaning |
|---|---|
| **inverse dynamics** | Given motion, compute the generalized forces required to produce it. |
| **forward dynamics** | Given generalized forces, compute acceleration. |
| **RNEA** | Recursive Newton--Euler Algorithm, used for inverse dynamics. |
| **ABA** | Articulated-Body Algorithm, used for forward dynamics. |
| **CRBA** | Composite Rigid-Body Algorithm, used to compute the joint-space mass matrix. |
| **CCRBA** | A composite rigid-body pass that also computes the centroidal momentum map and momentum. |
| **mass matrix** | `M(q)`, the joint-space inertia relating acceleration to generalized force. |
| **bias forces** | Velocity-dependent and gravity forces present before applied generalized force. |
| **gravity torque** | Generalized force caused by gravity at a configuration. |
| **center of mass, COM** | Mass-weighted average position of the robot's bodies. |
| **centroidal momentum** | The robot's total linear and angular momentum about its center of mass. |

## Robot descriptions

| Term | Meaning |
|---|---|
| **URDF** | An XML robot-description format common in ROS tooling. |
| **MJCF** | MuJoCo's XML model format. |
| **intermediate representation, IR** | The parser-neutral model that URDF, MJCF, and programmatic construction produce before `build_model`. |
| **asset resolver** | An object that maps a mesh or asset URI to a local path. |

## Pinocchio storage names

BetterRobot keeps standard algorithm names but uses descriptive storage
fields:

| Pinocchio | BetterRobot |
|---|---|
| `oMi` | `joint_pose_world` |
| `oMf` | `frame_pose_world` |
| `liMi` | `joint_pose_local` |
| `nle` | `bias_forces` |
| `M` | `mass_matrix` |
| `Ag` | `centroidal_momentum_matrix` |
| `hg` | `centroidal_momentum` |

The full naming rule is in {doc}`/conventions/naming`.
