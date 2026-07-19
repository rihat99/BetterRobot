# Joints, bodies, and frames

A robot description is a tree. Joints are the edges that permit motion.
Bodies are the rigid objects connected by those edges. Frames are named
coordinate systems attached to a joint or body so that users can ask about a
tool tip, camera, foot, or contact point.

Keeping these roles separate avoids a common ambiguity: a body is physical,
but a frame is only a place from which a pose or velocity is measured.

## Joints describe allowed motion

Every joint implements the same small set of operations:

- turn its configuration slice into a parent-to-child transform;
- describe its motion subspace;
- integrate a local tangent step;
- take the difference between two configurations; and
- produce a neutral or random valid configuration.

The built-in families cover:

| Family | Motion | `nq` / `nv` |
|---|---|---:|
| fixed | none | 0 / 0 |
| revolute | rotation about one axis | 1 / 1 |
| continuous revolute | unbounded rotation stored on a circle | 2 / 1 |
| prismatic | translation along one axis | 1 / 1 |
| helical | coupled rotation and translation | 1 / 1 |
| spherical | arbitrary rotation | 4 / 3 |
| planar | translation in a plane plus yaw | 4 / 3 |
| free-flyer | arbitrary pose | 7 / 6 |
| composite | an ordered product of joints | sum / sum |

Axis-aligned joint classes are convenient names, while unaligned versions
store an arbitrary normalized axis. All of them share the same transform
formulas, so the object API and the compiled dispatch path cannot drift apart.

## Why `nq` can differ from `nv`

`nq` is storage width. `nv` is the number of independent local motion
coordinates. A unit quaternion uses four stored numbers constrained to a
three-dimensional surface. It therefore contributes four configuration
numbers but only three tangent coordinates.

That distinction matters whenever code makes a perturbation or builds a
Jacobian. Columns correspond to `nv`, not necessarily to `nq`. Use
`Model.integrate` and `Model.difference` so each joint applies the correct
geometry.

## Fixed and floating bases share the tree

Joint zero represents the universe and has no degrees of freedom. A
fixed-base robot reaches its first body through a fixed joint. A
floating-base robot reaches it through a free-flyer. All later algorithms see
the same tree and the same joint protocol.

This is why `solve_ik`, FK, and dynamics have no `floating_base` branch. The
model expresses the difference once. See {ref}`decision-unified-base` for
the rejected two-family design.

## Mimic joints and public coordinates

A mimic joint follows another joint through a fixed multiplier and offset.
The public configuration contains only independent coordinates. Static maps
expand them before a tree pass and project Jacobians, forces, and mass
matrices back afterward. Users therefore optimize a reduced state without
manually enforcing the mimic relation.

This reduction is an implementation detail of the model, not a second robot
API. Joint names and frame names still include the physical elements loaded
from the description.

## Bodies carry inertial properties

A `Body` represents a rigid link. Its mass, center of mass, and rotational
inertia live in the model's numerical values. Dynamics uses those values;
kinematics needs only the topology and joint placements.

Body identity is distinct from joint identity. A joint says how two parts may
move relative to one another. A body says what mass moves.

## Frames name useful coordinate systems

A `Frame` is attached to a parent joint with a fixed placement. It may mark a
body origin, a tool center point, a sensor, or any user-defined location.
`model.frame_id(name)` resolves the name once, and frame placement or Jacobian
functions use the integer id in repeated calculations.

Frame Jacobians can be expressed in local, world, or local-world-aligned
coordinates. Those conventions are explained in
{doc}`kinematics_and_jacobians`.

## Custom joint types

A custom joint must honor the same storage, tangent, transform, integration,
and difference contracts. It can use the object fallback without changing
the built-in compiled dispatch. Add one only when the motion cannot be
expressed by an existing family or a composite joint; every new joint expands
the surface that kinematics, dynamics, random configuration, and testing must
cover.

Next read {doc}`lie_and_spatial` for the pose and motion mathematics used by
these joints, or {doc}`/guides/load_a_robot` to build a tree from a file or
from Python.
