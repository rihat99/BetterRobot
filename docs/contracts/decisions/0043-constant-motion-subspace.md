# 0043 A joint's motion subspace is constant in its child frame; no `velocity` method

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record [0013](0013-joint-kinds.md) defines the joint kinds under an abstract `JointModel` and drops the hooks nothing
calls. Its plan still listed a `velocity(q, v)` method, and the kinematic plans read each joint's motion subspace
once, at `neutral()`, which is right only when the subspace does not depend on `q`. Nothing stated or checked that
assumption, so a new joint kind with a configuration-dependent subspace would have given wrong Jacobians silently.

## Decision
We will state that `JointModel.motion_subspace(q)` is constant in the child frame, have
`better_robot.testing.check_joint_model` assert `motion_subspace(q) == motion_subspace(neutral())` for every
drawn `q`, and leave `velocity(q, v)` out of `JointModel`, since the child's twist is `motion_subspace(q) @ v` by
definition.

## Example
```python { .no-run }
from better_robot.testing import check_joint_model

check_joint_model(JointRevolute(axis=(0.0, 1.0, 0.0)), dtype=torch.float64, generator=generator)
twist = joint.motion_subspace(q) @ v   # the child's twist; no joint.velocity(q, v)
```

## Consequences
One fewer method on every joint kind, and the plans may keep reading the subspace at `neutral()`. Cost: a joint
whose subspace depends on `q` (a curvilinear or coupled joint) cannot be added without changing the plans and this
record.

## Alternatives
Keep `velocity(q, v)` as Pinocchio's `JointModel::calc` does: a second way to compute `motion_subspace(q) @ v`.
Read the subspace at each `q` in the plans: a per-call cost for every joint kind the library has, none of which
needs it.

## Compatibility
No public name changes; `velocity` never shipped. A user-defined joint kind with a `q`-dependent subspace fails
`check_joint_model`.

## Enforcement
C-EXT-2; `tests/model/test_joints.py` (every joint kind through `check_joint_model`).

## Open questions
None.
