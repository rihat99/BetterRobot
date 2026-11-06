# 0013 Joint kinds are final classes under an abstract `JointModel`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`JointModel` is a `runtime_checkable` Protocol with a mixin (`joint_models/revolute.py::_Revolute`) and an overridable
`kind` field, so `JointRX(kind="spherical")` is accepted. `JointRX/RY/RZ/PX/PY/PZ` duplicate the packed axis table.
`integrate` and `difference` exist twice, kept equal by a 300-line test
(`test_integrate_vectorized.py`). `JointComposite` has no caller and no parity test;
`JointHelical`, `JointPlanar` and `JointTranslation` have Pinocchio parity and no consumer. The BetterHuman studio
reads `.kind` strings and keeps its own `_REVOLUTE_AXIS`.

## Decision
We will make `JointModel` an abstract base with final kinds `JointFixed`, `JointRevolute(axis)`,
`JointRevoluteUnbounded`, `JointPrismatic(axis)`, `JointHelical`, `JointSpherical`, `JointPlanar`, `JointTranslation`
and `JointFreeFlyer`, each declaring `nq`, `nv`, `axis` and `manifold`, with `integrate` and `difference` written once
per manifold and called as `kinematics.integrate(model, q, v)` and `kinematics.difference(model, q0, q1)` (no `Model`
method); `JointComposite`, `JointUniverse` (joint 0 becomes a `JointFixed` named "universe") and `JointKind` go.

## Example
```python { .no-run }
@final
@dataclass(frozen=True)
class JointRevolute(JointModel):
    axis: tuple[float, float, float]
    nq: ClassVar[int] = 1
    nv: ClassVar[int] = 1
    manifold: ClassVar[Manifold] = EUCLIDEAN

builder.add_revolute("elbow", parent="upper_arm", child="forearm", axis=(0, 1, 0), lower=-2.0, upper=2.0)
model.joints[3].axis        # the studio reads the axis as data, not a kind string
q_next = br.kinematics.integrate(model, q, v)   # was model.integrate(q, v)
```

## Consequences
Manifold math exists once; the kind and axis are data. Joint hooks nothing calls (`joint_bias_acceleration`,
`joint_motion_subspace_derivative`) go, and so does the planned `velocity(q, v)` method: the child's twist is
`motion_subspace(q) @ v` by definition, and the subspace is constant in the child frame (the plans read it once at
`neutral()`; `check_joint_model` asserts it for every drawn `q`). The oracle-tested kinds that no consumer uses stay
because their parity tests cost nothing; the composite goes.

## Alternatives
Keep the axis subclasses: no gain, the kind/axis split stays. Ball joints as three revolute joints: gimbal lock, and
every SMPL golden breaks (ADR 0005 already rejects a rotation-vector ball joint, singular at pi). Drake's
`RevoluteJoint` and MuJoCo's hinge take an axis argument; Pinocchio's `JointModelRX/RY/RZ` are C++ template
specialisations with no use in a batched PyTorch library.

## Compatibility
Breaks the BetterHuman studio (1 file), `viewer/layers/skeleton.py::_MARKER`, BetterRetarget's `joint_models` (2
sites) and `Model.integrate`/`Model.difference` (BetterHuman 6 lines, each app 1). BetterRobot ADR 0005 keeps its rule
(`nq` and `nv` distinct, these two the only maps) under the function spelling and becomes Superseded by this record.
Adding a kind later is compatible; removing one is not.

## Enforcement
`tests/parity/pinocchio/test_rnea_missing_joint_kinds.py` keeps the helical, planar and translation parity;
`tests/model/test_joints.py` runs `better_robot.testing.check_joint_model` on every kind; the 300-line duplicate-math
test is deleted with the duplication; new `tests/contract/test_abstract_or_final.py` (every class in `model.joints` is
abstract or `@final`).

## Open questions
None.
