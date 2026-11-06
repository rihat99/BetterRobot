# 0036 MJCF loads through MuJoCo's compiler, or not at all

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`io/parsers/mjcf.py::parse_mjcf` reads the uncompiled `MjSpec`. A jointed body's own `pos` and `quat` are dropped;
`jnt.range` in file units (degrees by default) is taken as radians; `limited="auto"` joints without a range get limits
`[0, 0]`; `euler` and `axisangle` are ignored. On a two-body file, BetterRobot gives both frames the identity pose and
limits `[-90, 90]` and `[0, 0]`, where MuJoCo gives body positions `[0,0,1]` and `[0,1,1]` and a range of `±1.5708` rad, unlimited otherwise. 17 `hasattr`
probes default silently. The README sells MJCF, `load` warns nothing, and no test compares MJCF kinematics with MuJoCo.

## Decision
We will compile MJCF with MuJoCo (`MjModel`), read placements, limits and inertias from the compiled model, add an FK
parity test against MuJoCo, keep `mujoco` an optional extra, and, if the compiled model cannot give back the body tree,
drop MJCF loading entirely (README, public surface, coupling standard), keeping no hand-written parser.

## Example
```python { .no-run }
model = br.load("scene.xml")      # MuJoCo resolves defaults, units, auto limits and orientations; mujoco imported lazily
```
```python { .no-run }
mj_model = mujoco.MjModel.from_xml_path(path); data = mujoco.MjData(mj_model); mujoco.mj_kinematics(mj_model, data)
state = br.forward_kinematics(model, q)       # q: the same configuration, joints matched by name
torch.testing.assert_close(state.joint_translations[..., br_joint_ids, :], torch.as_tensor(data.xpos[mj_body_ids]), atol=1e-6, rtol=0)
```

## Consequences
BetterRobot stops re-implementing MuJoCo's defaults, unit and orientation rules. The first implementation step checks
that the compiled `MjModel` gives parent body, body pose, joint axes, ranges and inertias; if it does not, MJCF is
removed in the same change and a later record may reintroduce it. Cost: the MJCF reader depends on MuJoCo's compiler
and its version.

## Alternatives
Keep `MjSpec`, raise on every unsupported attribute and add the parity test: BetterRobot would own a second copy of
MuJoCo's rules. Drop MJCF now: simplest, but a working compile route may exist. Pinocchio and Drake each parse MJCF by
hand and document the elements they do not support.

## Compatibility
Changes MJCF results from silently wrong to MuJoCo-consistent, or removes `parse_mjcf` and the `.xml` route. Apps use
no MJCF today. Partly reversible: a removed loader can return later as an addition.

## Enforcement
New `tests/parity/mujoco/test_mjcf_kinematics.py` (oracle MuJoCo: body poses and limits against `mj_kinematics` on a
file with body poses, degree ranges, `limited="auto"` and `euler`); `tests/contract/test_optional_imports.py` (C-DEP-2,
mujoco lazy); `tests/contract/test_no_probing.py` (the 4.2 lint for literal-default `getattr` and `hasattr` probes).

## Open questions
None.
