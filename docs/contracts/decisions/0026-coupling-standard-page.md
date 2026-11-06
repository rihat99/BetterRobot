# 0026 One coupling-standard page in BetterRobot, tested by every consumer

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
No page lists what BetterHuman, BetterVideoReconstruction and BetterRetarget rely on in BetterRobot. A first draft
misses three things BetterHuman depends on: model values with leading batch axes (0028), the Euler order (`so3.from_euler`
and `to_euler` compose `Rz Ry Rx`; the MHR rig and the studio rely on it), and joint 0 (SOMA maps its native joint 0
onto BetterRobot joint 0: `SomaRig.native_joint_ids` returns `arange(J)`), while `JointUniverse` is deleted (0013).

## Decision
We will write one page in BetterRobot `docs/contracts/` that changes only by addition or by a decision record, give
BetterHuman a section for `HumanState`, and have BetterHuman, BetterVideoReconstruction and BetterRetarget each run a
coupling test against it.

## Example
```text
q          (B..., nq)   revolute, prismatic, helical 1 | continuous [cos, sin] 2 | spherical [qx,qy,qz,qw] 4
                        planar [x, y, cos, sin] 4 | translation 3 | free flyer [tx,ty,tz,qx,qy,qz,qw] 7 | fixed 0
v          (B..., nv)   tangent, never added to q
pose (..., 7) [tx,ty,tz,qx,qy,qz,qw]   transform (..., 3, 4) [R | p]   twist, wrench (..., 6) linear first
inertia    (..., 10)    [m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz] about the centre of mass
jacobian   (B..., 6, nv) default local_world_aligned     euler Rz Ry Rx
joints     source order (0012); joint 0 is a fixed "universe" with the identity transform; joint 1 is the root
mimic joints appear only in the full layout (*_full)     identifiers (0029)     files (0030)     batch axes (0028)
```

## Consequences
A change to a layout is visible as a diff to one page and a failing consumer test, before an app breaks. Additions are
free; a removal needs a record (11.6). The page holds no function names (principle 1.3): the verbs can be renamed
without touching it.

## Alternatives
Prose in each repository's agent file, as today: three copies that drift. Drake publishes a stability
guarantee per kind of artifact, which is the model for the "additions only" rule.

## Compatibility
Adds a page and tests. It freezes the listed layouts: changing one later is a breaking change by definition and needs
a record that shows the app migration.

## Enforcement
New BetterRobot `tests/contract/test_coupling_standard.py` (the page's shapes against the code); a coupling test in
each consumer: `BetterHuman/tests/contract/test_coupling_standard.py`, and one each in BetterVideoReconstruction and
BetterRetarget, run in their fast tiers.

## Open questions
None.
