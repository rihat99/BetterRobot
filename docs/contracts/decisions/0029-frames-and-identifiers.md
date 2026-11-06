# 0029 Names at every user boundary, ids inside

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
BetterRobot terms take frame names; BetterHuman terms take native indices (`WRIST = 20` in `BetterHuman/examples/03_ik_reach.py`).
`RobotSpheres.from_frames` skips the range check for integers while `RobotCapsules.from_frames` resolves both.
BetterHuman's contact frames live in a JSON file that BetterVideoReconstruction and BetterRetarget read through the
checkout path.

## Decision
We will accept names at every user boundary and use ids inside: `joint_id`, `frame_id` and `body_id` take an int or a
str and range-check ints, BetterHuman's `terms.joints` and `forces_layer` take names, names stay exactly as the source
gives them (0018), and BetterHuman's contact frames become BetterRobot frames added at build from a versioned shipped
frame-set file (BetterHuman 0013).

## Example
```python { .no-run }
model.frame_id("panda_hand")          # str -> int
model.frame_id(7)                     # int, range-checked
terms.joints(human, {"left_wrist": target})   # a name, not WRIST = 20; ids never leave the library
spheres = br.collision.RobotSpheres.from_frames(model, ["panda_hand", 3])   # both forms checked the same way
```

## Consequences
Code survives a joint-order change (0012) because names do not move. Cost: a name lookup per call at the boundary,
done once per call, not per element.

## Alternatives
Ids at user boundaries with name-lookup helpers (BetterHuman today): an id changes meaning with the joint order, and
the studio stored poses against ids. Pinocchio and Drake both address frames and bodies by name at the API (`getFrameId`,
`GetFrameByName`) and by index inside algorithms.

## Compatibility
Breaks BetterHuman `terms.joints` and `forces_layer` argument types and any caller passing ids. Migration: names for
ids at the call sites (BetterVideoReconstruction, BetterRetarget, the examples).

## Enforcement
The coupling test (0026) on `joint_id`, `frame_id` and `body_id`, with an out-of-range int raising;
`tests/collision/test_collision.py::test_from_frames_agree` (`RobotSpheres.from_frames` and
`RobotCapsules.from_frames` resolve names and ids the same way); `BetterHuman/tests/contract/test_public_surface.py`
(no BetterHuman public function takes a joint index).

## Open questions
None.
