# 0018 A body's frame carries exactly the body's name

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
The URDF link `panda_hand` becomes the frame `body_panda_hand`, a prefix the source never wrote. The prefix appears in
the README and docs examples, in BetterRetarget (31 frame-name lines in 7 files) and in
the Pinocchio parity conftest (`frame_map`); BetterHuman code has no `body_` string. Principle 3.6 says names stay
exactly as the source gives them.

## Decision
We will name the one frame of each body exactly after the body, keep user frames under their own names, and raise
`ModelError` at build when a body frame and a user frame share a name.

## Example
```python { .no-run }
model = br.load("panda.urdf")
model.frame_id("panda_hand")        # was model.frame_id("body_panda_hand")
state.frame_pose_of("panda_hand")
builder.add_frame("panda_hand", parent="hand", placement=tip)   # build raises ModelError: clashes with the body frame
```

## Consequences
A URDF reader finds the link name in the model. A user cannot reuse a body's name for an extra frame; the build error
names both. Every `"body_..."` string in docs, examples and apps changes once.

## Alternatives
Keep the prefix and state the rule on the README's first screen: the rule stays a surprise at every call site.
Pinocchio's operational frame for a URDF link carries the link name, Drake's body frame carries the body name, and
MuJoCo uses names as given.

## Compatibility
Breaks every `"body_<name>"` string. Migration: README and docs examples, BetterRetarget's 31 lines,
the Pinocchio parity conftest. Reversing it is another global string change.

## Enforcement
New `tests/io/test_frame_names.py` (every body frame equals its body name; a clash raises `ModelError`); the docs
snippet tests in `tests/docs/` fail on a stale `body_` string in a runnable example.

## Open questions
None.
