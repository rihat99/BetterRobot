# 0023 Naming sweep: full words, plural axes, `_ids` for integer indices

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
Principles 3.1 to 3.5 and 3.8 are settled; their application is not. `KinematicState` has `joint_pose` and
`joint_poses` side by side, `frame_parent_joint_ids` is a tuple of integers, `joints` would hold joint objects, and
`mimic_source`, `mimic_multiplier` and `mimic_offset` are singular although their axis runs over joints.
`forward_kinematics_state`, `Model.fk`, `fext`, `lower_pos_limit`, `get_subtree` and the package name `data_model`
break the grammar. Call sites: BetterHuman `forward_kinematics_state` 8 source, 6 test, 3 example lines; state fields
BetterHuman 10, BetterRetarget 10 plus the `metrics.py::_subset` constructor call, BetterVideoReconstruction 3.

## Decision
We will rename `forward_kinematics_state` to `forward_kinematics`, remove `Model.fk`, make `KinematicState` fields
plural (`joint_transforms`, `frame_poses`, `joint_rotations`, ...), add `joint_pose_of` beside `frame_pose_of`, rename
`structure` to `model`, `lower/upper_pos_limit` to `lower/upper_position_limit`, `joint_models` to `joints`,
`get_subtree/get_support` and `expand_configuration` to the `kinematics` functions `subtree(model, joint)`,
`support(model, joint)` and `full_configuration(model, q)`, `fext` to `external_forces`, the package `data_model` to `model`, and give modules the name of their job; and we will end every
integer index field in `_ids` (`frame_parent_joint_ids`, `mimic_source_ids`, `part_body_ids`), keep bare plurals for
objects or values (`joints`, `joint_placements`) and make per-joint fields plural (`mimic_multipliers`, `mimic_offsets`).

## Example
```python { .no-run }
state = br.forward_kinematics(model, q)                  # was br.forward_kinematics_state(model, q); model.fk(q) is gone
state.joint_transforms, state.frame_poses, state.joint_pose_of("elbow")
model.frame_parent_joint_ids, model.mimic_source_ids, model.mimic_multipliers
br.kinematics.subtree(model, "elbow")                  # was model.get_subtree("elbow"); Model stays a record
```

## Consequences
A name tells its type: `_ids` is integers, a bare plural is objects or values. BetterHuman's names agree with
BetterRobot's after its renames land (0027). Every doc, example and app line with an old name changes once.

## Alternatives
The renames alone, without the `_ids` and per-joint plural rule: `frame_parent_joints` reads as objects but holds
integers, and the two repositories keep different conventions. Pinocchio's `joints` and `parents` are plain plurals
and its ids are untyped integers; Drake suffixes ids with a type (`BodyIndex`); the `_ids` rule is the tensor-library
equivalent.

## Compatibility
Breaks every renamed public name; no aliases (3.4). Migration is mechanical by search and replace; the BetterRobot
changelog lists each as Changed with its replacement. Not reversible without another break.

## Enforcement
`tests/contract/test_naming.py` (C-NAME-2) extended: an integer field named without `_ids`, or a `_ids` field that is
not integer, fails; `tests/contract/test_se3_vocabulary.py` (C-NAME-5); the glossary page (0027).

## Open questions
None.
