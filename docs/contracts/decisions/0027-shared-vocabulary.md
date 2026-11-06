# 0027 One glossary for `KinematicState` and `HumanState`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`KinematicState` has `joint_pose`/`joint_poses`, `joint_translation`/`joints`, `frame_pose`/`frame_poses`, and a
`fk`/`forward` pair. BetterHuman mirrors BetterRobot's plural fields on `HumanState` but keeps `rest_joints
(I..., J, 3)` on `Identity`, uses `values` (plural) and `stream` (singular) for two mappings keyed by group, and says
"affine" (`skin_affines`, `inverse_bind_affines`) for the `(..., 3, 4)` shape that BetterRobot calls "transform".

## Decision
We will keep one glossary in BetterRobot (`docs/reference/glossary.md`) that BetterHuman extends, with the rules plural
for an axis over named things, `_ids` for integer indices (0023), "transform" for a rigid `[R | p]` and "affine" for
a `(..., 3, 4)` that may scale (MHR), and BetterHuman's `forward` (kinematics plus skinning) distinct from
BetterRobot's `forward_kinematics`; and BetterHuman renames `rest_joints` to `rest_joint_translations`, the input
`stream` to `streams`, `pelvis` to `root_offset` and `params` to `values`.

## Example
```python { .no-run }
state = identity.forward(q, streams={"expression": expression})   # BetterHuman: kinematics plus skinning
state.joint_poses, state.joint_translations, state.frame_poses     # same plural words as KinematicState
identity.rest_joint_translations, identity.root_offset, identity.values
br.forward_kinematics(model, q)                                    # BetterRobot: kinematics only
```

## Consequences
A reader moving between the repositories meets one word for one thing. The cost is renames in BetterHuman (four
names, plus the callers) after BetterRobot's renames land, and a glossary page that both repositories keep true.

## Alternatives
BetterHuman's names as written in its design: two vocabularies for the same shapes. Drake's multibody notation page
defines the vocabulary once for the whole library, which is the model here.

## Compatibility
Breaks the four BetterHuman names and their uses in docs, examples and the apps; lands after 0015. Reversible only by
another rename.

## Enforcement
`tests/contract/test_naming.py` (BetterRobot, C-NAME-2) and `BetterHuman/tests/contract/test_naming.py` over the
`HumanState`, `Identity` and `HumanModel` field names; `tests/contract/test_vocabulary.py::test_glossary_words` in each
repository checks every word of every public name against the glossary.

## Open questions
None.
