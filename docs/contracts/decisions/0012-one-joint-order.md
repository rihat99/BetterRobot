# 0012 One joint order: stable source order

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`io/build_model.py::build_model(preserve_joint_order=False)` chooses between a historical depth-first order (the
default) and a stable Kahn sort. For the Panda, source order, depth-first and Kahn agree. For the G1, source order
equals Kahn and depth-first differs from joint 20 on. BetterHuman passes `preserve_joint_order=True` at three sites
(`core/rig.py`, `families/mhr_rig.py`, `families/soma_rig.py`) and hard-codes ids (`RegressedRig` uses
`arange(1, J + 1)`, `SomaRig` uses `arange(J)`, `mhr_rig::_joint_tree` checks `idx_qs[1] == 0`).

## Decision
We will order joints with one stable source-order sort (Kahn: parents first, ties keep file order) and remove
`preserve_joint_order`.

## Example
```python { .no-run }
def _order(ir: IRModel, root: IRJoint) -> list[IRJoint]:
    """Parents first; among joints whose parent is placed, the one listed first in the file comes first."""

model = br.load("g1.urdf")
model.joint_names           # the file's joint order, parents first
# Pinocchio parity maps joints by name (q_permutation), never by position
```

## Consequences
BetterHuman is unchanged and its hard-coded ids stay valid. The G1 `q` layout differs from today's and from
Pinocchio's, so any code that indexes G1 `q` by position changes; a name lookup does not.

## Alternatives
Depth-first only, as Pinocchio's URDF loader does: SMPL's kinematic-tree order is not depth-first, so every
BetterHuman rig, its joint ids and the joint order of every golden change. Both orders behind a flag (today): two
layouts for one robot. MuJoCo numbers bodies in the order the file lists them, which is the source-order choice.

## Compatibility
Breaks G1 index-based code: about 12 G1 parity cases map joints by name through `q_permutation`;
`tests/kinematics/fk_reference.npz` is regenerated; BetterRetarget's `g1.to_q` is rebuilt from joint names. Not
reversible once the apps depend on the order.

## Enforcement
The Pinocchio parity suite `tests/parity/pinocchio/` (184 cases) maps joints by name; new
`tests/io/test_joint_order.py` checks the Panda and the G1 against the file order; `tests/contract/test_keyword_only.py`
guards the removal of the flag from the signature.

## Open questions
None.
