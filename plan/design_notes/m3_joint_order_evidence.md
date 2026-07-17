# M3 joint-order decision evidence

The default remains DFS. On the local SMPL-like fixture, DFS changes the
already-topological source prefix
`root, left_hip, right_hip, spine1, left_knee, ...` to
`root, left_hip, left_knee, left_ankle, left_foot, ...`. The opt-in stable
Kahn path preserves all 24 source joint names exactly and produces the same
FK and RNEA results after a synthetic name-keyed coordinate remap.

The existing exact DFS branching-tree test remains green, as do the focused
Panda/G1 FK and Pinocchio RNEA parity tests. Flipping the default would change
their established q/v indexing; T3.5 therefore exposes
`preserve_joint_order=True` without changing default behavior.

