---
name: add-joint-kind
description: Add a built-in joint kind to BetterRobot, a JointModel subclass under src/better_robot/model/joints/. Use when a task adds a joint type to the library itself; a kind that lives in user code needs only docs/guides/extend.md.
---

# Add a built-in joint kind

A joint kind is a final frozen dataclass under the abstract `JointModel` (record 0013, rule C-EXT-2 in
`docs/contracts/08-extension-points.md`). Its members and the `check_joint_model` contract are in
`docs/guides/extend.md`, section "A joint kind"; `src/better_robot/model/joints/helical.py` is a complete example.

1. Write `src/better_robot/model/joints/<kind>.py`: `@final @dataclass(frozen=True) class Joint<Kind>(JointModel)`
   with `nq`, `nv`, `manifold` (one of `src/better_robot/model/manifolds.py`, or a new one there) and `axis`, and
   `placement`, `motion_subspace`, `neutral`, `random` with their shapes in the docstrings. Export it from
   `src/better_robot/model/joints/__init__.py` and `src/better_robot/model/__init__.py`.
2. Forward kinematics needs no branch: the kernel evaluates a kind it does not know through its own `placement`.
   `src/better_robot/kinematics/_plans.py` reads a kind's class only for the affine fast path (revolute, unbounded
   revolute, prismatic, and the quaternion kinds of `_QUATERNION`, which `check_quaternion_norm=True` covers).
3. Construction: `build_model` needs no entry (it reads the shape of the limits from the kind's `manifold`), and
   `ModelBuilder.add_joint(kind=...)` takes any kind. Add a `ModelBuilder.add_<kind>` method in
   `src/better_robot/io/builder.py` when the kind is common; the URDF or MJCF type that maps to it in
   `src/better_robot/io/parser_urdf.py` or `src/better_robot/io/parser_mjcf.py`; the kind in
   `_SUPPORTED_MIMIC_JOINTS` of `src/better_robot/io/model_from_ir.py` when it may be a mimic target.
4. Viewer: its word in `_KIND_WORD` of `src/better_robot/viewer/robot.py` and its marker in `_MARKER` of
   `src/better_robot/viewer/layers/skeleton.py`; without them the skeleton draws it as a sphere.
5. Tests: the kind in `KINDS` of `tests/model/test_joints.py` (runs `check_joint_model` in float32 and float64),
   in the per-kind cases of `tests/contract/test_coupling_standard.py` and `tests/io/test_build_model.py`, in
   `tests/io/test_builder_kinds.py` for a builder method, and a Pinocchio parity test under `tests/parity/pinocchio/`
   when Pinocchio has the joint.
6. Public surface, in the same change: a builder in `tests/contract/test_repr.py`, every new word in
   `docs/reference/glossary.md`, the `q` layout line appended to `docs/contracts/coupling-standard.md` and
   `tests/contract/coupling_standard_lines.txt`, its row in `docs/concepts/joints-bodies-frames.md`, an `Added`
   entry in `docs/reference/changelog.md`, then `uv run python tools/api_snapshot.py --write`.
7. Check: the fast tier, then
   `uv run pytest tests/model tests/io tests/kinematics tests/dynamics tests/viewer tests/parity -q`.
