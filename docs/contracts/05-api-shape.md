# 05 API shape

Math is functions; data is frozen dataclasses; classes exist only for resources with a lifecycle (a viewer
server, a file loader with an open handle). Keeping these three kinds apart is what lets the library grow by
adding functions rather than by extending class hierarchies.

## C-API-1 Functions for math, frozen dataclasses for data, classes only for lifecycles (MUST)

A new computation is a module-level function taking a model/state and tensors. A new bundle of tensors is a
`@dataclass(frozen=True)`. A class with methods is justified only when it owns a resource (`Viewer` owns a
viser server) or an evolving run (`optim.run` phases). Methods on data classes are thin conveniences that
delegate to functions.

```python { .no-run }
# good
def frame_jacobian(state: KinematicState, frame: int | str, *, reference=LOCAL_WORLD_ALIGNED) -> Tensor: ...
# bad
class Kinematics:                         # a class with no resource, only to group functions
    def __init__(self, model): ...
    def jacobian(self, frame): ...
```

Enforcement: `tests/contract/test_frozen.py` (every public dataclass is frozen); `[review]`.

## C-API-2 Arguments after the second are keyword-only (MUST)

A public callable takes at most two positional arguments (typically the model/state and the main tensor).
Everything else sits after `*`. This lets us add parameters without breaking calls and makes call sites
readable without the signature.

```python { .no-run }
# good
def solve_ik(model, targets, *, q0, solver=None, iterations=100): ...
# bad
def solve_ik(model, targets, q0, solver=None, iterations=100): ...
```

Enforcement: `tests/contract/test_keyword_only.py` (inspects every `__all__` callable). Its permanent list holds
`kinematics.integrate(model, q, v)` and `kinematics.difference(model, q0, q1)`, whose third argument is the
operand of `q (+) v` and `q1 (-) q0` (record [0013](decisions/0013-joint-kinds.md)); the shrink-only ratchet
`tests/contract/ratchets/keyword_only.json` holds `lie.so3.slerp` and `lie.se3.sclerp`, whose `t` is positional
today.

## C-API-3 No boolean positional arguments (MUST)

`f(x, True)` is unreadable. A boolean parameter is keyword-only and named for the true case (`normalize=True`,
not `skip_normalize=False`).

Enforcement: `[ruff:FBT001,FBT002]`.

## C-API-4 Frames, joints and bodies are named by `int | str` everywhere (MUST)

Any public function parameter that identifies a frame, joint or body (`frame`, `joint`, `body` and their plurals)
accepts the id *or* the name; `Model.frame_id`, `Model.joint_id` and `Model.body_id` resolve either to the id, and an
unknown name raises `UnknownNameError` listing the valid names. `ModelBuilder` names bodies before any id exists,
so it takes names only; records that hold resolved data (`model.AttachedShape.body`, `viewer.MeshSpec.body`, the
`frame_ids` of `collision.RobotSpheres`) hold ids (record [0029](decisions/0029-frames-and-identifiers.md)).

```python { .no-run }
# good  (src/better_robot/terms/robot.py::frame_pose)
frame_pose(model, frame="panda_hand", target="goal"); frame_pose(model, frame=8, target="goal")
state.frame_pose_of("panda_hand"); state.frame_pose_of(8)
# bad
def frame_pose_of(self, frame_id: int): ...   # forces users to look up ids themselves
```

Enforcement: `tests/contract/test_identifiers.py` (every such parameter in `tests/contract/public_api.json`).

## C-API-5 Small top-level surface; specialised APIs stay package-qualified (SHOULD)

`better_robot.__all__` exports the model, FK, Jacobians, IK and the viewer entry point. Everything else is
reached as `better_robot.<package>.<name>`. A symbol enters `__all__` only when two examples use it.

Enforcement: `tests/contract/test_public_surface.py`, `tests/contract/test_submodule_public_imports.py`.
