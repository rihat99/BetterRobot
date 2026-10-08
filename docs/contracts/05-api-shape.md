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

Enforcement: `[review]`.

## C-API-2 Arguments after the second are keyword-only (MUST)

A public callable takes at most two positional arguments (typically the model/state and the main tensor).
Everything else sits after `*`. This lets us add parameters without breaking calls and makes call sites
readable without the signature.

```python { .no-run }
# good
def solve_ik(model, target_pose, *, frame, q0=None, max_iters=50): ...
# bad
def solve_ik(model, target_pose, frame, q0=None, max_iters=50): ...
```

Enforcement: `tests/contract/test_keyword_only.py` (inspects every `__all__` callable; an explicit allowlist
of current violations, to be emptied by the polish phase, lives in the test).

## C-API-3 No boolean positional arguments (MUST)

`f(x, True)` is unreadable. A boolean parameter is keyword-only and named for the true case (`normalize=True`,
not `skip_normalize=False`).

Enforcement: `[ruff:FBT001,FBT002]`.

## C-API-4 Frames, joints and bodies are named by `int | str` everywhere (MUST)

Any parameter that identifies a frame, joint or body accepts the index *or* the name `[planned: L-2]`.
Today `Model.frame_id(name)` (`data_model/model.py:195`) takes a `str` only and
`KinematicState.frame_pose_of(frame_id)` (`kinematics/state.py:146`) an `int` only; the ledger entry widens
`frame_id` to `int | str` and routes every public id parameter through it. `terms.frame_pose`
(`terms/robot.py:62`) already accepts `int | str`.

```python { .no-run }
# good  (terms/robot.py:62)
frame_pose(model, frame="panda_hand", target="goal"); frame_pose(model, frame=8, target="goal")
# bad  (kinematics/state.py:146, until L-2 lands)
def frame_pose_of(self, frame_id: int): ...   # forces users to look up ids themselves
```

Enforcement: `tests/contract/test_public_api.py`; `[review]`.

## C-API-5 Small top-level surface; specialised APIs stay package-qualified (SHOULD)

`better_robot.__all__` exports the model, FK, Jacobians, IK and the viewer entry point. Everything else is
reached as `better_robot.<package>.<name>`. A symbol enters `__all__` only when two examples use it.

Enforcement: `tests/contract/test_public_api.py`, `tests/contract/test_submodule_public_imports.py`.
