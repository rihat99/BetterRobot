# 03 State and mutability

The library follows the Pinocchio split: a *model* (static topology plus differentiable parameter tensors) and
a *state* (the result of one query). Both are values, not objects with a lifecycle: you build them, pass them
around, and build new ones instead of changing them. See ADR [0002](decisions/0002-immutable-fk-state.md).

## C-STATE-1 Model and state dataclasses are frozen (MUST)

`ModelStructure`, `ModelValues`, `KinematicState`, every `spatial` value type (`Motion`, `Inertia`,
`Symmetric3`) and the `optim` result types are `@dataclass(frozen=True)` (`optim.Phase` and `optim.Result`,
`optim/phases.py:15,38`, are not yet: `[planned: L-5]`). Derived quantities are computed by
functions or cached properties, never by assignment after construction.

```python { .no-run }
# good  (spatial/motion.py)
@dataclass(frozen=True)
class Motion:
    linear: Tensor      # (..., 3)
    angular: Tensor     # (..., 3)
# bad
state.oMf = new_poses          # mutates a shared value; every holder of `state` changes
```

Enforcement: `tests/contract/test_frozen.py` (imports every public dataclass, asserts
`__dataclass_params__.frozen` and that attribute assignment raises `FrozenInstanceError`).

## C-STATE-2 Queries return new values (MUST)

`forward_kinematics_state(model, q)` returns a new `KinematicState`; `Model.integrate(q, v)` returns a new `q`.
A function never takes a state to fill in. To change a model parameter, build a new `ModelValues` with
`dataclasses.replace` or the builder API.

```python { .no-run }
# good
state = forward_kinematics_state(model, q)
model2 = replace(model, values=replace(model.values, joint_placements=new_X))
# bad
forward_kinematics(model, data, q)      # Pinocchio-style in/out `data` argument
```

Enforcement: `[review]`; `tests/contract/test_frozen.py` covers the result types.

## C-STATE-3 Never mutate an input tensor (MUST)

In-place ops (`x.add_()`, `x[...] = `, `x.copy_()`) on a tensor the caller passed in change the caller's data,
break autograd when the tensor needs gradient, and are rejected by `torch.func.vmap`. Clone first, or build the
output with out-of-place ops. In-place ops on a tensor the function itself created are fine.

```python { .no-run }
# good
q_next = q + dq
# bad
q.add_(dq)                      # caller's `q` silently changes
out = q.clone(); out[..., 3:7] = normalize(q[..., 3:7])     # fine: `out` is ours
```

Enforcement: `tests/contract/test_no_inplace.py` (calls every public entry point and asserts each input
tensor's `_version` counter is unchanged).

## C-STATE-4 Caches are explicit and keyed (SHOULD)

If a function caches (for example a joint-ordering table), the cache lives in a named attribute or a
`functools.lru_cache` keyed on immutable inputs, never in a module-level mutable or a hidden attribute set on
the user's object. A cache never changes a result, only its cost. Documented exception: `KinematicState._memo`
(`kinematics/state.py:111`) is a private memo dict inside the frozen state. Its allowed keys, each derived only
from the state's own `joint_transform`/`frame_transform` and the model structure, are:

| key | filled by | value |
|-----|-----------|-------|
| `"joint_pose"` | `KinematicState.joint_pose` | `(B..., njoints, 7)` world poses |
| `"frame_pose"` | `KinematicState.frame_pose` | `(B..., nframes, 7)` world poses |
| `"jacobian_columns"` | Jacobian helpers in `kinematics/state.py` | world Jacobian column table |
| `"world_geometry"` | `dynamics/state.py` (`WorldGeometry`) | per-joint world rotations, translations, columns |

A new key is added to this table in the same change that writes it. The model-side caches
(`ModelStructure._fk_lean_plans`, `ModelValues._fk_placement_matrices`, `ModelValues._spatial_inertias_cache`)
are invalidated by the tensors' `_version` and are bypassed when the tensors require grad, are `torch.func`
wrapped, are inference tensors, or are being traced by `torch.compile`.

Enforcement: `[review]`.

## C-STATE-5 A `Problem` is built once; `with_` returns a view (MUST)

`optim.Problem` is assembled by its `add`/`cached` calls and then treated as fixed: solvers never write to it
(`Phase.on_start` writing `problem.data` is ledger item L-5). Per-call changes go through
`Problem.with_(weights=..., free=...)` (`optim/problem.py:226`), which returns a new view sharing the terms and
leaves the original untouched; the view carries the overrides, so two solves with different weights can run from
one problem.

Enforcement: `tests/optim/test_problem.py`; `[review]`.

## C-STATE-6 `Model.to` and the dynamics caches (MUST)

`Model.to(device, dtype)` (`data_model/model.py:298`) returns a new `Model` with new `ModelValues`; it never moves
tensors in place. Dynamics caches hang off the object they derive from: the packed spatial inertias on the
`ModelValues` keyed by the tensor `_version` (`dynamics/_world.py:170`), and world geometry on the
`KinematicState._memo` (`dynamics/state.py:39`). A moved model therefore starts with empty caches, and a state made
from the old model is not valid for the moved one; recompute it with `forward_kinematics_state`.

Enforcement: `tests/contract/test_dtype_device.py`; `[review]`.
