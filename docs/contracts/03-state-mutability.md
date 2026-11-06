# 03 State and mutability

The library follows the Pinocchio split: a *model* (static topology plus differentiable parameter tensors) and
a *state* (the result of one query). Both are values, not objects with a lifecycle: you build them, pass them
around, and build new ones instead of changing them. See ADR [0002](decisions/0002-immutable-fk-state.md).

## C-STATE-1 Model and state dataclasses are frozen (MUST)

`Model`, `KinematicState` and the `optim` result types are `@dataclass(frozen=True)` (`optim.Phase` and `optim.Result`,
`optim/phases.py:15,38`, are not yet: `[planned: L-5]`). Derived quantities are computed by
functions or cached properties, never by assignment after construction. The tensor fields are read-only after
construction too (the freeze is shallow): an in-place write and `requires_grad_()` on a model tensor are edits a
user must not make; build a new model with `with_values` (record [0040](decisions/0040-model-value-memo.md)).

```python { .no-run }
# good  (src/better_robot/kinematics/forward_kinematics.py)
@dataclass(frozen=True)
class KinematicState:
    q: Tensor                  # (B..., nq)
    joint_transforms: Tensor   # (B..., njoints, 3, 4)
# bad
state.oMf = new_poses          # mutates a shared value; every holder of `state` changes
```

Enforcement: `tests/contract/test_frozen.py` (imports every public dataclass, asserts
`__dataclass_params__.frozen` and that attribute assignment raises `FrozenInstanceError`).

## C-STATE-2 Queries return new values (MUST)

`forward_kinematics(model, q)` returns a new `KinematicState`; `kinematics.integrate(model, q, v)` returns a new `q`.
A function never takes a state to fill in. To change a model parameter, build a new `Model` with
`Model.with_values` or the builder API.

```python { .no-run }
# good
state = forward_kinematics(model, q)
model2 = model.with_values(joint_placements=new_X)
# bad
compute_forward_kinematics(model, data, q)      # Pinocchio-style in/out `data` argument
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
(`src/better_robot/kinematics/forward_kinematics.py::KinematicState`) is a private memo dict inside the frozen
state, keyed by `(name, grad mode)` and written only by `KinematicState._memoised`, so a value made under
`torch.no_grad` never serves a differentiable call. Its allowed names, each derived only from the state's own
`joint_transforms`/`frame_transforms` and the model structure, are:

| name | filled by | value |
|------|-----------|-------|
| `"joint_poses"` | `KinematicState.joint_poses` | `(B..., njoints, 7)` world poses |
| `"frame_poses"` | `KinematicState.frame_poses` | `(B..., nframes, 7)` world poses |
| `"jacobian_columns"` | `KinematicState.jacobian_columns` | `(B..., 6, nv_full)` world Jacobian column table |
| `"world_geometry"` | `src/better_robot/dynamics/state.py` (`WorldGeometry`) | per-joint world rotations, translations, columns |

A new name is added to this table in the same change that writes it. The model-side cache is `Model._plans`:
constant tensors derived from the static fields, keyed by kind, dtype and device. `Model._memo` holds values
derived from the tensor fields (the placement matrices and the packed spatial inertias), keyed by kind, grad
mode and the tensor fields' `requires_grad` flags, written only when the value carries no autograd graph and
bypassed under `torch.inference_mode`, so nothing on the model is keyed by a tensor's `_version` (decision
records 0011 and 0040).

Enforcement: `tests/contract/test_caches.py`; `[review]`.

## C-STATE-5 A `Problem` is built once; `with_` returns a view (MUST)

`optim.Problem` is assembled by its `add`/`cached` calls and then treated as fixed: solvers never write to it
(`Phase.on_start` writing `problem.data` is ledger item L-5). Per-call changes go through
`Problem.with_(weights=..., free=...)` (`optim/problem.py:226`), which returns a new view sharing the terms and
leaves the original untouched; the view carries the overrides, so two solves with different weights can run from
one problem.

Enforcement: `tests/optim/test_problem.py`; `[review]`.

## C-STATE-6 `Model.to` and the dynamics caches (MUST)

`Model.to(device, dtype)` returns a new `Model` with every tensor field moved; it never moves tensors in
place. Caches hang off the object they derive from: the plans in `Model._plans` and the world geometry in
`KinematicState._memo`. A moved model therefore starts with empty caches, and a state made from the old model is
not valid for the moved one; recompute it with `forward_kinematics`.

Enforcement: `tests/contract/test_caches.py`, `tests/contract/test_dtype_device.py`; `[review]`.
