# 03 State and mutability

The library follows the Pinocchio split: a *model* (static topology plus differentiable parameter tensors) and
a *state* (the result of one query). Both are values, not objects with a lifecycle: you build them, pass them
around, and build new ones instead of changing them. See ADR [0002](decisions/0002-immutable-fk-state.md).

## C-STATE-1 Model and state dataclasses are frozen (MUST)

`Model`, `KinematicState` and the `optim` result types (`optim.Phase`, `optim.Result`) are
`@dataclass(frozen=True)`. Derived quantities are computed by
functions or cached properties, never by assignment after construction. The tensor fields are read-only after
construction too (the freeze is shallow): an in-place write and `requires_grad_()` on a model tensor are edits a
user must not make; build a new model with `with_values` (record [0040](decisions/0040-model-value-memo.md)).

```python { .no-run }
# good  (src/better_robot/kinematics/forward.py)
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

Enforcement: `tests/contract/test_no_inplace.py` (calls every public kinematics, dynamics, lie and spatial
function with a tensor input and asserts each input tensor's `_version` counter is unchanged).

## C-STATE-4 Caches are explicit and keyed (SHOULD)

If a function caches (for example a joint-ordering table), the cache lives in a named attribute or a
`functools.lru_cache` keyed on immutable inputs, never in a module-level mutable or a hidden attribute set on
the user's object. A cache never changes a result, only its cost. Documented exception: `KinematicState._memo`
(`src/better_robot/kinematics/forward.py::KinematicState`) is a private memo dict inside the frozen
state, keyed by `(name, grad mode)`, written only by `KinematicState._memoised` and bypassed under
`torch.inference_mode` and `torch.compile` like `Model._value`, so a value made under `torch.no_grad` never
serves a differentiable call. Its allowed names, each derived only from the state's own
`joint_transforms`/`frame_transforms` and the model structure, are:

| name | filled by | value |
|------|-----------|-------|
| `"joint_poses"` | `KinematicState.joint_poses` | `(B..., njoints, 7)` world poses |
| `"frame_poses"` | `KinematicState.frame_poses` | `(B..., nframes, 7)` world poses |
| `"jacobian_columns"` | `KinematicState.jacobian_columns` | `(B..., 6, nv_full)` world Jacobian column table |
| `"world_geometry"` | `KinematicState.world_geometry` (record 0041) | `(N, 6, nv_full)` columns, joint rotations, translations and origin about the first joint |

A new name is added to this table in the same change that writes it. The model-side cache is `Model._plans`:
constant tensors derived from the static fields, keyed by kind, dtype and device, and shared by every
`with_values` and `to` copy of a model (`Model._plan_owner`; a copy keeps every static field, and keeps the first
model of its line alive, with that model's tensors and value memo). `Model._memo` holds
values derived from the tensor fields (the placement matrices and the packed spatial inertias), keyed by kind,
grad mode and the tensor fields' `requires_grad` flags, written only when the value carries no autograd graph and
bypassed under `torch.inference_mode`, so nothing on the model is keyed by a tensor's `_version` (decision
records 0011, 0040 and 0042).

Enforcement: `tests/contract/test_caches.py`; `[review]`.

## C-STATE-5 A `Problem` only gains names; `with_` returns a new, sealed problem (MUST)

`optim.Problem` is built by its constructor and its registration calls (`residual`, `scalar`, `cached`), which only
add new names; every field of a registered term is fixed, and solvers never write to the problem. `Phase.on_start`
returns `(x, data)` instead of writing into the problem's data. Per-call changes go through
`Problem.with_(data=..., weights=..., free=...)` (`src/better_robot/optim/problem.py::Problem.with_`), which returns a
new problem sharing the terms and leaves the original untouched, so two solves with different weights can run from
one problem. That new problem is sealed, as are the views `run`, `solve_constrained` and `solve_layer` hand to
callbacks: registering on a sealed problem raises `ArgumentError` (record
[0047](decisions/0047-problem-registration-and-sealed-views.md)).

Enforcement: `tests/optim/test_problem.py::test_problem_is_immutable`,
`tests/optim/test_problem.py::test_callbacks_cannot_register_on_the_views_run_hands_them`.

## C-STATE-6 `Model.to` and the dynamics caches (MUST)

`Model.to(device, dtype)` returns a new `Model` with every tensor field moved; it never moves tensors in
place. Caches hang off the object they derive from: the plans in `Model._plans` and the world geometry in
`KinematicState._memo`. A moved model therefore starts with an empty value memo and shares only the plans, which
depend on the static fields, dtype and device alone; a state made from the old model is not valid for the moved
one; recompute it with `forward_kinematics`. Every function taking `(model, state)` calls
`check_state` (`src/better_robot/kinematics/forward.py::check_state`) once: a state in another dtype or
on another device raises `DtypeMismatchError` or `DeviceMismatchError`, a state of another structure `ModelError`;
a copy of the same robot (`with_values`) is accepted.

Enforcement: `tests/contract/test_caches.py`, `tests/contract/test_dtype_device.py`,
`tests/contract/test_state_attachment.py`; `[review]`.
