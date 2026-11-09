# 06 Validation and errors

A *public boundary* is a function listed in a package `__all__`. It validates structure (shapes, dtype,
device, model attachment through `check_state` (C-STATE-6), id resolution) once; everything it calls trusts the validated input. Validation
is about structure, not values: no finiteness scans by default. A value check is allowed only behind a keyword
flag that defaults to off, such as `check_quaternion_norm=False` on `forward_kinematics`
(`src/better_robot/kinematics/forward.py::forward_kinematics`), so the hot path pays nothing unless the caller asks.

## C-ERR-1 Validate at public boundaries only (MUST)

Private helpers (`_name`) and the raw tensor passes in `kinematics`/`dynamics` do no checks. Public wrappers
check each tensor input once with the one validator, `better_robot.lie._checks.check_tensor` (`check_tensors` for
several tensors on one dtype and device; `lie`, `spatial` and the model boundaries all call it), align batches
with `better_robot.model._checks`, and nothing else re-checks. The validator accepts float32 and float64 tensors.
Validation must not read tensor values (no `isfinite`, no `.item()`), because that is a host sync (`02`) and
breaks `vmap` (`04`).

```python { .no-run }
# good  (illustrative; real entry point: src/better_robot/kinematics/forward.py::forward_kinematics)
def forward_kinematics(model, q):
    check_tensor("q", q, shape=(model.nq,), dtype=model.dtype, device=model.device)
    return _forward_kinematics(model, q)
# bad
def _fk_pass(model, q):
    if q.shape[-1] != model.nq: raise ...           # re-validates on every internal call
```

Enforcement: `tests/contract/test_boundary_validation_count.py` (counts validation calls per public entry: one
per tensor input, `EXPECTED_CHECKS`); `[review]`.

## C-ERR-2 Specific exception types from the `BetterRobotError` tree (MUST)

Every raised error is a subclass of `better_robot.exceptions.BetterRobotError` (`ShapeError`,
`DeviceMismatchError`, `ModelError`, `UnknownNameError`, ...). Plain `ValueError`/`RuntimeError` are not raised by the library. The one
exception: a value of the wrong Python type (not a `torch.Tensor`, not a `KinematicState`) raises Python's
`TypeError` (the tree has no node for it; `PLAIN_RAISES` in `tests/contract/test_error_tree.py` lists each
site). An exception class exists only if something raises it; never-raised types are removed.
An unknown name (a model's joint, frame or body, a problem's term or variable, a viewer's group) raises
`UnknownNameError` (also a `KeyError`) with one message shape, `unknown <kind> '<name>' (valid names: [...])`; every
public id argument accepts `int | str` and resolves through `model.joint_id` / `model.frame_id` / `model.body_id`.

Enforcement: `tests/contract/test_error_tree.py` (every class in `exceptions.__all__` is raised somewhere in
`src/`); `[review]`.

## C-ERR-3 Messages say expected vs received (MUST)

A message names the parameter, the expected structure and what arrived, with no internal jargon:
`"q: expected shape (..., 7) for model 'panda', got (3, 9)"`. Include the frame/joint name when an id fails to
resolve, and the list of valid names when it is short (≤ 20).

```python { .no-run }
# good
raise ShapeError(f"q: expected trailing axis {model.nq}, got {q.shape[-1]}")
# bad
raise ShapeError("bad q")
```

Enforcement: `[review]`.

## C-ERR-4 No silent fallbacks, no broad excepts (MUST)

Never catch `Exception`/bare `except`, never fall back to a default on failure (a missing frame is an error, not
the world frame). Optional dependencies raise `ImportError` with the install hint at first use (`07`), not
later with an `AttributeError`.

```python { .no-run }
# good
try:
    import viser
except ImportError as exc:
    raise ImportError("better_robot.viewer needs viser: `uv add better-robot[viewer]`") from exc
# bad
try: ...
except Exception: return None
```

Enforcement: `[ruff:BLE001,E722]`; `[review]`.

## C-ERR-5 String-wired reads fail when the problem is built (MUST)

Terms and cached values name what they read by string (`reads=("state",)`). An unknown name raises
`UnknownNameError` (a `KeyError`) from `Problem.add`/`Problem.cached` (`optim/problem.py:401`), never from inside a
solve: wiring mistakes are found before the first step, with the message naming the known values.

Enforcement: `tests/optim/test_problem.py::test_unknown_read_fails_at_registration`.
