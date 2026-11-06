# 0020 One error tree under `BetterRobotError`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`exceptions.py` has 8 classes. `BackendNotAvailableError` describes a backend registry that does not exist.
`io.IRError` and `ModelInconsistencyError` overlap. A missing `mujoco` raises a plain `ImportError` while a missing
`yourdfpy` raises `BackendNotAvailableError` with a wrong install hint. `optim` raises builtins. Mixed dtypes pass
silently through `integrate`.

## Decision
We will keep `BetterRobotError` above `ShapeError`, `DtypeMismatchError`, `DeviceMismatchError`, `ArgumentError`,
`ModelError` (absorbing `IRError`) and `QuaternionNormError` (all also `ValueError`), `UnknownNameError` (also
`KeyError`, listing the valid names when there are at most 20) and `OptionalDependencyError` (also `ImportError`,
carrying the install line), raise every class somewhere, and let `optim` import `exceptions`, which imports nothing.

## Example
```python { .no-run }
try:
    model = br.load("scene.xml")
except br.exceptions.OptionalDependencyError as error:
    print(error)            # "mujoco is not installed; run: uv sync --extra mjcf"
except br.exceptions.BetterRobotError:
    raise                   # one except clause catches the library, shape and dtype errors included
```

## Consequences
Callers catch the library with one class or the builtin they already know. `optim` stays robot-agnostic: `exceptions`
has no imports. A class nobody raises is deleted.

## Alternatives
`optim` keeps importing only torch and raises builtins, so it could leave as its own package: callers then handle two
vocabularies. numpy's `AxisError` derives from both `ValueError` and `IndexError`, the precedent for classes that are
also builtins; Pinocchio and Drake raise standard C++ exceptions with no library root.

## Compatibility
Renames `BackendNotAvailableError` to `OptionalDependencyError` and `ModelInconsistencyError` to `ModelError`, removes
`io.IRError`, adds `ArgumentError`. Callers that catch `ValueError` or `KeyError` keep working. Partly reversible:
adding classes is compatible, removing is not.

## Enforcement
`tests/contract/test_public_surface.py` (C-ERR-2); new `tests/contract/test_error_tree.py` (the bases above; every
class in `exceptions.py` is raised somewhere in `src`); `[ruff:BLE001,E722]` (C-ERR-4); the edge table row for
`optim` allows `exceptions` only.

## Open questions
None.
