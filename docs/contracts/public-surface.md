# Public surface

A name is public when every segment of its dotted path is listed in its parent's `__all__`; everything else may
change without notice. This page is the one definition of what BetterRobot promises: what users may do, what the
maintainers keep in return, and which changes count as compatible.

## Both sides

| | |
|---|---|
| Public | a dotted path whose every segment is in its parent's `__all__` (C-EXT-3); the tracked snapshot `tests/contract/public_api.json` lists them all with their signatures |
| Canonical | the root for the fourteen first-level names (record [0019](decisions/0019-root-namespace.md)), else the package whose `__all__` lists the name |
| Users may | call public names; add a joint kind by subclassing `model.JointModel`, a solver by subclassing `optim.Solver`, a viewer layer by subclassing `viewer.Layer`; write a term as a plain function and a parser as a function returning `io.IRModel` (record [0025](decisions/0025-extension-model.md)) |
| Users may not | subclass a concrete class (every one is `@final`); rely on dataclass field order (fields after the first two are keyword-only); import a `_` name, or `tests/`, `examples/` or `tools/` |
| Experimental | nothing in this release |

## Horizons

- Python API before 1.0: a minor release may change or remove a public name, with a migration line in the
  changelog; a patch release changes no public signature (C-EXT-6). This major keeps no deprecated alias (C-EXT-4).
- Files: BetterRobot writes no model file (record [0032](decisions/0032-br-serialization-non-goal.md)); it reads
  URDF and MJCF as their own specifications define them, within the loaders' documented scope.
- Numerical results: within the documented tolerance of the named oracle (Pinocchio or MuJoCo in
  `tests/parity/`); bitwise only where a test pins it.
- [Coupling standard](coupling-standard.md) (shapes, pose layout, axes, frame identifiers that other repositories
  rely on): additions only (record [0026](decisions/0026-coupling-standard-page.md)).

## Compatible changes

Every change not in a compatible row is breaking.

| Change | Class |
|---|---|
| Add a function, class or module | compatible |
| Add a keyword-only parameter with a default | compatible |
| Add a field to a frozen dataclass, keyword-only with a default | compatible |
| Add an option to an open `Literal` set (one whose docstring says callers must accept unknown values) | compatible |
| Add a method to a class users subclass (`JointModel`, `Solver`, `Layer`) | breaking: it can shadow a user's method |
| Add, remove or reorder a positional parameter | breaking |
| Change a default value | breaking (behaviour) |
| Remove or rename a public name, field or parameter | breaking |
| Add an option to a closed `Literal` set (every other one) | breaking |
| Change a result beyond its documented tolerance | breaking |

## Checks

`tests/contract/test_public_surface.py` walks the `__all__` tree; `tests/contract/test_api_snapshot.py` fails on any
change of a public name or signature until the snapshot changes in the same commit, which makes the class of each
change visible in review; `tests/contract/test_abstract_or_final.py` keeps every class abstract or `@final`;
`tests/contract/test_keyword_only.py` keeps fields and parameters after the second keyword-only.
