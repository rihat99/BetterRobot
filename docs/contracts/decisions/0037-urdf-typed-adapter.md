# 0037 URDF parsing goes through one typed adapter

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`io/parsers/urdf.py` makes 35 `getattr(x, "literal", default)` calls, including yourdfpy's private
`urdf._filename_handler` and `urdf._material_map`. `_extract_geoms` swallows resolver errors (`except (OSError,
ValueError): pass`) and keeps a branch "if the backend ever switches to a field-style object". An `ImportError` branch
guards `yourdfpy`, a core dependency. `parse` accepts a path, a `yourdfpy.URDF` object or a callable.

## Decision
We will wrap the yourdfpy version in the lockfile in one typed adapter with direct attribute access and no private
attribute (the filename handler is passed explicitly), raise `ModelError` naming the file and the element on every
failure, and make `parse` take a path.

## Example
```python { .no-run }
ir = br.io.parse("panda.urdf")                        # path only; a yourdfpy.URDF or a callable is rejected
br.io.parse("broken.urdf")                            # ModelError: broken.urdf: <link name="hand">: mesh 'hand.dae' not found
```

## Consequences
A yourdfpy upgrade that renames an attribute fails at the adapter with a clear error, not silently with a default.
Errors name the file and element. The dead `ImportError` branch goes (0020 covers optional extras). Cost: the adapter
is pinned to the lockfile's yourdfpy.

## Alternatives
Replace yourdfpy with BetterRobot's own URDF XML reader: one dependency fewer and more code to own and test. Keep the
probing: failures stay silent. Pinocchio parses URDF through urdfdom's typed C++ objects with direct member access,
which is the shape the adapter takes.

## Compatibility
Breaks `parse(yourdfpy_object)` and `parse(callable)`; a caller who held a `yourdfpy.URDF` passes its path. Reversible:
yes for the internals; the signature narrowing is a break to undo.

## Enforcement
`tests/contract/test_no_probing.py` (the 4.2 lint: no `getattr` with a literal default and no `hasattr` on third-party
objects in `io/`, with a keyed exception list); ruff `SLF001` forbids the private attributes; new
`tests/io/test_urdf_errors.py` (an unreadable mesh raises `ModelError` naming file and element).

## Open questions
None.
