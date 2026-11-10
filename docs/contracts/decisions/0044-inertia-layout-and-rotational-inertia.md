# 0044 The inertia vector follows MuJoCo's `fullinertia` order; `spatial.rotational_inertia` is public

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record [0014](0014-tensor-functions-for-lie-and-spatial.md) lists the spatial functions that replace the value
classes; the list has `inertia_matrix` (the 6x6 matrix) but no function that returns the 3x3 rotational inertia,
which callers (BetterHuman's mass tools, the dynamics tests) rebuilt by hand from the packed vector. Record
[0026](0026-coupling-standard-page.md) writes the 10-vector as `[m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]` without
naming whose order that is, and Pinocchio's `toDynamicParameters` uses another one.

## Decision
We will add `spatial.rotational_inertia(inertia) -> (..., 3, 3)` to the functions of record 0014, and name the
10-vector's order in the coupling standard as MuJoCo's `fullinertia` order, about the centre of mass.

## Example
```python { .no-run }
inertia = torch.tensor([1.0, 0.1, 0.0, 0.0, 0.02, 0.03, 0.04, 0.0, 0.0, 0.0])   # [m, c, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
br.spatial.rotational_inertia(inertia)    # (3, 3) about the centre of mass
br.spatial.inertia_matrix(inertia)        # (6, 6) about the body origin, linear-first
```

## Consequences
One more public name in `spatial`. The order matches MuJoCo's MJCF attribute, so an MJCF `fullinertia` reads in
without a permutation; a Pinocchio comparison converts explicitly (the parity tests do).

## Alternatives
Pinocchio's `[m, m c, I about the origin as Ixx, Ixy, Iyy, Ixz, Iyz, Izz]`: a layout that mixes mass into the first
moment and changes with the reference point. Keeping the 3x3 helper private: every caller writes it again.

## Compatibility
Adds a name; the layout itself is unchanged since record 0014.

## Enforcement
C-TENS-1 and C-STATE-3 through `tests/contract/_public_ops.py` (`spatial.rotational_inertia` is a registered entry
point); `tests/contract/test_api_snapshot.py` (the name is in the public snapshot).

## Open questions
None.
