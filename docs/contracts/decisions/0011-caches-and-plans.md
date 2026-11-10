# 0011 Plans live on the model; nothing is keyed on a tensor version

**Status:** Accepted (owner); amended by 0040; amendment proposed in 0041, 0042. **Date:** 2026-11-01

## Context
Five caches are written with `object.__setattr__` onto frozen objects owned by other packages, keyed on torch's
private `_version` and `torch._C._functorch.is_functorch_wrapped_tensor`: `kinematics/_fk_lean.py::placement_matrices`
and `::get_plan`, `kinematics/_jacobian_columns.py::get_plan`, `dynamics/_world.py::get_dynamics_plan` and
`::spatial_inertias`. `KinematicState._memo` is not declared `init=False`, so `dataclasses.replace(state, q=q2)`
returns a stale `frame_pose`, and `dynamics/state.py` writes `_memo["world_geometry"]` from another package.

## Decision
We will keep plans (FK, Jacobian columns, dynamics support, manifold groups, mimic matrices) in `Model._plans`, keyed
by (kind, dtype, device) and built as pure functions of the static fields; compute placement matrices and 6x6 spatial
inertias per call; declare `KinematicState._memo` with `init=False`, keyed by grad mode; and let `dynamics` derive its
world geometry per call from the memoised Jacobian columns.

## Example
```python { .no-run }
def plan(self, kind: str, build: Callable[[Model], Plan], *, dtype: torch.dtype, device: torch.device) -> Plan:
    key = (kind, dtype, device)               # no tensor version, no functorch probe
    if key not in self._plans:
        self._plans[key] = build(self)        # a pure function of the static fields
    return self._plans[key]

state2 = dataclasses.replace(state, q=q2)     # the memo is not copied: state2.frame_poses is recomputed
```

## Consequences
A cache changes cost, never a result, and no hidden input remains. Each FK call converts the `(njoints, 7)` placements
to matrices again (two small ops). If the FK benchmark at batch 1 and 4096 shows that conversion above the threshold,
a value-side memo on `Model`, keyed by grad mode, is added by its own record with the measurement (5.6).

## Alternatives
A declared value-side memo from the start: it adds a cache before a measurement asks for one. A `DynamicsState` type
for the world geometry: a new type for one subtraction. Keeping the `_version`-keyed caches behind a declared field:
the key is a hidden input (1.5). Drake's `Context` caches declared entries with declared prerequisites, never a
counter on a buffer; Pinocchio's `Data` holds every intermediate, which ADR 0002 rejects.

## Compatibility
Internal; no public name changes. `dataclasses.replace` on a state becomes correct. Reversible: yes.

## Enforcement
New `tests/contract/test_caches.py`: `dataclasses.replace` on a field, then every memoised property recomputes (the
stale `frame_pose` reproduction), and no `_version`, `torch._C` or `object.__setattr__` outside the memo functions in
`src`; `tests/kinematics/test_fk_compile.py` covers the public wrapper with `fullgraph=True`;
`tests/contract/test_compile_counters.py` checks graph breaks and recompiles on CPU against
`benchmarks/thresholds/torch-<version>.json` in the full tier, and the release tier checks them on CUDA.

## Open questions
None.
