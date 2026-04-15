# io/ — Robot Loading and Parsing

## Architecture

Parser → IR (Intermediate Representation) → `build_model()` → frozen Model.

Parsers live at the boundary. They emit a common IR; a single factory produces the Model. This decouples format support from the data model.

## IR Structure

```python
IRModel → list[IRJoint] + list[IRBody] + list[IRFrame] + list[IRGeom]
```

IR is flat and order-unconstrained — no topo-sort or idx_q/idx_v yet. That happens in `build_model()`.

## Parsers

| Parser | Input | Module |
|--------|-------|--------|
| `parse_urdf` | URDF file or `yourdfpy.URDF` object | `parsers/urdf.py` |
| `parse_mjcf` | MuJoCo MJCF XML | `parsers/mjcf.py` |
| `ModelBuilder` | Python fluent API | `parsers/programmatic.py` |

## build_model()

`build_model(ir, root_joint=None, device=None, dtype=float32)`:
1. Replace root joint with `root_joint` if supplied (e.g., `JointFreeFlyer` for floating-base)
2. Resolve mimic edges
3. Topo-sort parents-before-children
4. Assign `idx_q`, `idx_v` by accumulating nq, nv
5. Select concrete `JointModel` from kind + axis
6. Pack tensors, build frames, return frozen `Model`

## Public Entry Point

`load(source, *, free_flyer=False, device=None, dtype=None)` — dispatches by suffix or type, calls parser + `build_model()`.

## Adding a New Format

1. Create parser under `parsers/` that emits `IRModel`
2. Register suffix in `load()` dispatch
3. No changes to `build_model()` or downstream code needed
