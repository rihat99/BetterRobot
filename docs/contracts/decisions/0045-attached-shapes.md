# 0045 A shape is attached to a body by an `AttachedShape` record

**Status:** Accepted (integrator, 2026-11-10). **Date:** 2026-11-10

## Context
Record [0016](0016-geometry-out-of-model.md) puts the body frame and a `collision: bool` on each shape record
(`Sphere`, `Box`, `Cylinder`, `Capsule`, `Mesh`). The same shape records are what a caller passes to
`ModelBuilder.add_shape` (record [0015](0015-io-entry-points.md)), where the body and the collision flag are
arguments. Holding them on the shape as well gave two places to say which body a shape belongs to.

## Decision
We will keep a shape record to its geometry, placement and colour, and attach it through
`model.AttachedShape(body, shape, collision)`, the record type of `Geometry.shapes`, so that
`ModelBuilder.add_shape(body, shape, *, collision=True)` takes the same shape records.

## Example
```python { .no-run }
capsule = br.model.Capsule(radius=0.04, length=0.1, placement=placement)   # placement in the body frame
builder.add_shape("hand", capsule, collision=True)
geometry = br.io.build_geometry(ir, model)
attached = geometry.shapes[0]        # AttachedShape(body=<body id>, shape=Capsule(...), collision=True)
```

## Consequences
One more public name in `model`. A shape record is reusable across bodies. `AttachedShape.body` holds the body id,
so an error about a shape names the body by id; naming it by name needs the model at the call (an open item for
the owner, not decided here).

## Alternatives
The body and the flag on each shape record, as 0016 wrote: two places hold the same fact once `add_shape` takes the
body. Pinocchio's `GeometryObject` carries the parent joint and frame beside the shape, which is this record's
split.

## Compatibility
Adds `AttachedShape`; shape records lose the body frame and the flag that 0016 named. No release shipped either
form.

## Enforcement
`tests/io/test_geometry.py` (shapes round-trip into `Geometry` as `AttachedShape` records);
`tests/contract/test_api_snapshot.py`.

## Open questions
None.
