# `io/` — Parsing and Model Construction

```text
URDF | MJCF | ModelBuilder -> IRModel -> build_model -> Model
```

Parsers own source-format interpretation and emit flat, order-independent IR
records. `build_model` owns tree validation, topological order, concrete joint
selection, reduced mimic coordinates, packed tensors, frames, limits, and
metadata. Do not move format-specific cases into kinematics or dynamics.

`load(...)` dispatches by suffix/type or an explicit format. `free_flyer=True`
selects a `JointFreeFlyer` root; `root_joint=` is the general form.
`preserve_joint_order=True` keeps an already-topological source order through a
stable Kahn sort; the default retains the established DFS order.

`ModelBuilder` and file parsers meet at the IR boundary. Programmatic custom
joint objects may use the builder's opaque payload, but file formats must use
serializable joint fields. Mimic tags stay on supported concrete scalar joints
and are resolved centrally by `build_model`.

For a new format, write a function returning `IRModel` and pass its result to
`build_model`; there is no registry, and model construction does not change.

Design contracts (rule IDs `C-*`, enforcement matrix, ADRs): `docs/contracts/00-index.md`.
