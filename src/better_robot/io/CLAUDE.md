# `io/` — Parsing and Model Construction

```text
URDF | MJCF | ModelBuilder -> IRModel -> build_model -> Model
```

Parsers own source-format interpretation and emit flat, order-independent IR
records. `build_model` owns tree validation, topological order, concrete joint
selection, reduced mimic coordinates, packed tensors, frames, limits, and
metadata. Do not move format-specific cases into kinematics or dynamics.

`load(...)` dispatches by suffix or an explicit format. `free_flyer=True`
selects a `JointFreeFlyer` root; `root_joint=` is the general form. Joints keep
the source's order wherever it is topological (a Kahn sort, parents first); a
body's frame carries the body's name; `q_neutral` is clamped into the limits.

`ModelBuilder` and file parsers meet at the IR boundary: every `IRJoint`
carries its `JointModel` (`ModelBuilder.add_joint(kind=)` takes any instance).
Mimic tags stay on supported concrete scalar joints and are resolved centrally
by `build_model`.

For a new format, write a function returning `IRModel` and pass its result to
`build_model`; there is no registry, and model construction does not change.

Design contracts (rule IDs `C-*`, enforcement matrix, ADRs): `docs/contracts/00-index.md`.
