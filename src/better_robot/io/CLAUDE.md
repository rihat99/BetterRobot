# `io/` — Parsing and Model Construction

```text
URDF | MJCF | ModelBuilder -> IRModel -> build_model -> Model
```

Parsers own source-format interpretation and emit flat, order-independent IR
records. `build_model` owns tree validation, joint order, reduced mimic
coordinates, packed tensors, frames and limits, as named stages into one
`Model(...)`. Do not move format-specific cases into kinematics or dynamics.

`load(path, *, format=None, root_joint=None, device=None, dtype=...)` takes a
path only and picks the parser by suffix or `format`; `root_joint=` replaces
the joint under the universe (`JointFreeFlyer()` floats the base). Joints keep
the source's order wherever it is topological (a Kahn sort, parents first); a
body's frame carries the body's name; `q_neutral` is clamped into the limits.

`ModelBuilder` and file parsers meet at the IR boundary: every `IRJoint`
carries its `JointModel` (`ModelBuilder.add_joint(kind=)` takes any instance).
Mimic tags stay on supported concrete scalar joints and are resolved centrally
by `build_model`.

For a new format, write a function returning `IRModel` and pass its result to
`build_model`; there is no registry, and model construction does not change.

Design contracts (rule IDs `C-*`, enforcement matrix, ADRs): `docs/contracts/00-index.md`.
