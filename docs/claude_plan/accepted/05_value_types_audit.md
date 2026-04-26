# 05 · Value-types audit: `Inertia`, `Symmetric3`, `Motion`, `Force`

★★ **Structural.** Resolves a specification ambiguity in
[03_LIE_AND_SPATIAL.md §7](../../design/03_LIE_AND_SPATIAL.md) about
whether `Inertia` is a typed dataclass or a packed `(..., 10)`
tensor. Answer: it should be the *type*, with the packed tensor as
its on-storage layout, and that distinction should be visible in the
API.

## Problem

Today `Inertia` is both:

- a `@dataclass(frozen=True)` with named accessors (`mass`, `com`,
  `inertia_matrix`) and named-constructor factories
  (`from_sphere`, `from_box`, `from_capsule`, `from_ellipsoid`,
  `from_mass_com_sym3`); and
- a packed `(..., 10)` tensor in storage on `Model.body_inertias`,
  unboxed back into `Inertia` only on demand.

Both decisions are individually correct. The issue is that the
unboxing is implicit and inconsistent:

- `model.body_inertias` is a raw tensor.
- `inertia.apply(motion)` returns a `Force` (typed).
- `inertia.add(other)` returns an `Inertia` (typed).
- `inertia.se3_action(T)` takes a *raw tensor* `T`, not an `SE3`
  (cross-references [Proposal 01](01_lie_typed_value_classes.md)).
- A user who reads `model.body_inertias[0]` gets a `(10,)` tensor,
  not an `Inertia` instance — they have to wrap by hand.

`Symmetric3` has the same shape: `(..., 6)` packed lower-triangular,
factories and `to_matrix()`/`from_matrix()` round-trips, no clear
boundary at which the typed view is the canonical one.

## Goal

A single, documented rule: **storage on `Model`/`Data` uses raw
tensors; the typed wrapper is the user-facing accessor**. Same as
the SE3/SO3 decision in [Proposal 01](01_lie_typed_value_classes.md).

## The proposal

### 5.A Add typed accessors on `Model` and `Data`

```python
# data_model/model.py
@dataclass(frozen=True)
class Model:
    body_inertias: Float[Tensor, "*B nbodies 10"]  # storage
    ...

    def body_inertia(self, body_id: int) -> "Inertia":
        """Typed view of the body's inertia."""
        return Inertia(self.body_inertias[..., body_id, :])

    def all_body_inertias(self) -> tuple["Inertia", ...]:
        """Typed views of every body. Reads from the same storage —
        does not copy."""
        return tuple(self.body_inertia(i) for i in range(self.nbodies))
```

```python
# data_model/data.py
@dataclass
class Data:
    joint_pose_world: SE3Tensor | None = None     # storage
    ...

    def joint_pose(self, joint_id: int) -> "SE3":
        if self.joint_pose_world is None:
            raise StaleDataError("...")
        return SE3(self.joint_pose_world[..., joint_id, :])

    def frame_pose(self, frame_or_id: str | int) -> "SE3":
        ...
```

User code that wants the typed view asks for it; user code that
wants the raw tensor reads `data.joint_pose_world` directly.

### 5.B Pin `Inertia.se3_action` to accept either form

```python
# spatial/inertia.py
def se3_action(self, T: SE3 | Float[Tensor, "*B 7"]) -> "Inertia":
    T_data = T.data if isinstance(T, SE3) else T
    ...
```

Same for `Motion.se3_action`, `Force.se3_action`, `SE3.act_motion`,
`SE3.act_force`, `SE3.act_inertia`. The pattern: methods accept the
typed wrapper *or* the raw tensor; methods return the typed wrapper.
This rules out the awkward middle state where users have to unbox
manually.

### 5.C `Symmetric3` stays in `spatial/__init__.py` — unpromote, don't hide

`Symmetric3` is used in two places:

- packed inertia matrix inside `Inertia` (`(..., 6)` lower-triangular).
- `compute_centroidal_dynamics` (future), as the symmetric block of
  the centroidal inertia.

It is not a typical user-facing concept. The proposal:

- **Keep** `Symmetric3` reachable via
  `from better_robot.spatial import Symmetric3`. It is a small,
  coherent type already part of the spatial vocabulary, and an
  earlier draft of this proposal that hid it was a step too far —
  the gpt_plan review correctly pointed out there is no evidence of
  user misuse to justify the breakage.
- **Do not** promote `Symmetric3` to top-level `better_robot`
  ([Proposal 06](06_public_api_audit.md) keeps it submodule-only).
- The factories `Inertia.from_sphere(...)` etc. remain the
  user-facing construction path; a `from_mass_com_matrix` helper is
  added for ergonomics:

```python
@classmethod
def from_mass_com_matrix(cls, mass, com, I3x3) -> "Inertia":
    sym3 = Symmetric3.from_matrix(I3x3).data
    return cls.from_mass_com_sym3(mass, com, sym3)
```

Documentation calls `Symmetric3` an "advanced helper for callers
who need direct access to the packed lower-triangular layout" —
not the headline API.

### 5.D Document the spatial-cross duality; do **not** stub-fill `Force.cross_motion`

`spatial/force.py:cross_motion` raises `NotImplementedError` today.
An earlier draft of this proposal pushed for filling it in for
"symmetry" with `Motion.cross_force`. The gpt_plan review rejected
that, correctly:

- `Force × Motion` is **not** a standard spatial-algebra operation.
  The dual action that does exist is `Motion.cross_force` (the dual
  ad operator).
- Adding a "symmetric" method that is not a textbook operation
  teaches users a non-standard algebra. Worse, the convention they
  *would* use (e.g. for muscle-driven dynamics derivatives) is
  algorithm-specific, not type-level.

Direction adopted instead:

- **Keep** `Force.cross_motion` raising `NotImplementedError` — but
  upgrade the message to point at `Motion.cross_force` and at a new
  `docs/design/03_LIE_AND_SPATIAL.md §7.X` paragraph explaining the
  duality.
- **Do not** add the method until a concrete dynamics algorithm
  needs it; if and when one does, the operation gets a name that
  reflects the algorithm, not faux symmetry.
- Tests retain the `Motion.cross_force` coverage; no new
  `Force.cross_motion` test is added.

### 5.E `Motion`, `Force` accept and return the typed-pose argument

Mirror of 5.B for non-Inertia value types. `Motion.se3_action(T)`
should accept `SE3 | Tensor`. The internal implementation reaches
into `T.data` if needed.

## What the typed boundary buys

Reading user-side code becomes self-explanatory:

```python
# Before
F = Inertia(model.body_inertias[..., link_id, :]).se3_action(
        data.joint_pose_world[..., joint_id, :]).apply(
        Motion(data.joint_velocity_world[..., joint_id, :]))

# After
T = data.joint_pose(joint_id)             # SE3
v = data.joint_velocity(joint_id)         # Motion
I = model.body_inertia(link_id)           # Inertia
F = T.act_inertia(I).apply(v)             # Force
```

Same number of operations, half the noise.

## The amendment to 03_LIE_AND_SPATIAL §7

Currently:

> `Inertia` lives in `spatial/inertia.py` (see 03 LIE_AND_SPATIAL).

Add:

> Storage of inertias lives on `Model.body_inertias` as a packed
> `(..., nbodies, 10)` tensor (the layout
> `[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`). The typed
> `Inertia` wrapper is the user-facing accessor — obtained via
> `model.body_inertia(body_id)` — and is what library functions
> *return*. Functions that take an `Inertia` may also accept the
> raw `(..., 10)` tensor; functions that return an `Inertia` always
> return the typed wrapper. The same rule applies to `SE3`, `SO3`,
> `Motion`, `Force` (see [Proposal 01](01_lie_typed_value_classes.md)).

## Files that change

```
spatial/__init__.py                  unchanged — Symmetric3 stays re-exported
spatial/inertia.py                   from_mass_com_matrix added; se3_action accepts SE3
spatial/motion.py                    se3_action accepts SE3
spatial/force.py                     se3_action accepts SE3; cross_motion error message
                                     updated to point at Motion.cross_force and §7.X
data_model/model.py                  body_inertia(), all_body_inertias()
data_model/data.py                   joint_pose(), frame_pose(),
                                     joint_velocity(), frame_velocity()
                                     (cross-references Proposal 07)
tests/data_model/test_model_typed.py new — round-trip tests on body_inertia()
docs/design/03_LIE_AND_SPATIAL.md    §7 amendment + new §7.X (cross duality)
```

## Tradeoffs

| For | Against |
|-----|---------|
| One canonical place for the typed/raw boundary. | Two ways to read the same data (`model.body_inertia(i)` vs `model.body_inertias[..., i, :]`). Mitigation: doc 03 makes the storage-vs-view distinction explicit. |
| User-side code becomes the math. | Migration touches `examples/`. |
| `Symmetric3` stays where existing internal callers expect to find it. | None significant — this preserves the status quo. |

## Acceptance criteria

- `model.body_inertia(body_id)` returns an `Inertia` whose
  `.data` field is `model.body_inertias[..., body_id, :]` (no copy
  — shared storage).
- `inertia.se3_action(T)` accepts both `SE3` and the raw `(..., 7)`
  tensor; same for `Motion.se3_action`, `Force.se3_action`.
- `Force.cross_motion` continues to raise `NotImplementedError`,
  with a message that names `Motion.cross_force` as the standard
  dual operation and references the §7.X duality paragraph.
- `from better_robot.spatial import Symmetric3` continues to work.
- `Symmetric3` is **not** exported from top-level `better_robot`
  (per [Proposal 06](06_public_api_audit.md)).
- `examples/01_basic_ik.py`, `examples/02_g1_ik.py` use the typed
  accessors at every call site that addresses a single joint or
  frame.

## Cross-references

- [Proposal 01](01_lie_typed_value_classes.md) — `SE3` and `SO3`
  follow the same pattern.
- [Proposal 06](06_public_api_audit.md) — `Inertia`, `Motion`,
  `Force` are added to `__all__`; `Symmetric3` is not.
- [Proposal 07](07_data_cache_invariants.md) — `Data.joint_pose(...)`
  and `Data.joint_velocity(...)` raise on stale caches.
