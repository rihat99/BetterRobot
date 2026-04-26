# 07 · `Data` cache invariants — make `_kinematics_level` enforce

★★ **Structural.** Closes a documented invariant in
[02_DATA_MODEL.md §3](../../design/02_DATA_MODEL.md) that is currently
declared but not enforced.

## Problem

`Data` is the per-query workspace, with a `_kinematics_level: int`
field intended to track whether forward kinematics has been computed
through level 1 (placements), 2 (velocities), or 3 (accelerations).
The doc says:

> `_kinematics_level` mirrors Pinocchio's "kinematic level" invariant
> — it lets functions that need `v`-level kinematics assert that
> level-2 has been computed.

In practice today:

```python
# data_model/data.py
@dataclass
class Data:
    ...
    _kinematics_level: int = 0
```

Nothing reads it. `compute_joint_jacobians(model, data)` does *not*
verify `data.joint_pose_world is not None`. If FK has not been run,
the function reads `None` and crashes deep in the math layer with an
unhelpful `TypeError: 'NoneType' has no attribute '__getitem__'`.

The invariant is there. The enforcement is not.

The `q` field is even more subtle. After

```python
data = forward_kinematics(model, q)        # computes everything from q
data.q = new_q                              # caller mutates q
J = compute_joint_jacobians(model, data)   # uses joint_pose_world from old q
```

the cache is **stale** but no one notices until the IK gets the
wrong gradients.

## Goal

Make `Data` carry its own consistency contract:

1. Functions that need a kinematics level assert it.
2. Mutating `q` (or any other input) invalidates downstream caches.
3. Errors are typed (`StaleCacheError`) and tell the user what to
   call.

## The proposal

### 7.A `KinematicsLevel` enum

Already proposed in [Proposal 04](04_typing_shapes_and_enums.md):

```python
class KinematicsLevel(int, Enum):
    NONE          = 0
    PLACEMENTS    = 1
    VELOCITIES    = 2
    ACCELERATIONS = 3
```

`Data._kinematics_level: KinematicsLevel = KinematicsLevel.NONE`.

### 7.B Functions assert their required level

```python
# kinematics/jacobian.py
def compute_joint_jacobians(model: Model, data: Data) -> None:
    data.require(KinematicsLevel.PLACEMENTS)
    ...
```

```python
# data_model/data.py
class Data:
    ...
    def require(self, level: KinematicsLevel) -> None:
        if self._kinematics_level < level:
            raise StaleCacheError(
                f"this function needs kinematic level {level.name} "
                f"({level.value}); data is at {self._kinematics_level.name}. "
                f"Call forward_kinematics(model, data) (level 1) before "
                f"compute_joint_jacobians."
            )
```

`StaleCacheError` is a new exception, derived from
`BetterRobotError, RuntimeError` per the
[17_CONTRACTS.md §2 taxonomy](../../conventions/17_CONTRACTS.md).

The matrix:

| Function | Required level |
|----------|----------------|
| `update_frame_placements` | `PLACEMENTS` (joint_pose_world filled) |
| `compute_joint_jacobians` | `PLACEMENTS` |
| `get_frame_jacobian`, `get_joint_jacobian` | `PLACEMENTS` (and joint_jacobians filled, see 7.D) |
| `rnea` (when implemented) | computes its own way; no level needed |
| `bias_forces` | none — internal |
| `compute_centroidal_map` | `PLACEMENTS` |
| `center_of_mass` | `PLACEMENTS` |

### 7.C Setting `q`, `v`, `a` invalidates downstream caches

Today `Data.q = new_q` is allowed (the dataclass is mutable) but
silently leaves caches stale. Add `__setattr__` discipline:

```python
@dataclass
class Data:
    ...

    def __setattr__(self, name: str, value) -> None:
        if name == "q" and getattr(self, "_kinematics_level", 0) > 0:
            self._invalidate(KinematicsLevel.NONE)
        elif name == "v" and getattr(self, "_kinematics_level", 0) > KinematicsLevel.PLACEMENTS:
            self._invalidate(KinematicsLevel.PLACEMENTS)
        elif name == "a" and getattr(self, "_kinematics_level", 0) > KinematicsLevel.VELOCITIES:
            self._invalidate(KinematicsLevel.VELOCITIES)
        object.__setattr__(self, name, value)

    def _invalidate(self, level_to_keep: KinematicsLevel) -> None:
        """Drop caches above the given level."""
        if level_to_keep < KinematicsLevel.PLACEMENTS:
            self.joint_pose_local = None
            self.joint_pose_world = None
            self.frame_pose_world = None
            self.joint_jacobians = None
            self.joint_jacobians_dot = None
        if level_to_keep < KinematicsLevel.VELOCITIES:
            self.joint_velocity_world = None
            self.joint_velocity_local = None
        if level_to_keep < KinematicsLevel.ACCELERATIONS:
            self.joint_acceleration_world = None
            self.joint_acceleration_local = None
        # always drop the (q, v, a)-derived dynamics caches
        self.mass_matrix = self.coriolis_matrix = self.gravity_torque = None
        self.bias_forces = self.ddq = None
        self.centroidal_momentum_matrix = self.centroidal_momentum = None
        self.com_position = self.com_velocity = self.com_acceleration = None
        object.__setattr__(self, "_kinematics_level", level_to_keep)
```

The `__setattr__` runs on the constructor too — initializing `q`
sets `_kinematics_level` to `NONE` (the default), which is correct.

### 7.D Typed accessors enforce non-staleness

[Proposal 05](05_value_types_audit.md) introduces typed accessors on
`Data`. They check non-None, throwing `StaleCacheError`:

```python
def joint_pose(self, joint_id: int) -> SE3:
    if self.joint_pose_world is None:
        raise StaleCacheError(
            "joint_pose_world is not populated. "
            "Call forward_kinematics(model, data) first."
        )
    return SE3(self.joint_pose_world[..., joint_id, :])
```

This works in tandem with 7.B: assert via `require(...)` for
internal functions; assert via `joint_pose(...)` for user-facing
access. Both raise the same error type.

### 7.E.0 What this proposal cannot detect — in-place mutation

The `__setattr__` hook fires on **assignments** like
`data.q = new_q`. It does not — and cannot — fire on in-place
tensor mutation:

```python
data.q[..., 0] += 1.0      # silently stale; __setattr__ is NOT called
data.q.add_(other)         # silently stale; same reason
data.q.copy_(new_q)        # silently stale; same reason
```

The caches downstream of `data.q` will *not* be invalidated in
these cases. The cause is a fundamental Python limitation: tensor
views and in-place operations bypass the dataclass attribute
machinery entirely.

The contract this proposal lands is therefore narrower than
"in-place mutation is detected":

> BetterRobot detects **reassignment** of `q`, `v`, `a`, and any
> public input field on `Data`, and invalidates downstream caches
> on reassignment. In-place tensor mutation of those fields is
> **not** detected; it is a documented misuse pattern. Public API
> docs and the typed accessors raise on staleness, not on
> mutation.

We cover this with documentation, not detection. The user-facing
guidance:

- **Reassign**, do not mutate: `data.q = new_q`, never
  `data.q[...] = new_q[...]`.
- If a hot path absolutely needs in-place updates, follow them
  with `data.invalidate(KinematicsLevel.NONE)` — a public method
  that resets the level explicitly.
- The IK and trajopt solvers do reassign; user code following
  their pattern stays in the safe zone.

A future hardening (out of scope here) could store `q` behind a
property that returns a read-only view; that adds overhead and
breaks autograd in subtle ways, so it is not worth it pre-1.0.

### 7.E `forward_kinematics` is the only writer of `_kinematics_level`

Today the field is set nowhere because nothing reads it. After this
proposal it must be advanced explicitly:

```python
# kinematics/forward.py
def forward_kinematics(model, q_or_data, *, compute_frames=False) -> Data:
    ...
    data.joint_pose_world = ...
    data.joint_pose_local = ...
    if compute_frames:
        data.frame_pose_world = ...
    object.__setattr__(data, "_kinematics_level", KinematicsLevel.PLACEMENTS)
    return data
```

Future kinematics-level-2 functions (RNEA-style first pass,
joint_velocity propagation) advance to `VELOCITIES`. Same pattern at
level 3.

### 7.F The deprecation shims still work

[02_DATA_MODEL.md §11](../../design/02_DATA_MODEL.md) defines
`@property` shims for the legacy short names (`oMi`, `oMf`, …). They
already use `object.__setattr__` to forward writes; they continue to
work under 7.C because they bypass the field-name dispatch.

For the v1.1 removal target, this is fine; the shims emit
`DeprecationWarning` and forward.

## What about `tau`, `joint_forces`?

`tau` is an *input* (or RNEA *output*), not a cache. It's not part
of the kinematics-level invariant. We keep it raw.

`joint_forces` (RNEA's `data.f`) is internal scratch. Same treatment:
it's invalidated whenever any `(q, v, a)` field changes. Already
covered by 7.C's wholesale dynamics-cache invalidation.

## Cost discussion

`__setattr__` is on the hot path of nothing — `Data` is mutated a
handful of times per query. The dispatch overhead is negligible.

## Files that change

```
data_model/data.py                   _kinematics_level: KinematicsLevel; require(); _invalidate(); __setattr__
exceptions.py                        StaleCacheError
kinematics/forward.py                advance level after writes
kinematics/jacobian.py               require(KinematicsLevel.PLACEMENTS) at entry
                                     of compute_joint_jacobians, get_*_jacobian
dynamics/centroidal.py:center_of_mass require(KinematicsLevel.PLACEMENTS) (when v=None)
                                     require(KinematicsLevel.VELOCITIES) (when v passed)
tests/data_model/test_cache_invariants.py  new — exercises the __setattr__ logic
                                     and StaleCacheError messages
docs/conventions/17_CONTRACTS.md     §2 add StaleCacheError row
```

## Tradeoffs

| For | Against |
|-----|---------|
| Stale-cache bugs become impossible (or at worst, immediate type errors). | A small `__setattr__` overhead — bounded, not on the kernel hot path. |
| Error messages tell the user what to call. | One more exception class to learn. |
| Future dynamics functions land into a contract that already exists. | Some user code that mutates `data.q` and *expects* caches to remain (incorrectly) will need to rewrite. Mitigation: this is buggy anyway; the new error message is louder than the old wrong answer. |
| Mirrors Pinocchio's level invariant — port-friendly for users coming from C++ Pinocchio. | None significant. |

## Acceptance criteria

- `Data._kinematics_level` is `KinematicsLevel`, not raw `int`.
- `compute_joint_jacobians(model, fresh_data)` raises
  `StaleCacheError` (because nothing has computed FK yet).
- `forward_kinematics(model, q)` returns `Data` with
  `_kinematics_level == KinematicsLevel.PLACEMENTS`.
- After `data.q = new_q`, `data.joint_pose_world is None` and
  `data._kinematics_level == KinematicsLevel.NONE`.
- Calling `data.joint_pose(0)` on stale data raises a
  `StaleCacheError` with message naming `forward_kinematics`.
- `tests/data_model/test_cache_invariants.py` exercises every
  transition in the matrix, **and** has an explicit "documented
  limitation" test that asserts in-place mutation
  (`data.q[..., 0] += 1`) is *not* detected — codifying the scope
  of the contract so a future contributor cannot quietly claim
  more.
- `Data.invalidate(level: KinematicsLevel)` is a public method;
  user code that does in-place mutation calls it explicitly.
- The user-facing docs (and the docstring on `Data`) name
  reassignment as the supported pattern.

## Cross-references

- [Proposal 04](04_typing_shapes_and_enums.md) — defines
  `KinematicsLevel`.
- [Proposal 05](05_value_types_audit.md) — adds typed accessors
  that hook into `StaleCacheError`.
- [Proposal 06](06_public_api_audit.md) — `KinematicsLevel` does
  *not* need to be in `__all__` (it's an internal contract); the
  exception class belongs under `better_robot.exceptions`.
- [17_CONTRACTS.md §2](../../conventions/17_CONTRACTS.md) — adds
  `StaleCacheError` to the typed exceptions table.
