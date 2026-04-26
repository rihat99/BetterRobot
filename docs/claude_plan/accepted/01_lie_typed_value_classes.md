# 01 · Lie groups: add typed value classes alongside the functional API

★★★ **Foundational.** Affects every call site that handles SE(3) or
SO(3). Proposed amendment to
[03_LIE_AND_SPATIAL.md §3, §7, §9](../../design/03_LIE_AND_SPATIAL.md).

## Problem

The library exposes Lie-group operations as **free functions** in
`better_robot.lie.se3` / `lie.so3`:

```python
from better_robot.lie import se3
T_world_ee = se3.compose(T_world_link, T_link_ee)
T_inv      = se3.inverse(T_world_ee)
xi         = se3.log(T_inv)
```

This is correct, autograd-clean, and `torch.compile`-friendly. It is
also **verbose at call sites that compose many transforms**, and it
forces every user to remember which trailing-axis layout
`(..., 7) = [tx, ty, tz, qx, qy, qz, qw]` we picked. A paper-style
expression like

```
T_world_ee = T_world_arm @ T_arm_link3 @ T_link3_ee
T_link3_world = T_world_link3.inverse()
p_world = T_world_ee @ p_ee_local
```

cannot be written today; it desugars to four nested `se3.compose` /
`se3.act` / `se3.inverse` calls. For a robotics library used widely
the readability of these call sites is part of the product.

The `spatial/` layer already does **half** of the right thing for
6D twists / wrenches / inertias: typed dataclasses (`Motion`, `Force`,
`Inertia`) wrap a `torch.Tensor` and expose named methods plus the
unambiguous vector-space operators (`+`, `-`, `__neg__`). What's
missing is the same pattern for the Lie *groups* — `SE3` and `SO3`.

## Why not subclass `torch.Tensor` (the PyPose route)

Tempting but wrong. Subclassing `torch.Tensor` and intercepting ops
via `__torch_function__` means:

- The whitelist of permitted ops is a maintenance burden — every new
  PyTorch op either passes through (correct semantics) or accidentally
  unwraps (silent breakage).
- `__torch_function__` interactions with `torch.compile`,
  `torch.func.vmap`, `torch.func.jacrev`, autograd, FSDP, distributed
  ops are subtly different and version-dependent.
- This is exactly the path that produced
  [PYPOSE_ISSUES.md](../../status/PYPOSE_ISSUES.md): correct forward,
  wrong backward, projected-tangent gradients leaking into ambient
  callers.

We pay this cost today. We should not pay it twice.

## The proposal: thin frozen dataclasses, no tensor subclass

Add `SE3` and `SO3` as **frozen dataclasses** in `lie/types.py`
— the `lie/` layer already owns SE(3)/SO(3) semantics; the typed
wrappers are the same semantics with named methods. The `spatial/`
layer continues to host the 6D-algebra value types
(`Motion`/`Force`/`Inertia`) and re-exports `SE3`/`SO3` for
convenience. A `Pose` alias points at `SE3` for users who prefer
the geometric name.

> **Why `lie/`, not `spatial/`?** SO(3) and SE(3) are Lie groups —
> conceptually they belong with the rest of the Lie machinery.
> `spatial/` already depends on `lie/` (for `Motion.se3_action`,
> `Inertia.apply`, etc.); putting Lie group classes in `spatial/`
> would invert the layering story or add a circular dependency.
> An earlier draft of this proposal sited the classes in `spatial/`;
> the gpt_plan review caught the layering error.

> **Field name `.tensor`, not `.data`.** `Data` is itself a class in
> this library; `data.data` and `pose.data` next to one another read
> badly. New typed wrappers use `.tensor`. The existing
> `Motion.data` / `Force.data` / `Inertia.data` keep their name for
> now and pick up a `.tensor` alias; the rename is a low-priority
> cleanup, not a load-bearing decision.

```python
# src/better_robot/lie/types.py
from dataclasses import dataclass
import torch
from . import se3 as _se3
from . import so3 as _so3

@dataclass(frozen=True)
class SE3:
    """Rigid transform in SE(3), stored as ``(..., 7)``
    ``[tx, ty, tz, qx, qy, qz, qw]``.

    The tensor attribute is a plain ``torch.Tensor`` — autograd, vmap,
    and ``torch.compile`` see a tensor, not a custom subclass. All
    operations go through ``better_robot.lie.se3.*`` (which itself
    routes through the active backend per
    [Proposal 02](02_backend_abstraction.md)), so the eventual Warp
    swap is invisible to users of this type.
    """
    tensor: torch.Tensor      # (..., 7)

    # ---- factories ----

    @classmethod
    def identity(cls, *, batch_shape=(), device=None, dtype=torch.float32) -> "SE3":
        return cls(_se3.identity(batch_shape=batch_shape, device=device, dtype=dtype))

    @classmethod
    def from_translation(cls, t: torch.Tensor) -> "SE3": ...

    @classmethod
    def from_rotation(cls, r: SO3) -> "SE3": ...

    @classmethod
    def from_axis_angle(cls, axis: torch.Tensor, angle: torch.Tensor) -> "SE3":
        return cls(_se3.from_axis_angle(axis, angle))

    @classmethod
    def from_matrix(cls, m: torch.Tensor) -> "SE3":  # (..., 4, 4)
        ...

    @classmethod
    def from_components(cls, t: torch.Tensor, r: SO3) -> "SE3": ...

    @classmethod
    def exp(cls, xi: torch.Tensor) -> "SE3":
        return cls(_se3.exp(xi))

    # ---- properties ----

    @property
    def translation(self) -> torch.Tensor:    # (..., 3)
        return self.tensor[..., :3]

    @property
    def rotation(self) -> SO3:
        return SO3(self.tensor[..., 3:7])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape[:-1])

    # ---- group ops ----

    def compose(self, other: "SE3") -> "SE3":
        return SE3(_se3.compose(self.tensor, other.tensor))

    def inverse(self) -> "SE3":
        return SE3(_se3.inverse(self.tensor))

    def log(self) -> torch.Tensor:            # (..., 6)
        return _se3.log(self.tensor)

    def adjoint(self) -> torch.Tensor:        # (..., 6, 6)
        return _se3.adjoint(self.tensor)

    def adjoint_inv(self) -> torch.Tensor:    # (..., 6, 6)
        return _se3.adjoint_inv(self.tensor)

    def normalize(self) -> "SE3":
        return SE3(_se3.normalize(self.tensor))

    def to_matrix(self) -> torch.Tensor:      # (..., 4, 4)
        ...

    # ---- actions ----

    def act_point(self, p: torch.Tensor) -> torch.Tensor:   # (..., 3)
        return _se3.act(self.tensor, p)

    def act_motion(self, m: "Motion") -> "Motion":          # adjoint action
        return m.se3_action(self.tensor)

    def act_force(self, f: "Force") -> "Force":
        return f.se3_action(self.tensor)

    def act_inertia(self, I: "Inertia") -> "Inertia":
        return I.se3_action(self.tensor)

    # ---- unambiguous operators ----

    def __matmul__(self, other):
        if isinstance(other, SE3):
            return self.compose(other)
        if isinstance(other, torch.Tensor) and other.shape[-1] == 3:
            return self.act_point(other)
        return NotImplemented

    # NB: deliberately no __mul__ — the brax footgun (leafwise scale).
    # NB: deliberately no __invert__ ('~T') — sigil-overloads age badly.
```

`SO3` mirrors this pattern at `(..., 4)`.

## Storage in `Model` / `Data` stays raw tensors

`Model.joint_placements`, `Data.joint_pose_world`,
`Data.frame_pose_world` keep their current `torch.Tensor` shapes
(`(..., njoints, 7)`, etc.). Boxing every entry into an `SE3`
instance for storage would defeat batching. The typed wrapper is a
**user-facing convenience**: returned from public methods, accepted
as an input, and otherwise free to construct on demand:

```python
# Internal hot path — raw tensors, no boxing.
data.joint_pose_world = compose_all(...)        # (B, njoints, 7)

# Public method — return the boxed type.
def get_frame_pose(model, data, name: str) -> SE3:
    fid = model.frame_id(name)
    return SE3(data.frame_pose_world[..., fid, :])

# User code — clean.
T_ee = get_frame_pose(model, data, "panda_hand")
T_target = SE3.from_axis_angle(z_axis, math.pi / 4) @ SE3.from_translation(p)
err = (T_ee.inverse() @ T_target).log()
```

A small pair of helpers (`SE3.stack(items)` /
`SE3.unbox_to_tensor(t)`) makes round-trips between the typed and raw
representations explicit.

### Re-export from `spatial/`

`spatial/__init__.py` re-exports `SE3`, `SO3`, and `Pose` so that
`from better_robot.spatial import SE3` continues to work for users
who think of poses as living next to `Motion`/`Force`/`Inertia`.
The canonical home is still `better_robot.lie.types`; the
re-export is a convenience, not a second source of truth.

## Operator policy

A single operator: `__matmul__` (`@`). The semantics are
unambiguous:

| Right operand | Result | Maps to |
|---------------|--------|---------|
| `SE3` | composition | `lie.se3.compose` |
| `torch.Tensor` of shape `(..., 3)` | rigid action on a point | `lie.se3.act` |
| anything else | `NotImplemented` | dispatch up the chain |

The brax `__mul__ = leafwise_scale` mistake is impossible because
`__mul__` is **not implemented**. Sigil overloads (`~T`, `**T`) are
not implemented either — they age badly across teams.

## Why this preserves what's already right

| Concern | Solution in the proposal |
|---------|--------------------------|
| `lie/` must stay autograd-clean and `torch.compile`-friendly | The `lie/` functional facade (`lie.se3`, `lie.so3`) is unchanged. `SE3.compose` *delegates* to `lie.se3.compose`. Hot paths still call the functional layer directly. |
| Backend swap (PyPose → Warp) must be invisible | The wrapper holds a tensor and calls the functional layer, which itself routes through the backend protocol from [Proposal 02](02_backend_abstraction.md). The swap is in `backends/`, not in the typed wrappers. |
| Tensor subclass fragility | `SE3` is a `@dataclass(frozen=True)`. It is **not** a `torch.Tensor`. PyTorch ops on `T.tensor` work; ops on `T` itself fall back to dataclass equality. |
| Top-level public API ceiling | Counted in [Proposal 06](06_public_api_audit.md). The recommendation there is to **expose typed Lie values via `better_robot.lie.SE3`** (submodule access) rather than top-level — top-level stays lean. |
| `spatial/` already has the named-method pattern | This proposal extends it. `SE3.act_motion(m)` calls `Motion.se3_action(T)` — a single canonical implementation. |

## Doc 03 amendments

[03_LIE_AND_SPATIAL.md §9](../../design/03_LIE_AND_SPATIAL.md) currently says:

> So `lie/` stays **functional over plain tensors** … and `spatial/`
> provides *shallow* value-type wrappers with explicit named methods …
> The kinematics layer works directly on tensors; `dynamics` uses
> `Motion/Force/Inertia` for readability.

That is preserved as the policy for `spatial/`'s 6D-algebra value
types. The amendment adds:

> `lie/types.py` exposes typed Lie-group wrappers (`SE3`, `SO3`,
> `Pose`) for user-facing call sites. They wrap the same raw
> `(..., 7)` / `(..., 4)` tensors that `Model` and `Data` store and
> delegate every operation to the functional Lie layer. They are not
> subclasses of `torch.Tensor`. Operator support is exactly
> `__matmul__` (compose / point-action); `__mul__` is deliberately
> absent. `spatial/__init__.py` re-exports them as a convenience.

## Files that change

```
lie/types.py                   new — SE3, SO3, Pose dataclasses
lie/__init__.py                export SE3, SO3, Pose from lie.types
spatial/__init__.py            re-export SE3, SO3, Pose for convenience
kinematics/__init__.py         add get_frame_pose() returning SE3
kinematics/jacobian.py         get_frame_jacobian — accept ReferenceFrame enum
                               (cross-references Proposal 04)
io/build_model.py              accept SE3 origin in IRJoint, IRFrame
data_model/frame.py            joint_placement field can accept SE3 in __init__,
                               but stores raw (7,) tensor
__init__.py                    no top-level export by default — users reach
                               typed values via better_robot.lie.SE3 etc.
                               (see Proposal 06 for the top-level decision)
```

A `SE3.__torch_function__` override is **not** added — the wrapper
intentionally does not present as a tensor.

## Migration plan

This change is additive. There is no deprecation cycle.

1. **Phase 1 — land the types.** `SE3`, `SO3` shipped as
   non-load-bearing wrappers in `lie/types.py`; tests live in
   `tests/lie/test_types_se3.py`, `test_types_so3.py`. Internal code
   unchanged. Equivalence tests assert that every typed method
   produces the same tensor as the matching `lie.se3.*` /
   `lie.so3.*` call.
2. **Phase 2 — adopt at boundaries.** Public functions that return a
   pose return `SE3`; public functions that return a quaternion
   return `SO3`. `forward_kinematics`, `solve_ik` continue to return
   `Data` and `IKResult` (which hold raw tensors), but expose
   convenience methods (`Data.frame_pose(name) -> SE3`,
   `IKResult.frame_pose(name) -> SE3`).
3. **Phase 3 — update the examples.** `examples/01_basic_ik.py` and
   `examples/02_g1_ik.py` use the typed API. Doc tutorials follow.
4. **Phase 4 (deferred) — `.tensor` rename for the spatial 6D types.**
   `Motion.data` / `Force.data` / `Inertia.data` pick up a `.tensor`
   alias; the rename is mechanical but not load-bearing — schedule
   independently of the public API freeze.

No internal hot path is touched. The performance budgets in
[14_PERFORMANCE.md §1](../../conventions/14_PERFORMANCE.md) are
unchanged because the boxes are constructed only at the public
boundary.

## Tradeoffs

| For | Against |
|-----|---------|
| Reads like the math (`T_inv @ T @ p`). | Two ways to do the same thing — typed methods *and* `lie.se3.compose`. We mitigate by documenting "use the typed API at boundaries, the functional API in hot loops". |
| Catches `T_ee.compose(quaternion_tensor)` at type-check time (different dataclasses). | More files in `spatial/`. |
| Closes the ergonomic gap with Pinocchio's `SE3` and Drake's `RigidTransform`. | Adds three names to `__all__` (see Proposal 06). |
| Makes the [Proposal 03] PyPose replacement invisible to users. | None significant. |

## Acceptance criteria

- `lie.types.SE3`, `lie.types.SO3` exist as `@dataclass(frozen=True)`
  and pass a unit-test suite that exercises every method against the
  functional `lie.se3.*` / `lie.so3.*` API.
- `from better_robot.lie import SE3, SO3, Pose` works.
- `from better_robot.spatial import SE3, SO3, Pose` works (re-export).
- The top-level `import better_robot as br; br.SE3` decision is
  deferred to [Proposal 06](06_public_api_audit.md).
- `examples/01_basic_ik.py` and `examples/02_g1_ik.py` use the typed
  API for every pose-valued variable.
- No file outside `lie/`, `spatial/`, and the corresponding test
  directories imports `pypose` (already enforced by
  `test_layer_dependencies.py`).
- `T1 @ T2`, `T @ p`, `T.inverse()`, `T.log()`, `SE3.exp(xi)`,
  `SE3.from_axis_angle(...)`, `SE3.identity(...)` all match the
  functional API to `1e-6` (fp32) / `1e-12` (fp64) on randomised
  batches.
- `T * 2.0` raises `TypeError`. `~T` raises `TypeError`.
- `torch.func.jacrev(lambda t: SE3(t).log())` runs and matches
  central FD inside the contract test for residual Jacobians.

## Out of scope here

- Subclassing `torch.Tensor` is rejected by Proposal 03 (PyPose
  replacement) for the same reasons noted above.
- Replacing the `lie/` functional API is out of scope. It stays.
- Adding typed `Twist` / `Wrench` aliases for `Motion` / `Force` is
  not proposed — those types already work.
