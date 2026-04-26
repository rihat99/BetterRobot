# 03 · Lie groups and spatial algebra

This layer is the *mathematical* substrate of BetterRobot. It is the only
place that imports a Lie-group backend (PyPose today, Warp later, pure
PyTorch as the default once the swap is green — see §10). Everything above
this layer (`data_model`, `kinematics`, `dynamics`, …) sees `torch.Tensor`
and plain functions for hot paths, plus typed value classes
(`SE3`, `SO3`, `Pose`, `Motion`, `Force`, `Inertia`) at the user-facing
boundary.

> **Naming.** Lie-algebra notation (`Jr`, `Jl`, `hat`, `vee`, `ad`, `Ad`,
> `exp`, `log`) is universal in Barfoot / Chirikjian and stays. The
> **public function** names are English (`right_jacobian_se3`,
> `hat_map_so3`, …); the short forms remain as aliases. See
> [13_NAMING.md §1.3](../conventions/13_NAMING.md) and §2.6.

> **Numerical stability.** Input quaternion norm, small-angle
> Taylor-expansion boundaries, and singularity fallbacks are specified
> in [17_CONTRACTS.md §3](../conventions/17_CONTRACTS.md) — the source of truth for
> what "accurate enough" means.

> **Backend boundary.** `lie/se3.py`, `lie/so3.py`, and `lie/tangents.py`
> route SE3/SO3 ops through the active `Backend` Protocol from
> [10_BATCHING_AND_BACKENDS.md](10_BATCHING_AND_BACKENDS.md). Functions
> accept an explicit `backend=` kwarg; `default_backend()` is the
> convenience fall-through. The PyPose path (`_pypose_backend.py`) is
> reachable via `BR_LIE_BACKEND=pypose` for regression testing during
> the swap; the default is the pure-PyTorch backend
> (`_torch_native_backend.py`).

## 1. Goals

1. **One clean SE3/SO3 facade.** A handful of pure functions in `lie/se3.py`
   and `lie/so3.py` that hide the backend.
2. **Typed Lie value classes** (`SE3`, `SO3`, `Pose`) in `lie/types.py` for
   user-facing call sites where readability matters
   (`T_world_ee = T_world_arm @ T_arm_ee`, `T.inverse() @ p`, `T.log()`).
   Hot paths still call the functional API on raw tensors.
3. **Named value types for spatial algebra.** `Motion`, `Force`, `Inertia`
   with methods like `.act(other)`, `.cross(other)`, `.log()`, `.exp()`.
   Syntax sugar is optional; named methods are the canonical form.
4. **Unified tangent Jacobians.** `Jr`, `Jl`, `Jr_inv`, `Jl_inv` for SE3 and
   SO3 available as named functions so that analytic Jacobians in
   `kinematics/jacobian.py` can stop approximating `Jlog ≈ I`.
5. **Backend-swappable.** Today: PyPose powers all SE3/SO3 math through
   `lie/_pypose_backend.py` behind the `lie.se3` / `lie.so3` facades. The
   pure-PyTorch backend lands in P10 (Phase L); after L-C it becomes the
   default and PyPose is reachable under `BR_LIE_BACKEND=pypose` for one
   release, then deleted in v1.2 (L-D). Tomorrow: Warp. Zero API change at
   the call sites; the swap is one file in
   `backends/torch_native/lie_ops.py`.

## 2. Directory layout

```
src/better_robot/
├── lie/
│   ├── __init__.py            # re-exports SE3/SO3/Pose, plus the functional API
│   ├── types.py               # SE3, SO3, Pose — frozen dataclasses around tensors
│   ├── se3.py                 # SE3 group operations (functional)
│   ├── so3.py                 # SO3 group operations (functional)
│   ├── tangents.py            # Jr, Jl, hat, vee, small-angle handling (pure-torch)
│   ├── _torch_native_backend.py  # (planned, P10/L-A) pure-PyTorch SE3/SO3 kernels — becomes default after L-C
│   └── _pypose_backend.py     # current default; reachable via BR_LIE_BACKEND=pypose after L-C; deleted in L-D
└── spatial/
    ├── __init__.py            # re-exports SE3/SO3/Pose; Motion, Force, Inertia, Symmetric3
    ├── motion.py              # Motion (6D twist)
    ├── force.py               # Force (6D wrench)
    ├── inertia.py             # Inertia — typed view; storage is (..., 10) on Model.body_inertias
    ├── symmetric3.py          # advanced helper (lower-triangular packing); not top-level
    └── ops.py                 # ad, Ad, cross, act
```

## 3. `lie/se3.py` — SE3 group operations

### Convention

- Storage: `(..., 7)` tensor, `[tx, ty, tz, qx, qy, qz, qw]` (scalar-last,
  PyPose-native). All tangent vectors are `(..., 6)` `[vx, vy, vz, wx, wy, wz]`.
- Everything is functional. No `SE3` class. The facade is a module of free
  functions. Users who want a classy interface import from `spatial/` instead.

### API

```python
# src/better_robot/lie/se3.py
import torch
from . import _pypose_backend as _pp

def identity(*, batch_shape=(), device=None, dtype=torch.float32) -> torch.Tensor:
    """Return an SE3 identity with the given batch shape."""

def compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SE3 composition. Shapes broadcast: a: (..., 7), b: (..., 7) → (..., 7)."""

def inverse(t: torch.Tensor) -> torch.Tensor:
    """SE3 inverse. (..., 7) → (..., 7)."""

def log(t: torch.Tensor) -> torch.Tensor:
    """SE3 → se3 tangent. (..., 7) → (..., 6)."""

def exp(v: torch.Tensor) -> torch.Tensor:
    """se3 tangent → SE3. (..., 6) → (..., 7)."""

def act(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply SE3 to a point. t: (..., 7), p: (..., 3) → (..., 3)."""

def adjoint(t: torch.Tensor) -> torch.Tensor:
    """6×6 adjoint matrix. (..., 7) → (..., 6, 6)."""

def adjoint_inv(t: torch.Tensor) -> torch.Tensor:
    """Inverse adjoint: Ad(T^{-1}). Faster than inverting the adjoint."""

def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Pure-rotation SE3 from axis-angle."""

def from_translation(axis: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Pure-translation SE3."""

def normalize(t: torch.Tensor) -> torch.Tensor:
    """Re-normalize quaternion part to project back onto SE3."""

def apply_base(base: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Compose a base transform with (..., N, 7) link poses."""
```

No function here knows whether the active backend is pure-PyTorch, PyPose,
or Warp; all of them forward to `current_backend().lie.*`. The torch-native
backend (default) routes to `_torch_native_backend`; the PyPose path
(`_pypose_backend`) is reachable via `BR_LIE_BACKEND=pypose` for the swap
window. A Warp backend supplies a parallel `backends/warp/lie_ops.py`.

## 4. `lie/so3.py`

Same pattern, `(..., 4)` quaternions `[qx, qy, qz, qw]`, tangents `(..., 3)`.

```python
identity, compose, inverse, log, exp, act, adjoint,
from_matrix, to_matrix, from_axis_angle, normalize
```

## 5. `lie/tangents.py` — right/left Jacobians

```python
def right_jacobian_so3(omega: torch.Tensor) -> torch.Tensor:
    """Jr(ω) — right Jacobian of SO3 exp. (..., 3) → (..., 3, 3)."""

def right_jacobian_inv_so3(omega: torch.Tensor) -> torch.Tensor: ...
def left_jacobian_so3(omega: torch.Tensor)  -> torch.Tensor: ...
def left_jacobian_inv_so3(omega: torch.Tensor) -> torch.Tensor: ...

def right_jacobian_se3(xi: torch.Tensor)    -> torch.Tensor:
    """Jr(ξ) — right Jacobian of SE3 exp. (..., 6) → (..., 6, 6)."""

def right_jacobian_inv_se3(xi: torch.Tensor) -> torch.Tensor: ...
def left_jacobian_se3(xi: torch.Tensor)      -> torch.Tensor: ...
def left_jacobian_inv_se3(xi: torch.Tensor)  -> torch.Tensor: ...

def hat_se3(xi: torch.Tensor) -> torch.Tensor:  # (..., 6) → (..., 4, 4)
def vee_se3(X: torch.Tensor)  -> torch.Tensor:  # (..., 4, 4) → (..., 6)
def hat_so3(w: torch.Tensor)  -> torch.Tensor:  # (..., 3) → (..., 3, 3)
def vee_so3(W: torch.Tensor)  -> torch.Tensor:  # (..., 3, 3) → (..., 3)
```

These are what `kinematics/jacobian.py` will use to drop the current
`jlog ≈ I` approximation in the pose residual Jacobian. They need safe
small-angle handling (Taylor expansion near ‖ω‖ = 0).

## 6. Backend modules — the private bridges

All Lie-backend imports live in two private modules, swappable by env var
and by explicit `backend=` kwarg:

```
lie/_torch_native_backend.py  # (planned, P10/L-A) default after L-C — pure PyTorch
lie/_pypose_backend.py        # current default; reachable via BR_LIE_BACKEND=pypose after L-C
```

After L-A, `backends/torch_native/lie_ops.py` will route through one of
them based on `BR_LIE_BACKEND`. The torch-native backend has a single
short closed-form for each operation (see §10 for the SE3 exp / log
formulas Barfoot §7.85, Murray-Li-Sastry §A.5); the PyPose backend is
preserved for one release as a regression oracle for the swap (see
[PYPOSE_ISSUES.md](../status/PYPOSE_ISSUES.md)).

Two contract tests guard the boundary:

- `tests/contract/test_layer_dependencies.py` — only `lie/_pypose_backend.py`
  may `import pypose`.
- `tests/contract/test_backend_boundary.py` — only `lie/`, `kinematics/`,
  `dynamics/` (and the `backends/<name>/` packages) cross the backend
  Protocol; everything above composes through them.

The torch-native backend is required to:

1. Pass `torch.autograd.gradcheck` at fp64 with `atol=1e-8, rtol=1e-6` on
   randomised unit-quaternion inputs for every `LieOps` member.
2. Match the PyPose forward pass to `1e-6` (fp32) / `1e-12` (fp64) on the
   existing test fixtures.
3. Lock behaviour at `θ ∈ {0, π/2, π − 1e-6}` for both `log` and `exp`
   via `tests/lie/test_singularities.py`.

The implementation sketch (Barfoot §7.85, *State Estimation for Robotics*):
the SE3 left-Jacobian is `V(ω) = I + b·W + c·W²` with
`b = (1 − cos θ) / θ²` and `c = (θ − sin θ) / θ³`. Both expand cleanly
near `θ = 0` (`b ≈ 1/2 − θ²/24`, `c ≈ 1/6 − θ²/120`) — those Taylor leads
are checked explicitly by the singularity tests. Do **not** transcribe; an
earlier draft missed the `1/6` leading term in the `c` Taylor and wrote
`V = a·I + b·W + c·W²`, both of which were wrong. Re-derive from a
primary source before committing.

## 7. Typed value classes — `lie/types.py` and `spatial/`

`lie/` is functional. **`lie/types.py`** adds typed `SE3`, `SO3`, `Pose`
dataclasses on top — frozen `@dataclass`es around a `torch.Tensor`,
**not** subclasses of `torch.Tensor`. They expose the same operations as
the functional API as named methods plus a single unambiguous operator
(`@`):

```python
# src/better_robot/lie/types.py
from dataclasses import dataclass
import torch
from . import se3 as _se3, so3 as _so3

@dataclass(frozen=True)
class SE3:
    """Rigid transform in SE(3), stored as ``(..., 7)``
    ``[tx, ty, tz, qx, qy, qz, qw]``.

    The ``tensor`` attribute is a plain ``torch.Tensor`` — autograd, vmap,
    and ``torch.compile`` see a tensor, not a custom subclass. All
    operations route through the functional ``lie.se3.*`` API.
    """
    tensor: torch.Tensor                              # (..., 7)

    @classmethod
    def identity(cls, *, batch_shape=(), device=None, dtype=torch.float32) -> "SE3": ...
    @classmethod
    def from_translation(cls, t: torch.Tensor) -> "SE3": ...
    @classmethod
    def from_rotation(cls, r: "SO3") -> "SE3": ...
    @classmethod
    def from_axis_angle(cls, axis: torch.Tensor, angle: torch.Tensor) -> "SE3": ...
    @classmethod
    def from_components(cls, t: torch.Tensor, r: "SO3") -> "SE3": ...
    @classmethod
    def from_matrix(cls, m: torch.Tensor) -> "SE3": ...
    @classmethod
    def exp(cls, xi: torch.Tensor) -> "SE3": ...

    @property
    def translation(self) -> torch.Tensor: ...        # (..., 3)
    @property
    def rotation(self) -> "SO3": ...
    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    def compose(self, other: "SE3") -> "SE3": ...
    def inverse(self) -> "SE3": ...
    def log(self) -> torch.Tensor: ...                # (..., 6)
    def adjoint(self) -> torch.Tensor: ...            # (..., 6, 6)
    def adjoint_inv(self) -> torch.Tensor: ...
    def normalize(self) -> "SE3": ...
    def to_matrix(self) -> torch.Tensor: ...          # (..., 4, 4)

    def act_point(self, p: torch.Tensor) -> torch.Tensor: ...   # (..., 3)
    def act_motion(self, m: "Motion") -> "Motion": ...
    def act_force (self, f: "Force") -> "Force":  ...
    def act_inertia(self, I: "Inertia") -> "Inertia": ...

    def __matmul__(self, other):
        # SE3 @ SE3 → compose;  SE3 @ Tensor[..., 3] → act_point
        ...

    # NB: deliberately no __mul__ (the brax footgun: leafwise scale).
    # NB: deliberately no __invert__ (`~T`); sigil overloads age badly.
```

`SO3` mirrors at `(..., 4)`. `Pose` is an alias for `SE3` for users who
prefer the geometric name. `lie/__init__.py` re-exports
`SE3`, `SO3`, `Pose` from `lie.types`; `spatial/__init__.py` re-exports
them as a convenience.

**Storage stays raw.** `Model.joint_placements` is `(njoints, 7)` tensor;
`Data.joint_pose_world` is `(B..., njoints, 7)` tensor. Boxing every entry
into an `SE3` instance for storage would defeat batching. The typed
wrapper is *user-facing*: returned from `Data.frame_pose(name)` /
`Data.joint_pose(joint_id)` / `IKResult.frame_pose(name)`; accepted as
input on every method that takes an SE3.

`spatial/` exposes named value types mirroring Pinocchio's
`Motion`/`Force`/`Inertia`. These are *thin* dataclasses around tensors,
with pure-function method bodies. No operator overloading on generic
containers; explicit named methods are the primary interface.

```python
# src/better_robot/spatial/motion.py
from dataclasses import dataclass
import torch
from .. import lie

@dataclass(frozen=True)
class Motion:
    """6D twist: linear + angular velocity.

    Stored as a single (..., 6) tensor. Methods return new `Motion` instances.
    """
    data: torch.Tensor   # (..., 6), [vx, vy, vz, wx, wy, wz]

    @classmethod
    def zero(cls, *, batch_shape=(), device=None, dtype=torch.float32) -> "Motion": ...

    @property
    def linear(self)  -> torch.Tensor: ...    # (..., 3)
    @property
    def angular(self) -> torch.Tensor: ...    # (..., 3)

    def cross_motion(self, other: "Motion") -> "Motion":
        """Motion × Motion (spatial acceleration cross)."""

    def cross_force(self, other: "Force")  -> "Force":
        """Motion × Force (spatial force cross / `ad*`)."""

    def se3_action(self, T: torch.Tensor) -> "Motion":
        """Apply SE3 transform to the twist (adjoint action)."""

    def compose(self, other: "Motion") -> "Motion":
        """Element-wise sum (valid because Motion is a vector space)."""

    def __neg__(self) -> "Motion": ...
    def __add__(self, other: "Motion") -> "Motion": ...
    def __sub__(self, other: "Motion") -> "Motion": ...
    # NB: no __mul__ — it is ambiguous between scale and cross.
```

Operators `+` and `-` are safe (vector space). `*` is **not** implemented to
avoid the brax footgun where `*` is a leafwise scale instead of SE3
composition. Users call `.cross_motion(...)`, `.se3_action(...)`, etc.

### `spatial/force.py`

Dual structure. Same shape. **`cross_motion` raises `NotImplementedError`
deliberately** — see §7.X below.

### `spatial/inertia.py`

```python
@dataclass(frozen=True)
class Inertia:
    """Spatial inertia of a rigid body — typed view over a packed
    ``(..., 10)`` tensor.

    Packing: ``[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``.

    Storage of inertias lives on ``Model.body_inertias`` as a packed
    ``(nbodies, 10)`` tensor. The typed ``Inertia`` wrapper is the
    user-facing accessor, obtained via ``model.body_inertia(body_id)``,
    and is what library functions *return*. Functions that take an
    ``Inertia`` may also accept the raw ``(..., 10)`` tensor; functions
    that return an ``Inertia`` always return the typed wrapper. The
    same rule applies to ``SE3``, ``SO3``, ``Motion``, ``Force``.
    """
    data: torch.Tensor   # (..., 10) — also exposed as `.tensor` alias

    @property
    def mass(self) -> torch.Tensor: ...
    @property
    def com(self)  -> torch.Tensor: ...
    @property
    def inertia_matrix(self) -> torch.Tensor: ...    # Symmetric3 expanded to (..., 3, 3)

    @classmethod
    def from_sphere(cls, mass, radius) -> "Inertia": ...
    @classmethod
    def from_box(cls, mass, size) -> "Inertia": ...
    @classmethod
    def from_capsule(cls, mass, radius, length) -> "Inertia": ...
    @classmethod
    def from_ellipsoid(cls, mass, radii) -> "Inertia": ...
    @classmethod
    def from_mass_com_sym3(cls, mass, com, sym3) -> "Inertia": ...
    @classmethod
    def from_mass_com_matrix(cls, mass, com, I3x3) -> "Inertia":
        """Convenience: pack a (3,3) symmetric inertia matrix."""

    def se3_action(self, T: "SE3 | torch.Tensor") -> "Inertia":
        """Transform the inertia by an SE3 (typed or raw (..., 7))."""

    def apply(self, v: "Motion") -> "Force":
        """I * v — inertia times twist = spatial momentum."""

    def add(self, other: "Inertia") -> "Inertia":
        """Composite-rigid-body inertia addition."""
```

Factories (`from_sphere`, `from_box`, …) steal directly from Pinocchio's
`InertiaTpl` named constructors. **Method signatures accept the typed
wrapper *or* the raw tensor; methods always return the typed wrapper.**
This rules out the awkward middle state where users have to unbox manually.

`Symmetric3` (in `spatial/symmetric3.py`) is the lower-triangular `(..., 6)`
packed helper used inside `Inertia` and (future) centroidal computations.
It is **reachable** as `from better_robot.spatial import Symmetric3` but is
**not** in the top-level `__all__` — documented as an "advanced helper for
callers who need direct access to the packed lower-triangular layout", not
the headline API.

### 7.X Force × Motion duality — what is implemented and what is not

The standard spatial-algebra crosses are:

| Operator | Implemented as | Output | Reference |
|----------|----------------|--------|-----------|
| `Motion × Motion` | `Motion.cross_motion(other)` | `Motion` | Featherstone, eq. (2.30); spatial acceleration cross |
| `Motion × Force`  | `Motion.cross_force(other)`  | `Force`  | Featherstone, eq. (2.31); the `ad*` operator |

`Force × Motion` is **not** a standard operation. The dual that exists is
`Motion.cross_force` above. Adding a `Force.cross_motion` for "symmetry"
would teach users a non-textbook algebra; worse, the convention they would
*reach for* in muscle-driven dynamics derivatives is algorithm-specific,
not type-level.

`Force.cross_motion` therefore stays as `NotImplementedError` deliberately.
The error message names `Motion.cross_force` as the standard dual:

```
NotImplementedError:
    Force × Motion is not a standard spatial-algebra operation.
    The dual that exists is Motion.cross_force(force) → Force,
    which is the `ad*` operator (Featherstone eq. 2.31). If you have
    a concrete dynamics derivation that needs a Force-acting-on-Motion
    operation, give it an algorithm-specific name; do not rely on
    type-level symmetry.
```

If a future dynamics algorithm ever needs the operation, give it an
algorithm-specific name (`muscle_force_jacobian_pull(...)` or whatever),
*not* a type-level method that pretends `Force × Motion` is a standard
spatial-algebra operator.

## 8. `spatial/ops.py` — shared operators

```python
def ad(v: Motion)  -> torch.Tensor:      # (..., 6, 6)  motion cross operator
def ad_star(f: Force) -> torch.Tensor:   # (..., 6, 6)  force cross operator
def cross_mm(a: Motion, b: Motion) -> Motion:   # same as a.cross_motion(b)
def cross_mf(a: Motion, b: Force)  -> Force:
def act_motion(T: torch.Tensor, m: Motion) -> Motion:
def act_force(T: torch.Tensor, f: Force)  -> Force:
def act_inertia(T: torch.Tensor, I: Inertia) -> Inertia:
```

## 9. Why this split instead of one big class

Pinocchio's `SE3` class with overloaded `operator*` (on `SE3`, `Motion`,
`Force`, `Inertia`) is elegant in C++ but turns into footguns in Python:

- `Base.__mul__` ambiguity (brax lesson)
- hidden autograd graph surgery when value types wrap a `torch.Tensor`
- `__torch_function__` subclass drift on tensor subclasses (PyPose lesson)

So `lie/` stays **functional over plain tensors** (backend-swappable,
autograd-clean, trivially jitted), and `spatial/` provides *shallow*
value-type wrappers with explicit named methods for code that reads cleaner
with them (notably `dynamics/`). The kinematics layer works directly on
tensors; `dynamics` uses `Motion/Force/Inertia` for readability.

## 10. Backend migration plan — PyPose retire, Warp land

Three swaps live in `backends/torch_native/lie_ops.py`. The PyPose
retirement runs as Phase P10 of
[UPDATE_PHASES.md](../UPDATE_PHASES.md), with sub-phases L-A → L-E
that are also tracked in
[PYPOSE_ISSUES.md](../status/PYPOSE_ISSUES.md):

**L-A → L-C — pure-PyTorch becomes default.**

1. Land `lie/_torch_native_backend.py` (~250 LOC of torch ops); pass
   `gradcheck` at fp64.
2. CI runs the full suite under both `BR_LIE_BACKEND=pypose` and
   `BR_LIE_BACKEND=torch_native` for one release.
3. Flip the default to torch-native; `kinematics.residual_jacobian`'s
   default switches from finite-diff to `torch.func.jacrev`. FD is
   retained as opt-in (`JacobianStrategy.FINITE_DIFF`).

**L-D / L-E — drop PyPose.**

4. `lie/_pypose_backend.py` is deletable in v1.2; `BR_LIE_BACKEND` env
   var goes away with it. `pyproject.toml` drops the `pypose` dep.

**Warp — when ready.**

5. Add `backends/warp/lie_ops.py` (and corresponding `kinematics`/
   `dynamics` ops) implementing the same `LieOps` Protocol.
6. `set_backend("warp")` then routes through it; user code is unchanged.
7. The full Warp landing isolates to `backends/warp/` (the Protocol pins
   the surface) but is *not* a single file — Warp owns its own kernel
   cache, stream bridge, autograd wrappers, graph-capture policy, and
   parity tests against the torch-native baseline.

Constraints during the Warp swap:

- Users never import warp.
- `torch.Tensor` in, `torch.Tensor` out. The Warp adapter owns the
  `torch.autograd.Function` wrapper with an analytic adjoint in `backward`.
- No `wp.Tape` leakage; `set_module_options({"enable_backward": False})` in
  every warp module, as mjwarp does.
- All Warp bridge code lives in `backends/warp/bridge.py`.
- Both `BR_LIE_BACKEND=torch_native|warp` and the explicit
  `lie.se3.compose(a, b, backend=get_backend("warp"))` form work; tests
  verify both.

Constraints during that swap:

- Users never import warp.
- `torch.Tensor` in, `torch.Tensor` out. The Warp adapter owns the
  `torch.autograd.Function` wrapper with an analytic adjoint in `backward`.
- No `wp.Tape` leakage; `set_module_options({"enable_backward": False})` in
  every warp module, as mjwarp does.
- All Warp bridge code lives in `backends/warp/bridge.py`; `lie/_warp_backend`
  just calls through. This keeps the Lie facade oblivious to device/buffer
  management.

## 11. What gets deleted

- `src/better_robot/math/se3.py`, `so3.py`, `spatial.py`, `transforms.py` →
  rewritten into `lie/` and `spatial/`. The new `lie/se3.py` is roughly the
  current `math/se3.py` with names shortened and the module layout fixed.
- Direct `import pypose` statements in `costs/`, `algorithms/`, `tasks/` →
  routed through the `Backend` Protocol via `lie/` and `spatial/` instead.
- `math/transforms.py` (viser quaternion converters) → moves to
  `viewer/helpers.py` where it belongs.
- The `solvers/levenberg_marquardt_pypose.py` benchmark solver → deleted
  outright. It exists only to benchmark pypose, which is not a use case we
  need to keep alive.
- `lie/_pypose_backend.py` itself → kept under `BR_LIE_BACKEND=pypose` until
  v1.2; deleted in v1.2 along with the env var, the contract test rule that
  forbade `import pypose` outside it, and the `pypose` dependency in
  `pyproject.toml`.
