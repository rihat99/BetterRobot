# 03 · Lie groups and spatial algebra

This layer is the *mathematical* substrate of BetterRobot. It is the only
place that imports `pypose`. Eventually it is the only place that will import
`warp`. Everything above this layer (`data_model`, `kinematics`, `dynamics`,
…) sees `torch.Tensor` and plain functions.

> **Naming.** Lie-algebra notation (`Jr`, `Jl`, `hat`, `vee`, `ad`, `Ad`,
> `exp`, `log`) is universal in Barfoot / Chirikjian and stays. The
> **public function** names are English (`right_jacobian_se3`,
> `hat_map_so3`, …); the short forms remain as aliases. See
> [13_NAMING.md §1.3](13_NAMING.md) and §2.6.

> **Numerical stability.** Input quaternion norm, small-angle
> Taylor-expansion boundaries, and singularity fallbacks are specified
> in [17_CONTRACTS.md §3](17_CONTRACTS.md) — the source of truth for
> what "accurate enough" means.

## 1. Goals

1. **One clean SE3/SO3 facade.** Replace scattered pypose calls with a
   handful of pure functions that hide the backend.
2. **Named value types for spatial algebra.** `Motion`, `Force`, `Inertia`
   with methods like `.act(other)`, `.cross(other)`, `.log()`, `.exp()`.
   Syntax sugar is optional; named methods are the canonical form.
3. **Unified tangent Jacobians.** `Jr`, `Jl`, `Jr_inv`, `Jl_inv` for SE3 and
   SO3 available as named functions so that analytic Jacobians in
   `kinematics/jacobian.py` can stop approximating `Jlog ≈ I`.
4. **Backend-swappable.** Today: PyPose. Later: Warp. Zero API change at the
   call sites.

## 2. Directory layout

```
src/better_robot/
├── lie/
│   ├── __init__.py            # public lie API
│   ├── se3.py                 # SE3 group operations
│   ├── so3.py                 # SO3 group operations
│   ├── tangents.py            # Jr, Jl, hat, vee, small-angle handling
│   └── _pypose_backend.py     # private — sole pypose importer
└── spatial/
    ├── __init__.py            # Motion, Force, Inertia, Symmetric3
    ├── motion.py              # Motion (6D twist)
    ├── force.py               # Force (6D wrench)
    ├── inertia.py             # Inertia (mass, com, Symmetric3)
    ├── symmetric3.py
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

No function here knows what PyPose is; all of them forward to
`_pypose_backend`. A Warp backend will simply supply a new
`_pypose_backend`-shaped module and flip an env variable.

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

## 6. `_pypose_backend.py` — the private bridge

All `import pypose` statements in the entire codebase live in exactly one
module:

```python
# src/better_robot/lie/_pypose_backend.py
"""The ONLY pypose importer in BetterRobot.

Every function in `lie/se3.py` and `lie/so3.py` forwards here. If you need to
swap pypose for warp, replace this file and nothing else.
"""
import pypose as _pp
import torch

def _se3_compose(a, b):
    return (_pp.SE3(a) @ _pp.SE3(b)).tensor()

def _se3_inverse(t):
    return _pp.SE3(t).Inv().tensor()

def _se3_log(t):
    return _pp.SE3(t).Log().tensor()

def _se3_exp(v):
    return _pp.se3(v).Exp().tensor()

# ... and so on
```

`lie/se3.py` imports from this module only. A lint rule in
`tests/test_layer_dependencies.py` will fail the suite if any other file
imports `pypose`.

## 7. `spatial/` — value types

Where `lie/` is functional, `spatial/` exposes named value types mirroring
Pinocchio's `Motion`/`Force`/`Inertia`. These are *thin* dataclasses around
tensors, with pure-function method bodies. No operator overloading on generic
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

Dual structure. Same shape. `cross_motion` etc.

### `spatial/inertia.py`

```python
@dataclass(frozen=True)
class Inertia:
    """Spatial inertia of a rigid body in a single (..., 10) packed tensor.

    Packing: [mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    """
    data: torch.Tensor   # (..., 10)

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

    def se3_action(self, T: torch.Tensor) -> "Inertia":
        """Transform the inertia by an SE3."""

    def apply(self, v: "Motion") -> "Force":
        """I * v — inertia times twist = spatial momentum."""

    def add(self, other: "Inertia") -> "Inertia":
        """Composite-rigid-body inertia addition."""
```

Factories (`from_sphere`, `from_box`, …) steal directly from Pinocchio's
`InertiaTpl` named constructors.

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

## 10. Warp migration plan

When the Warp backend lands:

1. Add `lie/_warp_backend.py` with the same private function names as
   `_pypose_backend.py` but Warp kernels under the hood.
2. Pick the backend via an environment variable
   (`BETTER_ROBOT_LIE_BACKEND=pypose|warp`) resolved at import time.
3. `lie/se3.py` re-exports from whichever backend is active.
4. The rest of the library does not change.

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
  routed through `lie/` and `spatial/` instead.
- `math/transforms.py` (viser quaternion converters) → moves to
  `viewer/helpers.py` where it belongs.
- The `solvers/levenberg_marquardt_pypose.py` benchmark solver → deleted
  outright. It exists only to benchmark pypose, which is not a use case we
  need to keep alive.
