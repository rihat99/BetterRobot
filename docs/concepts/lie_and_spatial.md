# Lie Groups and Spatial Algebra

The math layer is the foundation everything else stands on. Forward
kinematics composes SE(3) elements; pose residuals subtract them on
the manifold; dynamics computes spatial accelerations and wrenches;
the trajectory parameterisations retract through SE(3) and SO(3) for
free-flyer and spherical joints. If the math layer is wrong, every
algorithm above it is wrong in correlated ways that are very hard to
debug.

We had three choices to make. The first was the storage layout: how
do you spell an SE(3) pose? The second was whether to subclass
`torch.Tensor` (PyTorch's `__torch_function__` hook lets you) or to
keep the math layer purely functional. The third was whether to
depend on an external Lie library (PyPose was the obvious candidate)
or to implement the kernels from scratch. Each of those came with a
specific cost we wanted to avoid.

The storage layout is now fixed library-wide: `(..., 7)` for SE(3)
with the order `[tx, ty, tz, qx, qy, qz, qw]` — translation first,
scalar quaternion **last**. Tangent vectors are `(..., 6)` with
`[vx, vy, vz, wx, wy, wz]` — linear block first. SO(3) quaternions
are `(..., 4)` with the same scalar-last convention. Spatial
Jacobians are `(..., 6, nv)` with linear rows on top of angular rows.
None of these are negotiable; they are baked into every public
function and into `tests/contract/test_no_legacy_strings.py`. The
order matches what you read in Pinocchio's source and most modern
robotics libraries; the scalar-last convention matches PyPose's
native layout, which made the early prototype easy and stayed because
it composes cleanly with the rest.

The second choice was about subclassing. PyTorch's
`__torch_function__` mechanism lets you define a `class SE3(Tensor)`
that reuses the tensor machinery (`+`, `*`, `@`, slicing, autograd)
and overrides specific operations. PyPose did exactly that. It works
beautifully for two months and then starts producing subtle bugs
that are very expensive to diagnose: gradient overrides that fire
in unexpected places, autograd graphs that surgery themselves when a
subclass slips through a `torch.cat`, `__torch_function__` dispatch
that interacts oddly with `torch.compile`. We do not subclass
`torch.Tensor`. The typed wrappers (`SE3`, `SO3`, `Pose`, `Motion`,
`Force`, `Inertia`) are frozen `@dataclass`es around a tensor; their
operations route through the functional `lie.se3.*` / `lie.so3.*`
API; autograd, `vmap`, and `torch.compile` see plain tensors and do
the right thing.

The third choice was about dependencies. The early prototype used
PyPose for SE(3) ops, and it was the right call at the time —
re-deriving Barfoot's SE(3) Taylor expansions from the textbook is
not a task you want during a project's first month. Once the
prototype was working, two issues came up that pushed us off the
PyPose path. First, PyPose subclasses `torch.Tensor`, which is the
exact pattern we did not want to expose at our boundaries; routing
PyPose calls through the functional facade meant we were already
re-wrapping every operation. Second, PyPose's `Log` had an autograd
issue that forced our residual Jacobian path to use central finite
differences as the default — slow, and a tax on every iteration. The
torch-native backend (`lie/_torch_native_backend.py`) replaced
PyPose; it is gradcheck-clean by construction, has no external
dependencies, and has been the default ever since.

## Storage convention

| Object | Storage | Layout |
|--------|---------|--------|
| SE(3) pose | `(..., 7)` | `[tx, ty, tz, qx, qy, qz, qw]` (scalar last) |
| se(3) tangent | `(..., 6)` | `[vx, vy, vz, wx, wy, wz]` (linear first) |
| SO(3) quaternion | `(..., 4)` | `[qx, qy, qz, qw]` (scalar last) |
| so(3) tangent | `(..., 3)` | `[wx, wy, wz]` |
| Spatial Jacobian | `(..., 6, nv)` | rows `[v_lin (3); ω (3)]` |

These are repeated in {doc}`/conventions/contracts` and
{doc}`/conventions/style`. There is no point in the codebase where a
different convention is allowed — `tests/contract/test_no_legacy_strings.py`
prevents it.

## The functional API

`lie/` is purely functional. `lie/se3.py`, `lie/so3.py`, and
`lie/tangents.py` are modules of free functions that take and return
plain `torch.Tensor`s.

```python
# src/better_robot/lie/se3.py
def identity(*, batch_shape=(), device=None, dtype=torch.float32) -> torch.Tensor: ...
def compose(a, b)        -> torch.Tensor:   # (..., 7), (..., 7) → (..., 7)
def inverse(t)           -> torch.Tensor:   # (..., 7) → (..., 7)
def log(t)               -> torch.Tensor:   # (..., 7) → (..., 6)
def exp(v)               -> torch.Tensor:   # (..., 6) → (..., 7)
def act(t, p)            -> torch.Tensor:   # (..., 7), (..., 3) → (..., 3)
def adjoint(t)           -> torch.Tensor:   # (..., 7) → (..., 6, 6)
def adjoint_inv(t)       -> torch.Tensor:   # faster than inv(adjoint(...))
def from_axis_angle(axis, angle): ...
def from_translation(disp): ...
def normalize(t)         -> torch.Tensor:   # re-project onto SE(3)

# src/better_robot/lie/so3.py
identity, compose, inverse, log, exp, act, adjoint,
from_matrix, to_matrix, from_axis_angle, normalize

# src/better_robot/lie/tangents.py — right / left Jacobians of exp
def right_jacobian_so3(omega) -> torch.Tensor:    # Jr(ω), (..., 3) → (..., 3, 3)
def right_jacobian_inv_so3(omega): ...
def left_jacobian_so3(omega): ...
def left_jacobian_inv_so3(omega): ...
def right_jacobian_se3(xi)    -> torch.Tensor:    # Jr(ξ), (..., 6) → (..., 6, 6)
def right_jacobian_inv_se3(xi): ...
def left_jacobian_se3(xi):     ...
def left_jacobian_inv_se3(xi): ...
def hat_se3(xi): ...   # (..., 6) → (..., 4, 4)
def vee_se3(X):  ...   # (..., 4, 4) → (..., 6)
def hat_so3(w):  ...   # (..., 3) → (..., 3, 3)
def vee_so3(W):  ...   # (..., 3, 3) → (..., 3)
```

Every function routes through the active `Backend` Protocol from
{doc}`batching_and_backends`. The default backend
(`torch_native`) is implemented in `lie/_torch_native_backend.py`
and uses Barfoot's closed forms. A future Warp backend will provide
the same surface with `wp.kernel`-backed implementations; call sites
do not change.

## Singularity handling

The exp / log maps on SO(3) have a removable singularity at θ = 0
(the rotation matrix is the identity, the formula divides by sin(θ),
and a Taylor expansion around 0 is needed for stable gradients). The
SE(3) left-Jacobian has a similar singularity. Both are handled with
`torch.where` against a `θ²` cutoff:

```python
# Sketch — see src/better_robot/lie/_torch_native_backend.py
b = torch.where(theta2 < EPS,
                0.5 - theta2 / 24.0,                # Taylor lead at θ → 0
                (1.0 - cos_theta) / theta2)
c = torch.where(theta2 < EPS,
                1.0/6.0 - theta2 / 120.0,
                (theta - sin_theta) / theta3)
```

The Taylor leads are checked explicitly by
`tests/lie/test_singularities.py` at `θ ∈ {0, π/2, π − 1e-6}`.
Autograd flows smoothly through the `torch.where` because both
branches return finite, differentiable values.

## Quaternion double cover

A unit quaternion `q` and `-q` represent the same rotation. The naive
implementation of `so3.log(q)` returns different tangents for the two
representations, which breaks gradient flow at the seam. The
torch-native backend folds the two halves of the cover by flipping
the sign whenever `qw < 0`:

```python
q = torch.where((q[..., 3:4] < 0), -q, q)
```

The `log ∘ exp` round-trip is the identity modulo numerical noise,
and `assert_close_manifold` (see {doc}`/conventions/testing`) compares
on the manifold rather than element-wise.

## Numerical guarantees

| Routine | Guaranteed accuracy |
|---------|---------------------|
| `se3.exp(log(T))` | `‖Δ‖ < 1e-6` (fp32), `< 1e-12` (fp64) |
| Analytic FK Jacobian vs. `jacrev` | `‖ΔJ‖_F < 1e-4` (fp32), `< 1e-10` (fp64) |
| Long-chain FK (30 joints) | 1 ulp of a well-conditioned product of SE(3)s |

The torch-native backend passes `torch.autograd.gradcheck` at fp64
with `atol=1e-8, rtol=1e-6` on randomised unit-quaternion inputs for
every public op. See `tests/lie/test_torch_backend_gradcheck.py`.

## Typed value classes — `lie/types.py`

`lie/` is functional. `lie/types.py` adds typed `SE3`, `SO3`, `Pose`
dataclasses on top — frozen `@dataclass`es around a `torch.Tensor`,
**not** subclasses of one. They expose the functional API as named
methods plus a single unambiguous operator (`@`):

```python
@dataclass(frozen=True)
class SE3:
    """Rigid transform in SE(3), stored as (..., 7) [tx, ty, tz, qx, qy, qz, qw].

    The ``tensor`` attribute is a plain torch.Tensor — autograd, vmap,
    and torch.compile see a tensor, not a custom subclass. All
    operations route through the functional lie.se3.* API.
    """
    tensor: torch.Tensor

    @classmethod
    def identity(cls, *, batch_shape=(), device=None, dtype=torch.float32) -> "SE3": ...
    @classmethod
    def from_translation(cls, t: torch.Tensor) -> "SE3": ...
    @classmethod
    def from_rotation(cls, r: "SO3") -> "SE3": ...
    @classmethod
    def from_axis_angle(cls, axis, angle) -> "SE3": ...
    @classmethod
    def from_matrix(cls, m: torch.Tensor) -> "SE3": ...
    @classmethod
    def exp(cls, xi: torch.Tensor) -> "SE3": ...

    @property
    def translation(self) -> torch.Tensor: ...     # (..., 3)
    @property
    def rotation(self) -> "SO3": ...
    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    def compose(self, other: "SE3") -> "SE3": ...
    def inverse(self) -> "SE3": ...
    def log(self) -> torch.Tensor: ...             # (..., 6)
    def adjoint(self) -> torch.Tensor: ...         # (..., 6, 6)
    def to_matrix(self) -> torch.Tensor: ...       # (..., 4, 4)

    def act_point(self, p: torch.Tensor) -> torch.Tensor: ...
    def act_motion(self, m: "Motion") -> "Motion": ...
    def act_force (self, f: "Force")  -> "Force":  ...
    def act_inertia(self, I: "Inertia") -> "Inertia": ...

    def __matmul__(self, other):
        # SE3 @ SE3   → compose
        # SE3 @ (..., 3) → act_point
        ...
```

Source: `src/better_robot/lie/types.py`. `SO3` mirrors at `(..., 4)`.
`Pose` is an alias for `SE3`.

The typed wrappers exist so that user-facing call sites read like the
math:

```python
T_world_ee = T_world_arm @ T_arm_ee     # SE3 @ SE3 → SE3
p_world    = T_world_ee @ p_local       # SE3 @ tensor[..., 3] → tensor[..., 3]
xi         = T_world_ee.log()           # SE3 → tangent
```

Hot paths still call the functional `lie.se3.*` API directly on raw
tensors — boxing every entry of a `(B..., njoints, 7)` tensor into an
`SE3` instance would defeat batching. The convention is: storage on
`Model` and `Data` is the raw tensor; user-facing accessors
(`Data.frame_pose("name")`, `IKResult.frame_pose("name")`) return the
typed wrapper; functions accept either.

The deliberately-omitted operators are worth naming:

- **No `__mul__`.** PyTorch's `*` is element-wise multiplication on
  the underlying tensor, which would yield a leafwise scale of the
  pose — mathematical garbage. Composing two SE(3)s is `T1 @ T2`, not
  `T1 * T2`.
- **No `__invert__` (`~T`).** Sigil-overloaded inverse is unreadable;
  `T.inverse()` is one extra character and obviously correct.

## Spatial value types — `spatial/`

The 6D value types that dynamics consumes:

```python
@dataclass(frozen=True)
class Motion:
    """6D twist: linear + angular velocity. Stored as (..., 6) [vx, vy, vz, wx, wy, wz]."""
    data: torch.Tensor

    @classmethod
    def zero(cls, *, batch_shape=(), device=None, dtype=torch.float32) -> "Motion": ...

    @property
    def linear(self)  -> torch.Tensor: ...     # (..., 3)
    @property
    def angular(self) -> torch.Tensor: ...     # (..., 3)

    def cross_motion(self, other: "Motion") -> "Motion":
        """Motion × Motion (spatial acceleration cross). Featherstone eq. (2.30)."""

    def cross_force(self, other: "Force") -> "Force":
        """Motion × Force (the ad* operator). Featherstone eq. (2.31)."""

    def se3_action(self, T) -> "Motion": ...
    def compose(self, other: "Motion") -> "Motion":
        """Element-wise sum (valid because Motion is a vector space)."""
```

Source: `src/better_robot/spatial/`.

Two design choices stand out.

**Operators are explicit.** `+`, `-`, and `__neg__` are safe (Motion
is a vector space) and implemented. `__mul__` is **not** implemented
because it is ambiguous between leafwise scale and cross-product —
the same footgun as on `SE3`. Users call `.cross_motion(...)`,
`.cross_force(...)`, `.se3_action(...)` by name.

**`Force.cross_motion` raises `NotImplementedError` on purpose.**
`Force × Motion` is not a standard spatial-algebra operation. The
dual that exists is `Motion.cross_force`, which is the `ad*` operator
from Featherstone eq. (2.31). Adding a `Force.cross_motion` for
"symmetry" would teach users a non-textbook algebra. The error
message names the standard dual:

```
NotImplementedError:
    Force × Motion is not a standard spatial-algebra operation.
    The dual that exists is Motion.cross_force(force) → Force,
    which is the `ad*` operator (Featherstone eq. 2.31).
```

`Inertia` is the third value type — a typed view over a packed
`(..., 10)` tensor with layout
`[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`. Factories
(`from_sphere`, `from_box`, `from_capsule`, `from_ellipsoid`,
`from_mass_com_matrix`) exist for the common cases. Methods accept
the typed wrapper or the raw tensor; methods that return an inertia
return the typed wrapper.

## Why the split — `lie/` is functional, `spatial/` adds wrappers

Pinocchio's `SE3` class with overloaded `operator*` (across `SE3`,
`Motion`, `Force`, `Inertia`) is elegant in C++ but turns into
footguns in Python:

- `Tensor.__mul__` ambiguity (the brax lesson).
- Hidden autograd graph surgery when value types wrap a `torch.Tensor`.
- `__torch_function__` subclass drift on tensor subclasses (the
  PyPose lesson).

So `lie/` stays functional over plain tensors — backend-swappable,
autograd-clean, trivially `torch.compile`-friendly. `spatial/`
provides shallow value-type wrappers with explicit named methods for
code that reads cleaner with them (notably `dynamics/`). Kinematics
works directly on tensors; dynamics uses `Motion` / `Force` / `Inertia`
for readability.

## Sharp edges

- Quaternions are scalar-last `[qx, qy, qz, qw]`. Code that expects
  scalar-first will silently produce wrong rotations; the contract
  test `test_no_legacy_strings.py` catches the literal in `src/`.
- `lie.se3.log` returns a 6-vector with **linear first**, not angular
  first. If you compute `xi = T.log()` and pull out the angular part,
  it is `xi[..., 3:6]`, not `xi[..., :3]`.
- Storage on `Model` and `Data` is raw tensors, not `SE3` instances.
  Boxing every entry of `(B..., njoints, 7)` would be unreasonable;
  the typed wrapper is for user-facing call sites only.
- `gradcheck` for the math layer runs at fp64. fp32 is the production
  default but the gradient correctness proofs use fp64 to avoid the
  numerical noise.

## Where to look next

- {doc}`kinematics` — how the FK loop calls `lie.se3.compose` for
  every joint in `topo_order`.
- {doc}`batching_and_backends` — the `Backend` Protocol that routes
  the `lie.*` calls to `torch_native` (today) or `warp` (in
  progress).
- {doc}`/conventions/contracts` §1.3 — the quaternion-norm input
  contract.
