# Lie Types And Backend Boundary

## Current State

The Lie layer is currently a functional facade:

- `lie.so3` exposes functions over tensors shaped `(..., 4)`.
- `lie.se3` exposes functions over tensors shaped `(..., 7)`.
- `lie.tangents` owns hat/vee and Jacobians.
- `_pypose_backend.py` owns all PyPose imports.

This is clean for backend replacement, but it forces every caller to remember shape, convention, group semantics, and when composition vs point action is intended. It also makes the code read like "quaternion arrays" instead of "rotations" and "poses".

## Decision

Introduce `SO3` and `SE3` as first-class value types while keeping the functional tensor kernels.

This should be a breaking design update to `docs/design/03_LIE_AND_SPATIAL.md`: the old "no SE3 class" rule should be replaced with "typed value objects for public and ergonomic code; tensor functions for kernels and hot loops."

## Non-Goals

- Do not implement `SO3` or `SE3` as `torch.Tensor` subclasses in v1.
- Do not expose PyPose `LieTensor` objects.
- Do not expose Warp arrays or Warp objects.
- Do not make group operations magical enough to hide shape/device errors.
- Do not make `*` mean many things. Ambiguous multiplication has hurt other libraries.

## Proposed Module Layout

```text
better_robot/lie/
  __init__.py
  types.py              # SO3, SE3, TangentSO3, TangentSE3 if needed
  so3.py                # functional tensor API, stable low-level layer
  se3.py                # functional tensor API, stable low-level layer
  tangents.py           # hat/vee/Jr/Jl/Jlog utilities
  torch_impl.py         # pure Torch kernels, default implementation
  backend.py            # LieBackend protocol and dispatch helpers
  pypose_oracle.py      # optional test/reference bridge, not runtime default
```

The old `_pypose_backend.py` should become an optional oracle or disappear once pure Torch parity is established.

## `SO3` API Sketch

```python
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class SO3:
    tensor: torch.Tensor  # (..., 4), [qx, qy, qz, qw]

    @classmethod
    def identity(cls, *, batch_shape=(), device=None, dtype=torch.float32) -> "SO3": ...

    @classmethod
    def exp(cls, omega: torch.Tensor) -> "SO3": ...

    @classmethod
    def from_quat(cls, quat: torch.Tensor, *, normalize: bool = False) -> "SO3": ...

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> "SO3": ...

    def log(self) -> torch.Tensor: ...
    def inverse(self) -> "SO3": ...
    def compose(self, other: "SO3") -> "SO3": ...
    def act(self, points: torch.Tensor) -> torch.Tensor: ...
    def to_matrix(self) -> torch.Tensor: ...
    def normalize(self) -> "SO3": ...
    def to(self, *args, **kwargs) -> "SO3": ...
    def detach(self) -> "SO3": ...

    def __matmul__(self, other):
        # SO3 @ SO3 -> SO3
        # SO3 @ points -> rotated points
        ...
```

## `SE3` API Sketch

```python
@dataclass(frozen=True)
class SE3:
    tensor: torch.Tensor  # (..., 7), [tx, ty, tz, qx, qy, qz, qw]

    @classmethod
    def identity(cls, *, batch_shape=(), device=None, dtype=torch.float32) -> "SE3": ...

    @classmethod
    def exp(cls, xi: torch.Tensor) -> "SE3": ...

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, *, normalize: bool = False) -> "SE3": ...

    @classmethod
    def from_translation_rotation(cls, translation: torch.Tensor, rotation: SO3) -> "SE3": ...

    @property
    def translation(self) -> torch.Tensor: ...

    @property
    def rotation(self) -> SO3: ...

    def log(self) -> torch.Tensor: ...
    def inverse(self) -> "SE3": ...
    def compose(self, other: "SE3") -> "SE3": ...
    def act(self, points: torch.Tensor) -> torch.Tensor: ...
    def adjoint(self) -> torch.Tensor: ...
    def normalize(self) -> "SE3": ...

    def __matmul__(self, other):
        # SE3 @ SE3 -> SE3
        # SE3 @ points -> transformed points
        ...
```

## Operator Policy

Recommended:

- `A @ B` means group composition when both operands are the same Lie type.
- `T @ p` means group action on points when the right operand is a tensor with trailing dimension 3.
- `R @ p` means rotate points.
- `T.inverse()` and `R.inverse()` are explicit. Avoid `~T` unless the project strongly wants it.
- `T.log()` returns the tangent tensor, not a `TangentSE3` object at first.
- `SO3.exp(w)` and `SE3.exp(xi)` are constructors.

Possible but not preferred:

- `A * B` as same-type composition only. If added, it must reject points and scalar tensors, because `*` already has too many meanings in tensor code.

Do not implement:

- Elementwise arithmetic on `SO3` and `SE3`.
- `+` and `-` on group objects. Use `compose`, `inverse`, `exp`, and `log`.

## Functional API Policy

Keep the existing functions:

- `so3.identity`, `so3.exp`, `so3.log`, `so3.compose`, `so3.inverse`, `so3.act`.
- `se3.identity`, `se3.exp`, `se3.log`, `se3.compose`, `se3.inverse`, `se3.act`.

Their contract remains plain tensor input and output. They are the kernel layer used by:

- kinematics scans,
- dynamics recursions,
- collision kernels,
- Warp wrappers,
- benchmark code,
- internal vectorized paths where object allocation would be noise.

The typed classes call these functions. High-level code can accept `SE3 | torch.Tensor` during a transition, but new public APIs should prefer `SE3` for poses and `SO3` for rotations.

## Pure Torch Lie Kernels

Implement pure Torch SO3 and SE3 kernels before making these classes the primary API.

Needed kernels:

- SO3 normalize, compose, inverse, exp, log, act, to_matrix, from_matrix, slerp.
- SE3 compose, inverse, exp, log, act, adjoint, adjoint_inv, sclerp.
- Small-angle-safe Jacobians for SO3 and SE3.
- Ambient gradients that match central finite differences for the public tensor representation.

Use PyPose and Pinocchio only as test references. BetterRobot's runtime should not depend on PyPose semantics.

## Validation And Invariants

Constructors should validate trailing dimensions:

- `SO3`: `(..., 4)`.
- `SE3`: `(..., 7)`.
- SO3 tangent: `(..., 3)`.
- SE3 tangent: `(..., 6)`, `[vx, vy, vz, wx, wy, wz]`.

Quaternion convention remains scalar-last:

- SO3: `[qx, qy, qz, qw]`.
- SE3: `[tx, ty, tz, qx, qy, qz, qw]`.

Validation should be strict at public boundaries and cheap inside kernels:

- Public constructors can check finite values and quaternion norm thresholds.
- Internal `_unsafe_from_tensor` constructors can skip checks for hot loops.
- Normalization should be explicit unless the public contract says drift within tolerance is renormalized.

## Type Alias Cleanup

`src/better_robot/_typing.py` currently aliases `SE3 = torch.Tensor` and `SO3 = torch.Tensor`. Once real classes exist:

- Rename tensor aliases to `SE3Tensor` and `SO3Tensor`.
- Export class names only from `better_robot.lie` or a dedicated `better_robot.manifolds` namespace.
- Do not add `SO3` and `SE3` to the 25 top-level `better_robot.__all__` unless the project decides the top-level API should include manifold types.

## Migration Strategy

Phase 1:

- Add `SO3` and `SE3` classes without changing existing functions.
- Add tests for class/function equivalence.
- Add docs examples with typed classes.

Phase 2:

- Move high-level residual and task constructors to accept typed targets:
  `PoseResidual(target: SE3 | torch.Tensor)`.
- Keep internal storage in tensors.

Phase 3:

- Update `Frame.joint_placement`, collision rotations, and builder inputs to accept typed values while storing packed tensors in `Model`.

Phase 4:

- Decide whether typed Lie values should appear in top-level API.

## Tests To Add

- `SO3 @ SO3` equals `so3.compose`.
- `SO3 @ points` equals `so3.act`.
- `SE3 @ SE3` equals `se3.compose`.
- `SE3 @ points` equals `se3.act`.
- `SO3.exp(w).log()` roundtrips near zero and near pi.
- `SE3.exp(xi).log()` roundtrips under Pinocchio tolerances.
- `to`, `detach`, device, dtype, and batch broadcasting preserve class type.
- Invalid trailing shapes raise `ShapeError`.
- Class methods do not import PyPose.
- Gradients of pure Torch kernels match central finite differences.
