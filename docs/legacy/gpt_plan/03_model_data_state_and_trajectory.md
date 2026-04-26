# Model, Data, State, And Trajectory

## Current State

The `Model` and `Data` split is one of the strongest parts of the design:

- `Model` is immutable topology and device-resident constants.
- `Data` is mutable per-query workspace.
- `JointModel` protocols own per-joint behavior.
- `Model.integrate` and `Model.difference` provide a universal manifold interface.

The risk is that `Data` will become a large bag of optional fields as dynamics, collision, trajectory optimization, and human models land. This is already visible: `Data` mixes input state (`q`, `v`, `a`, `tau`), kinematic outputs, Jacobians, dynamics outputs, centroidal quantities, and cache bookkeeping.

## Recommendation

Keep `Model` and `Data`, but introduce smaller state and kernel views:

- `RobotState`: user/runtime state variables.
- `JointState`: named slices or packed q/v/a/tau for models.
- `ModelTensors`: compact tensor-only view for kernels.
- `AlgorithmData` or specialized workspaces for heavy algorithm outputs.
- `Trajectory`: first-class temporal state, not just flattened `(T * nq,)`.

## Proposed Concepts

### `RobotState`

```python
@dataclass(frozen=True)
class RobotState:
    q: torch.Tensor
    v: torch.Tensor | None = None
    a: torch.Tensor | None = None
    tau: torch.Tensor | None = None
    model_id: int = -1

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    def to_data(self, model: Model) -> Data: ...
```

Use `RobotState` in tasks and optimizers where the code needs variables, not every cached output.

### `ModelTensors`

```python
@dataclass(frozen=True)
class ModelTensors:
    parents: torch.Tensor
    idx_qs: torch.Tensor
    idx_vs: torch.Tensor
    nqs: torch.Tensor
    nvs: torch.Tensor
    joint_kind_ids: torch.Tensor
    joint_axes: torch.Tensor
    joint_placements: torch.Tensor
    body_inertias: torch.Tensor
```

`Model` remains friendly and Pythonic. `ModelTensors` is for Torch compile and Warp kernels.

### `Data`

`Data` should remain a mutable cache, but with stricter ownership:

- `Data.q` can remain for convenience.
- Cache fields are outputs of named algorithms.
- `reset()` clears derived fields only.
- `_kinematics_level` should become a small explicit enum or dirty flag set.
- Add `Data.require(model)` or validation helpers for model identity and batch/device/dtype.

## Cache Policy

Every cached field should answer:

- Which algorithm writes it?
- Which inputs is it valid for?
- What shape does it have?
- Is it in world frame, local frame, or local-world-aligned frame?
- Is it differentiable with respect to `q`, `v`, or `a`?

The current field names are good. The next improvement is cache validity.

Example:

```python
data.cache_tag = CacheTag(
    q_id=id(q),
    v_id=id(v) if v is not None else None,
    kinematics_level=KinematicsLevel.POSITION,
)
```

Avoid expensive hashing of tensor contents. Use identity/version style checks only where needed.

## Model Topology And Tensor Constants

`Model` currently stores many tuple fields and tensor fields together. That is readable. For future kernels, add internal split views:

- `ModelTopology`: names, parents, children, supports, joint models, frame metadata.
- `ModelConstants`: tensor buffers such as placements, inertias, limits.
- `ModelTensors`: compact numeric view suitable for kernels.

This can be added without breaking `Model` immediately.

## JointModel Direction

Keep the `JointModel` protocol, but add hooks before advanced joints make dynamics fragile:

- `joint_bias_acceleration(q_slice, v_slice) -> (..., 6)`.
- `integrate_jacobian(q_slice, v_slice)`.
- `difference_jacobian(q0_slice, q1_slice)`.
- Optional `kind_id` integer for kernels.

The current RNEA notes assume `c_J = 0`. That is correct for current joints, but anatomical, coupled, or custom joints will need this hook.

## State And Manifold Types

Move `StateMultibody` out of skeleton status earlier than optimal-control features. It is the natural place to standardize:

- `x = [q, v]`,
- `dx = [dq_tangent, dv]`,
- `integrate(x, dx)`,
- `diff(x0, x1)`,
- state Jacobians,
- neutral state.

Then solvers, dynamics action models, and trajectory optimization can share one state contract.

## Trajectory

`Trajectory` should be more than a container:

- Store `t`, `q`, optional `v`, `a`, `u`.
- Validate `(B, T, nq)` and `(B, T, nv)` shapes.
- Support `to(device, dtype)`.
- Support slicing by index and time.
- Support resampling for Euclidean joints first, with manifold interpolation for SO3/SE3.
- Support B-spline parameterization as a sibling type, not bolted onto `solve_trajopt`.

Proposed split:

```text
tasks/trajectory.py          # sample-based Trajectory
tasks/bspline.py             # BSplineTrajectory / control-point parameterization
tasks/trajopt.py             # solver facade
```

## Heterogeneous Batching

V1 can keep one `Model` per `Data`. Do not solve heterogeneous robot batches too early.

But prepare the naming:

- Homogeneous batch: one model, many states.
- Temporal batch: one model, `(B, T, ...)`.
- Heterogeneous world batch: future `WorldIndexing`.

Do not leak `world_start` or flat concatenated world layouts into public APIs.

## Dtype, Device, And Optional Precision

Settle the dtype contract now:

- Public algorithms support float32 and float64.
- Public algorithms reject float16 and bfloat16.
- `Model.to(device, dtype)` returns a new model and moves all tensor constants.
- `Data` follows the tensors passed to it.

Reduced precision can be a backend experiment later, not a v1 promise.

## Metadata

`Model.meta` is currently an untyped dict. Keep a dict for parser-specific payloads, but reserve well-known typed subkeys:

- `source`: parser name/path/hash.
- `assets`: mesh paths and package roots.
- `human`: OpenSim/SMPL/anatomical metadata.
- `backend`: cached kernel plans, if any.

Do not let algorithm code depend on arbitrary parser metadata.

## Tests To Add

- `ModelTensors` matches `Model` values across device/dtype moves.
- `RobotState.to_data(model)` preserves q/v/a/tau and batch shape.
- `Data.reset()` clears only derived cache fields.
- Calling FK with stale `Data.q` cannot silently reuse old cache.
- `StateMultibody.integrate` and `diff` roundtrip for revolute, spherical, and free-flyer joints.
- `Trajectory` validates shapes and model id.
- `Trajectory.resample` uses Lie interpolation for floating-base and spherical joints.
