# 05 · Kinematics — FK, Frame Updates, Jacobians

This is the hot path of the library. FK is batched, GPU-ready, and runs
through a single topological scan over joints using per-joint
`JointModel.joint_transform`. **Jacobians are unified under one factory**
that transparently dispatches analytic, autodiff, or functional strategies —
replacing the current copy-pasted analytic/autodiff floating-base twins.

> **Naming.** This doc uses the readable field names from
> [13_NAMING.md](../conventions/13_NAMING.md). If you are porting Pinocchio code, the old
> names (`oMi`, `oMf`, `liMi`) remain as deprecated shims — see
> [02_DATA_MODEL.md §11](02_DATA_MODEL.md).

> **Jacobian strategy & residual protocol.** This doc is the single source
> of truth for `JacobianStrategy` and the `Residual` protocol. Other docs
> refer back to these sections rather than restating them.

> **Cache invariant.** Every entry point below calls
> `data.require(KinematicsLevel.PLACEMENTS)` (or higher) on entry and
> raises `StaleCacheError` if the cache is below the required level. See
> [02_DATA_MODEL.md §3.1](02_DATA_MODEL.md). `forward_kinematics` is the
> only function that *advances* the level.

> **Reference-frame enum.** All `reference="..."` arguments are `ReferenceFrame`
> enum values (`WORLD`, `LOCAL`, `LOCAL_WORLD_ALIGNED`). The enum subclasses
> `str`, so `reference == "world"` keeps working during the migration; new
> code uses `ReferenceFrame.WORLD`. See
> [13_NAMING.md §Enums](../conventions/13_NAMING.md).

## 1. Entry points

```python
# src/better_robot/kinematics/__init__.py
from .forward   import forward_kinematics, update_frame_placements
from .jacobian  import (
    compute_joint_jacobians,     # all joints at once (world frame)
    get_joint_jacobian,          # single joint, world or local
    get_frame_jacobian,          # single frame, world or local
)
from .jacobian_strategy import JacobianStrategy
```

## 2. Forward kinematics

### `forward_kinematics(model, q_or_data, *, compute_frames=False) → Data`

```python
def forward_kinematics(
    model: Model,
    q_or_data: torch.Tensor | Data,
    *,
    compute_frames: bool = False,
) -> Data:
    """Compute the placements of every joint (and optionally every frame).

    Shapes
    ------
    q                              (B..., nq)
    data.joint_pose_world          (B..., njoints, 7)    filled
    data.joint_pose_local          (B..., njoints, 7)    filled
    data.frame_pose_world          (B..., nframes, 7)    filled iff compute_frames=True

    The function is pure: it returns a new (or updated) Data; Model is never
    touched. Batching is transparent — any leading shape `B...` is supported.
    """
```

### Algorithm (single topological pass)

```python
def forward_kinematics_raw(model: Model, q: torch.Tensor) -> tuple[Tensor, Tensor]:
    B = q.shape[:-1]
    joint_pose_world = q.new_empty((*B, model.njoints, 7))
    joint_pose_local = q.new_empty((*B, model.njoints, 7))

    # joint 0 is the universe; identity.
    joint_pose_world[..., 0, :] = lie.se3.identity(batch_shape=B, device=q.device, dtype=q.dtype)
    joint_pose_local[..., 0, :] = lie.se3.identity(batch_shape=B, device=q.device, dtype=q.dtype)

    for j in model.topo_order[1:]:
        jm     = model.joint_models[j]
        iq     = model.idx_qs[j]
        parent = model.parents[j]

        qj     = q[..., iq : iq + jm.nq]
        T_j    = jm.joint_transform(qj)               # joint's own motion
        Tfixed = model.joint_placements[j]            # parent → joint origin
        joint_pose_local[..., j, :] = lie.se3.compose(Tfixed, T_j)
        joint_pose_world[..., j, :] = lie.se3.compose(
            joint_pose_world[..., parent, :], joint_pose_local[..., j, :]
        )

    return joint_pose_world, joint_pose_local
```

### Properties of this FK (vs. the original prototype)

- No `model._fk_joint_parent_link`, no `model._fk_cfg_indices`, no
  `model._fk_joint_types` — everything is resolved via
  `JointModel.joint_transform`.
- No `if jtype in ('revolute', 'continuous')` branching in the hot path.
  All per-kind logic is encapsulated in the `JointModel`.
- No separate `base_pose` argument. A free-flyer root is just
  `joint_models[1] = JointFreeFlyer`; the first 7 entries of `q` become its
  configuration.
- Batched from day one: the loop is over joints (compile-time constant) not
  over batch entries.

### `update_frame_placements(model, data) -> Data`

```python
def update_frame_placements(model: Model, data: Data) -> Data:
    """Populate data.frame_pose_world from data.joint_pose_world and the
    model's frame metadata.

    Each frame has `parent_joint` and a fixed `joint_placement` (SE3 in the
    parent joint's frame):

        frame_pose_world[..., f, :] = joint_pose_world[..., parent, :] ⊙ joint_placement
    """
```

This is a gather + compose — vectorised over frames.

## 3. Jacobians — the unified strategy

### Why one strategy

The pre-skeleton prototype split Jacobian assembly across four
duplicated paths (`_solve_fixed`, `_solve_floating_analytic`,
`_solve_floating_autodiff`, `_analytic_collision_jacobian`). Those
twins are gone. The post-skeleton design folds Jacobians into **one
strategy** that:

- is configured on the **residual**, not on the solver;
- dispatches analytic / autodiff / functional at call time;
- works identically whether the variable is `q`, `(q, v)`, or a base-free
  flyer;
- composes cleanly with the cost stack (section 07).

### The `JacobianStrategy` enum

```python
# src/better_robot/kinematics/jacobian_strategy.py
from enum import Enum

class JacobianStrategy(str, Enum):
    ANALYTIC    = "analytic"     # call residual.jacobian(state); error if None
    AUTODIFF    = "autodiff"     # torch.func.jacrev(residual)(state)
    FUNCTIONAL  = "functional"   # torch.func.jacfwd (useful when outputs << inputs)
    FINITE_DIFF = "finite_diff"  # opt-in central FD; useful for hand-validating analytic
    AUTO        = "auto"         # prefer ANALYTIC, fall back to AUTODIFF
```

`AUTO` is the default everywhere. `FINITE_DIFF` is the opt-in path used to
hand-validate a hand-coded analytic Jacobian during development — it is
**not** the fallback once the pure-PyTorch Lie backend lands (see
[03_LIE_AND_SPATIAL.md §10](03_LIE_AND_SPATIAL.md)). Until the swap is
green, `kinematics.residual_jacobian` defaults to `FINITE_DIFF` to dodge
PyPose's autograd issue; once the swap is green it flips to `AUTODIFF`.

### The residual signature

Every residual in `residuals/` is a **callable object**, not a plain
function, exactly so it can optionally own its analytic Jacobian:

```python
# src/better_robot/residuals/base.py

class Residual(Protocol):
    name: str
    dim: int                                    # output dimension

    def __call__(self, state: ResidualState) -> torch.Tensor:
        """Return residual vector of shape (B..., dim)."""

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """Return analytic Jacobian of shape (B..., dim, state_dim), or None.

        If None, the caller falls back to autodiff on __call__.
        """
```

where `ResidualState` is a thin struct carrying `(model, data, variables)`
(see section 07).

### `compute_joint_jacobians(model, data) -> Data`

The pinocchio-named canonical function computes the **spatial Jacobian of
every joint** in one pass. Shape: `data.joint_jacobians = (B..., njoints, 6, nv)`.
The 6-column-per-joint structure is assembled by propagating the motion
subspace `S(q)` of each joint through the adjoint of the kinematic chain.

```python
def compute_joint_jacobians(model: Model, data: Data) -> Data:
    """Populate data.joint_jacobians — per-joint spatial (world) Jacobian.

    Pinocchio's `computeJointJacobians`: assembles
    data.joint_jacobians[..., i, :, :] as the 6 x nv Jacobian mapping joint
    velocities v (full) to the spatial velocity of joint i expressed in the
    world frame.

    Requires data.joint_pose_world populated (call forward_kinematics first).
    """
```

Algorithmically: for each joint `j` in topo order, take its motion subspace
`S_j(q_j)` (`jm.joint_motion_subspace(qj)`, shape `(B..., 6, nv_j)`), map it
to world frame via `Ad(joint_pose_world[..., j, :])`, and place it in the
`v_slice` columns of `joint_jacobians[..., j, :, :]`. The result is a
*sparse-by-topology* matrix filled via a pure gather.

### `get_joint_jacobian(model, data, joint_id, reference=...) -> Tensor`

```python
from better_robot.kinematics import ReferenceFrame

def get_joint_jacobian(
    model: Model,
    data: Data,
    joint_id: int,
    *,
    reference: ReferenceFrame = ReferenceFrame.LOCAL_WORLD_ALIGNED,
) -> torch.Tensor:  # (B..., 6, nv)
    """Extract the spatial Jacobian of a single joint from data.joint_jacobians,
    transformed to the requested reference frame."""
```

Three reference modes mirror Pinocchio's `ReferenceFrame` enum:
`WORLD`, `LOCAL`, `LOCAL_WORLD_ALIGNED`. Subclasses `str` so legacy
literals continue to work for one release.

### `get_frame_jacobian(model, data, frame_id, reference="world") -> Tensor`

Same semantics but for arbitrary frames; composes the joint Jacobian of the
parent joint with the frame placement.

### The residual-level Jacobian entry point

```python
# src/better_robot/kinematics/jacobian.py

def residual_jacobian(
    residual: Residual,
    state: ResidualState,
    *,
    strategy: JacobianStrategy = JacobianStrategy.AUTO,
) -> torch.Tensor:
    """Unified residual Jacobian. Shape: (B..., dim, state_dim).

    AUTO — call residual.jacobian(state); if it returns None, fall back to
    AUTODIFF. This lets individual residuals advertise analytic support
    without the solver caring.

    ANALYTIC — require residual.jacobian(state) to return a tensor.
    AUTODIFF — torch.func.jacrev over residual.__call__.
    FUNCTIONAL — torch.func.jacfwd over residual.__call__.
    """
```

The solver in `optim/` never writes jacobian code — it asks this function.
That is how the current four-way Jacobian mess becomes *one* path.

## 4. Concrete analytic Jacobians we ship

Every residual in `residuals/` gets an optional `jacobian()` override. The
first round (v1):

| Residual | Analytic support |
|----------|------------------|
| `pose_residual` (frame target) | ✔ — via `get_frame_jacobian` + right-Jacobian `Jr(log(Terr))` |
| `position_residual` | ✔ — top three rows of the frame jacobian |
| `orientation_residual` | ✔ — bottom three rows + `Jr_so3(log(Rerr))` |
| `limit_residual` | ✔ — diagonal ±1 / 0 |
| `rest_residual` | ✔ — identity |
| `smoothness_residual` (5-pt FD) | ✔ — tridiagonal over time axis |
| `manipulability_residual` | ✘ — autodiff only |
| `self_collision_residual` | partial — via sparse active-pair analytic Jacobian |

Residuals with `jacobian() = None` get autodiff transparently.

## 5. Pose residual analytic Jacobian — the elegant version

Current code uses `jlog ≈ I`. Redesign:

```python
class PoseResidual(Residual):
    def __init__(self, *, frame_id: int, target: torch.Tensor,
                 pos_weight: float = 1.0, ori_weight: float = 1.0) -> None: ...

    def __call__(self, state: ResidualState) -> torch.Tensor:
        data      = state.data
        T_frame   = data.frame_pose_world[..., self.frame_id, :]
        T_err     = lie.se3.compose(lie.se3.inverse(self.target), T_frame)
        log_err   = lie.se3.log(T_err)
        return log_err * self.weight_vec        # (B..., 6)

    def jacobian(self, state: ResidualState) -> torch.Tensor:
        data      = state.data
        T_err     = lie.se3.compose(lie.se3.inverse(self.target),
                                    data.frame_pose_world[..., self.frame_id, :])
        log_err   = lie.se3.log(T_err)
        Jr_inv    = lie.tangents.right_jacobian_inv_se3(log_err)        # (B..., 6, 6)
        J_frame   = get_frame_jacobian(state.model, data, self.frame_id,
                                       reference="local")                # (B..., 6, nv)
        return self.weight_mat @ Jr_inv @ J_frame                        # (B..., 6, nv)
```

A single formula. No fixed/floating distinction — floating base is
automatically handled because `get_frame_jacobian` walks the topology
and joint 1 is just another joint in the chain. The pre-skeleton
adjoint trick (`_solve_floating_analytic`) has no equivalent in the new
code path: it does not need one.

## 6. Functional access

`kinematics/forward.py` also exposes the tensors-only primitive:

```python
def forward_kinematics_raw(
    model: Model,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (joint_pose_world, joint_pose_local) without touching any
    `Data` object.

    Useful for `torch.func.jacrev` / `torch.vmap` closures where you do not
    want a stateful `Data` mutation in the graph.
    """
```

`forward_kinematics(model, q, ...)` is a thin wrapper that calls the raw
implementation and writes into `Data`.

## 7. Device & dtype discipline

- `forward_kinematics_raw` produces output on the same device/dtype as `q`.
- `Data` inherits `batch_shape` and device from its creating tensor.
- `model.joint_placements`, `model.axes`, etc. are coerced to the input's
  device/dtype inside the hot path via `.to(dtype=q.dtype,
  device=q.device)` — but only when it's free (dtype/device match).

## 8. Testing plan

1. **Identity check** — `forward_kinematics(model, q_neutral)` returns
   all-identity placements for joint 0 and sensible body positions.
2. **PyPose regression** — for a small Panda, compare
   `forward_kinematics(model, q)` against direct PyPose composition on a
   random batch; must match to `1e-6`.
3. **Analytic == autodiff** — for every residual with an analytic Jacobian,
   compare `strategy=ANALYTIC` and `strategy=AUTODIFF` at random `q` values;
   must match to `1e-5`.
4. **Floating-base equivalence** — loading a URDF with
   `root_joint=JointFreeFlyer()` and computing the frame Jacobian at a
   fingertip must produce a matrix of shape `(6, nv)` (not `(6, n_act)` with
   a separate 6-column base block). No `_solve_floating_*` code path exists.
5. **Batched FK** — `q` of shape `(B=128, nq)` returns `(B=128, njoints, 7)`
   with no Python loop over `B`.
6. **GPU parity** — results on CUDA match CPU to `1e-5`.

## 9. What gets deleted

- `src/better_robot/algorithms/kinematics/forward.py` — replaced by
  `kinematics/forward.py`. The topological traversal moves from using
  `_fk_joint_order` to using `model.topo_order` and `joint_models[j]`.
- `src/better_robot/algorithms/kinematics/jacobian.py` — replaced by
  `kinematics/jacobian.py`. The current `compute_jacobian`,
  `limit_jacobian`, and `rest_jacobian` become residual-level
  `.jacobian()` methods on `PoseResidual`, `LimitResidual`, and
  `RestResidual` respectively.
- `src/better_robot/algorithms/kinematics/chain.py` — merged into
  `data_model/topology.py` which holds `get_subtree`, `get_support`,
  and the like as pure tree utilities.
- The `jlog ≈ I` approximation — replaced by the proper right-Jacobian
  from `lie/tangents.py`. This alone should fix the current oscillation
  that forced `ori_weight = 0.1`.
