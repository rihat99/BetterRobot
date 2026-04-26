# Kinematics

Forward kinematics is the hot path. Every IK iteration starts with
an FK call; every trajectory cost evaluates it once per knot; every
visualisation update runs it on the current configuration. If FK is
slow, everything that depends on it is slow; if FK is wrong, every
analytic Jacobian above it is correlated-wrong. So the kinematics
layer has two jobs at once — be fast and be right — and a third
job that makes both sustainable: have only one path through.

The library has exactly one FK function (`forward_kinematics`),
exactly one Jacobian assembly function
(`compute_joint_jacobians`), and exactly one residual-Jacobian
dispatcher (`residual_jacobian`). The dispatcher picks analytic,
autodiff, functional, or finite-diff at call time via a
`JacobianStrategy` flag — but the call site, the residual, and the
solver never see the difference. This is the discipline that prevents
"fixed base" and "floating base" Jacobian variants from accreting
back into the codebase.

## Entry points

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

## Forward kinematics

```python
def forward_kinematics(
    model: Model,
    q_or_data: torch.Tensor | Data,
    *,
    compute_frames: bool = False,
) -> Data:
    """Compute the placements of every joint, batched.

    Shapes
    ------
    q                              (B..., nq)
    data.joint_pose_world          (B..., njoints, 7)    filled
    data.joint_pose_local          (B..., njoints, 7)    filled
    data.frame_pose_world          (B..., nframes, 7)    filled iff compute_frames=True
    """
```

Source: `src/better_robot/kinematics/forward.py`.

The algorithm is one topological pass:

```python
def forward_kinematics_raw(model: Model, q: torch.Tensor) -> tuple[Tensor, Tensor]:
    B = q.shape[:-1]
    joint_pose_world = q.new_empty((*B, model.njoints, 7))
    joint_pose_local = q.new_empty((*B, model.njoints, 7))

    # joint 0 is the universe; identity.
    joint_pose_world[..., 0, :] = lie.se3.identity(...)
    joint_pose_local[..., 0, :] = lie.se3.identity(...)

    for j in model.topo_order[1:]:
        jm     = model.joint_models[j]
        iq     = model.idx_qs[j]
        parent = model.parents[j]

        qj     = q[..., iq : iq + jm.nq]
        T_j    = jm.joint_transform(qj)
        Tfixed = model.joint_placements[j]
        joint_pose_local[..., j, :] = lie.se3.compose(Tfixed, T_j)
        joint_pose_world[..., j, :] = lie.se3.compose(
            joint_pose_world[..., parent, :], joint_pose_local[..., j, :]
        )

    return joint_pose_world, joint_pose_local
```

Properties of this FK:

- **No Python branching on joint kind.** All per-kind logic is
  encapsulated in `JointModel.joint_transform`. The FK loop does not
  contain `if jtype in ('revolute', 'continuous')`.
- **No `base_pose` argument.** A free-flyer root is just
  `joint_models[1] = JointFreeFlyer`; the first 7 entries of `q`
  become its configuration.
- **Batched from day one.** The loop is over joints (compile-time
  constant) not over batch entries.
- **Compile-friendly.** `model.topo_order` is a Python tuple;
  `joint_models` is a tuple; the loop unrolls cleanly under
  `torch.compile`.

`update_frame_placements(model, data)` is the second-step companion:

```python
def update_frame_placements(model: Model, data: Data) -> Data:
    """Populate data.frame_pose_world from data.joint_pose_world and the
    model's frame metadata.

    Each frame has parent_joint and a fixed joint_placement (SE3 in the
    parent joint's frame):

        frame_pose_world[..., f, :] = joint_pose_world[..., parent, :] ⊙ joint_placement
    """
```

It is a vectorised gather + compose. `forward_kinematics(...,
compute_frames=True)` is `forward_kinematics` followed by
`update_frame_placements`.

## The unified Jacobian dispatch

The library exposes Jacobians at three levels of granularity, all
sharing the same underlying assembly:

- **`compute_joint_jacobians(model, data)`** — fills
  `data.joint_jacobians` with the spatial Jacobian of every joint in
  one pass. Shape: `(B..., njoints, 6, nv)`. Frame: world.
- **`get_joint_jacobian(model, data, joint_id, *, reference=...)`**
  — extracts a single joint's Jacobian, transformed to the requested
  reference frame. Shape: `(B..., 6, nv)`.
- **`get_frame_jacobian(model, data, frame_id, *, reference=...)`**
  — same idea but for an arbitrary named frame.

```python
class ReferenceFrame(str, Enum):
    WORLD               = "world"
    LOCAL               = "local"
    LOCAL_WORLD_ALIGNED = "local_world_aligned"
```

`LOCAL_WORLD_ALIGNED` is the default for `get_frame_jacobian`. It is
the Jacobian where:

- linear rows = velocity of the frame origin expressed in the **world**
  frame,
- angular rows = angular velocity expressed in the **world** frame.

It is what most users want when they say "the Jacobian at the end
effector": you can multiply it by `dq` and read off how the tool
moves in world coordinates. `WORLD` returns the Jacobian at the
world origin (twist of the rigid body relative to the world);
`LOCAL` returns the Jacobian in the body-fixed frame.

### The body-frame Jacobian gotcha

To convert from `LOCAL_WORLD_ALIGNED` to a body-fixed Jacobian, only
**rotate** — do not apply the full SE(3) adjoint:

```python
# Correct (only rotate):
R_ee     = so3.to_matrix(T_ee[..., 3:])             # (B..., 3, 3)
J_local  = cat([R_ee.mT @ J_world[:3, :],
                R_ee.mT @ J_world[3:, :]])

# Wrong (adds spurious cross-term — adjoint_inv assumes WORLD, not LWA):
J_local  = se3.adjoint_inv(T_ee) @ J_world
```

The full adjoint is right when converting a `WORLD` Jacobian (twist
referenced at the world origin) to a body Jacobian; it is wrong for
LWA, where the angular and linear parts are already decoupled at the
frame origin. Getting this wrong produces correlation between linear
and angular errors that no IK solver will ever resolve. The unit test
`tests/kinematics/test_jacobian_reference_frames.py` pins the
convention.

## `JacobianStrategy` — one entry point, four strategies

```python
class JacobianStrategy(str, Enum):
    ANALYTIC    = "analytic"     # call residual.jacobian(state); error if None
    AUTODIFF    = "autodiff"     # torch.func.jacrev(residual)(state)
    FUNCTIONAL  = "functional"   # torch.func.jacfwd (useful when outputs << inputs)
    FINITE_DIFF = "finite_diff"  # opt-in central FD; useful for hand-validating analytic
    AUTO        = "auto"         # prefer ANALYTIC, fall back to AUTODIFF
```

Source: `src/better_robot/kinematics/jacobian_strategy.py`.

`AUTO` is the default everywhere. The library's analytic-vs-autodiff
discipline reads:

- Residuals that have a hand-written `.jacobian()` use it
  (`AUTO → ANALYTIC`).
- Residuals that return `None` from `.jacobian()` fall through to
  autodiff (`AUTO → AUTODIFF`). Their slot in the cost-stack
  Jacobian is filled by `torch.func.jacrev` over that one residual,
  not over the whole stack — so a single residual without analytic
  support does not force the whole problem onto autodiff.
- `FINITE_DIFF` is opt-in for hand-validating analytic Jacobians
  during development. eps: `1e-3` (fp32), `1e-7` (fp64). It is not
  used as a fallback in production code; the torch-native Lie backend
  has clean autograd, so `AUTO → AUTODIFF` is the production path.

```python
def residual_jacobian(
    residual: Residual,
    state: ResidualState,
    *,
    strategy: JacobianStrategy = JacobianStrategy.AUTO,
) -> torch.Tensor:
    """Unified residual Jacobian. Shape: (B..., dim, state_dim).

    AUTO        — call residual.jacobian(state); fall back to AUTODIFF if None.
    ANALYTIC    — require residual.jacobian(state) to return a tensor.
    AUTODIFF    — torch.func.jacrev over residual.__call__.
    FUNCTIONAL  — torch.func.jacfwd over residual.__call__.
    FINITE_DIFF — central FD, opt-in.
    """
```

The solver in `optim/` never writes Jacobian code — it asks this
function. That is how the four-way Jacobian path (fixed analytic,
fixed autodiff, floating analytic, floating autodiff) becomes one.

## Pose residual analytic Jacobian — the elegant version

The pose residual is the canonical example of an analytic Jacobian
written against the unified API:

```python
class PoseResidual(Residual):
    def __init__(self, *, frame_id: int, target: torch.Tensor,
                 pos_weight: float = 1.0, ori_weight: float = 1.0) -> None: ...

    def __call__(self, state: ResidualState) -> torch.Tensor:
        T_frame = state.data.frame_pose_world[..., self.frame_id, :]
        T_err   = lie.se3.compose(lie.se3.inverse(self.target), T_frame)
        log_err = lie.se3.log(T_err)
        return log_err * self.weight_vec        # (B..., 6)

    def jacobian(self, state: ResidualState) -> torch.Tensor:
        T_err   = lie.se3.compose(lie.se3.inverse(self.target),
                                  state.data.frame_pose_world[..., self.frame_id, :])
        log_err = lie.se3.log(T_err)
        Jr_inv  = lie.tangents.right_jacobian_inv_se3(log_err)        # (B..., 6, 6)
        J_frame = get_frame_jacobian(state.model, state.data,
                                     self.frame_id,
                                     reference=ReferenceFrame.LOCAL)   # (B..., 6, nv)
        return self.weight_mat @ Jr_inv @ J_frame                       # (B..., 6, nv)
```

A single formula. No fixed-vs-floating distinction — floating base is
handled because `get_frame_jacobian` walks the topology and joint 1
is just another joint in the chain. The `Jr_inv(log_err)` term
correctly accounts for the SE(3) manifold; an earlier draft used
`Jlog ≈ I` and produced enough correlation between rotation and
translation errors that pose IK had to weight orientation at 0.1 to
prevent oscillation. The right-Jacobian fix removed that workaround.

## Functional access (autograd-friendly)

```python
def forward_kinematics_raw(
    model: Model,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (joint_pose_world, joint_pose_local) without touching any Data.

    Useful for torch.func.jacrev / torch.vmap closures where you do not
    want a stateful Data mutation in the graph.
    """
```

`forward_kinematics(model, q, ...)` is a thin wrapper that calls the
raw implementation and writes into `Data`. For research code that
needs `jacrev` over FK without `Data` in the graph, the raw form is
the right entry point.

## Cache invariants

Every kinematics entry point calls `data.require(level)` and raises
`StaleCacheError` if the cache is below the required level. See
{doc}`model_and_data` for the invariant.

```python
data = forward_kinematics(model, q)            # _kinematics_level = PLACEMENTS
J = compute_joint_jacobians(model, data)       # OK
J = get_frame_jacobian(model, data, fid)       # OK

data.q = new_q                                 # invalidates cache
J = compute_joint_jacobians(model, data)       # raises StaleCacheError — must call FK first
```

## Device and dtype

`forward_kinematics_raw` produces output on the same device and dtype
as `q`. `Data` inherits both. `model.joint_placements`, `axes`, etc.
are coerced to `q`'s device and dtype inside the hot path via
`.to(dtype=q.dtype, device=q.device)` — but only when it is free
(dtype / device match). Mixed precision is rejected at the boundary
because analytic Jacobians are sensitive to it.

## What gets shipped

| Routine | Status |
|---------|--------|
| `forward_kinematics` | Live |
| `update_frame_placements` | Live |
| `compute_joint_jacobians` | Live (analytic, world frame) |
| `get_joint_jacobian` | Live |
| `get_frame_jacobian` | Live (LWA, LOCAL, WORLD reference frames) |
| `residual_jacobian` | Live (ANALYTIC / AUTODIFF / FUNCTIONAL / FINITE_DIFF / AUTO) |

## Sharp edges

- **`get_frame_jacobian` defaults to `LOCAL_WORLD_ALIGNED`.** The
  body-frame conversion is rotate-only; do not use the SE(3)
  adjoint.
- **In-place tensor mutation of `data.q` does not invalidate caches.**
  Reassign (`data.q = new_q`) instead. See {doc}`model_and_data`.
- **`compute_joint_jacobians` requires
  `forward_kinematics` to have run first.** It calls
  `data.require(KinematicsLevel.PLACEMENTS)` on entry.
- **Free-flyer Jacobians have shape `(B..., 6, nv)`** where `nv =
  6 + n_actuated`. The first 6 columns are the base block.

## Where to look next

- {doc}`residuals_and_costs` — residual implementations, including
  the analytic Jacobian for pose / position / orientation.
- {doc}`solver_stack` — how the optimiser feeds `JacobianStrategy`
  through to `residual_jacobian`.
- {doc}`/conventions/contracts` §3 — accuracy guarantees for the
  analytic Jacobian path.
