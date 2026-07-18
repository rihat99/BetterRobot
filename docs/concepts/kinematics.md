# Kinematics

Forward kinematics is the hot path. Every IK iteration starts with
an FK call; every trajectory cost evaluates it once per knot; every
visualisation update runs it on the current configuration. If FK is
slow, everything that depends on it is slow; if FK is wrong, every
analytic Jacobian above it is correlated-wrong. So the kinematics
layer has two jobs at once — be fast and be right — and a third
job that makes both sustainable: have only one path through.

The library has exactly one FK function (`forward_kinematics`) and exactly one
Jacobian assembly function (`compute_joint_jacobians`). Optimization residuals
share a single named-context protocol; `Problem` selects analytic blocks,
`jacrev`, `jacfwd`, or an explicit finite-difference debug path. This
discipline prevents
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
```

## Forward kinematics

```python
def forward_kinematics(
    model: Model,
    q_or_data: torch.Tensor | Data,
    *,
    compute_frames: bool = False,
    check_quaternion_norm: bool = False,
    use_warp: bool = False,
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
@dataclass(frozen=True)
class FKResult:
    joint_pose_world: torch.Tensor
    joint_pose_local: torch.Tensor


def forward_kinematics_raw(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> FKResult:
    q_full = expand_configuration(structure, q)
    world = [None] * structure.njoints
    local = [None] * structure.njoints

    for j in structure.topo_order:
        qj = q_full[
            ...,
            structure.idx_qs_full[j] : structure.idx_qs_full[j] + structure.nqs_full[j],
        ]
        Tj = joint_transform(
            structure.joint_models[j],
            structure.joint_kind_codes[j],
            structure.joint_axes[j],
            structure.joint_pitches[j],
            qj,
        )
        local[j] = lie.se3.compose(values.joint_placements[..., j, :], Tj)
        parent = structure.parents[j]
        world[j] = local[j] if parent < 0 else lie.se3.compose(world[parent], local[j])

    return FKResult(
        joint_pose_world=torch.stack(world, dim=-2),
        joint_pose_local=torch.stack(local, dim=-2),
    )
```

Properties of this FK:

- **Shared joint dispatch.** The raw pass uses stable kind codes and the
  shared `joint_transform` helper; the pass itself does not contain parser
  string cases.
- **No `base_pose` argument.** A free-flyer root is just
  `joint_models[1] = JointFreeFlyer`; the first 7 entries of `q`
  become its configuration.
- **Batched from day one.** The loop is over joints (compile-time
  constant) not over batch entries.
- **Compile-friendly.** `ModelStructure.topo_order` and its static mirrors
  are Python tuples; the loop unrolls cleanly under
  `torch.compile`.
- **No default host sync.** Free-flyer quaternions are assumed normalized.
  The public wrapper can opt into a diagnostic norm check with
  `check_quaternion_norm=True`; that check synchronizes and is excluded from
`forward_kinematics_raw`.

The public wrapper takes `use_warp=True` as an explicit whole-pass opt-in.
The CUDA-validated opt-in lane first checks eligibility and falls back to the
Torch raw pass
when the optional runtime or input layout is unsupported. There is no
per-Lie-operation dispatch or process-wide selector.

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

The `reference=` keyword takes one of the literal strings `"world"`,
`"local"`, or `"local_world_aligned"`. `"local_world_aligned"` is the
default for `get_frame_jacobian`. It is
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
`tests/test_pinocchio/test_frame_jacobian_matches_pinocchio.py` pins the
convention.

## Residual Jacobian selection

Residual Jacobian selection belongs to `better_robot.optim.Problem`, not the
kinematics layer. The configured value is a plain literal: `"auto"`,
`"analytic"`, `"jacrev"`, `"jacfwd"`, or
`"finite_difference"`. `"auto"` uses a declared `jacobian_blocks(ctx)`
block when available and otherwise chooses forward or reverse AD from the
residual and tangent dimensions. Finite differences remain an explicit debug
oracle; there is no residual-state dispatcher or enum-to-string adapter.

## Pose residual analytic Jacobian

The pose residual is the canonical example of an analytic Jacobian
written against the unified API:

```python
class PoseResidual:
    name = "pose"
    reads = ("q", "data")
    dim = 6

    def __call__(self, ctx) -> torch.Tensor:
        T_frame = ctx["data"].frame_pose_world[..., self.frame_id, :]
        T_err   = lie.se3.compose(lie.se3.inverse(self.target), T_frame)
        log_err = lie.se3.log(T_err)
        return log_err * self.weight_vec        # (B..., 6)

    def jacobian_blocks(self, ctx) -> dict[str, torch.Tensor]:
        data = ctx["data"]
        T_err   = lie.se3.compose(lie.se3.inverse(self.target),
                                  data.frame_pose_world[..., self.frame_id, :])
        log_err = lie.se3.log(T_err)
        Jr_inv  = lie.tangents.right_jacobian_inv_se3(log_err)        # (B..., 6, 6)
        J_frame = get_frame_jacobian(self.model, data,
                                     self.frame_id,
                                     reference="local")                # (B..., 6, nv)
        J = self.weight_mat @ Jr_inv @ J_frame                          # (B..., 6, nv)
        return {"q": J.index_select(-1, ctx.free_indices("q"))}
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
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> FKResult:
    """Return named world/local placements without touching any Data.

    Useful for torch.func.jacrev / torch.vmap closures where you do not
    want a stateful Data mutation in the graph.
    """
```

`frame_placements_raw(structure, values, joint_pose_world)` returns a
`FramePlacementsResult.frame_pose_world` tensor. The sequencing-free
`joint_jacobians_raw(structure, q, joint_pose_world)` returns
`JointJacobiansResult.joint_jacobians`; neither primitive mutates `Data`.

`forward_kinematics(model, q, ...)` selects one complete pass, then writes
the returned tensors into `Data`. For research code that
needs `jacrev` over FK without `Data` in the graph, the raw form is
the right entry point.

## Cache invariants

Kinematics cache consumers call `data.require(level)` and raise
`StaleCacheError` if the cache is below the required level. Producer passes,
including `forward_kinematics`, recompute their outputs and advance the level.
See {doc}`model_and_data` for the invariant.

```python
data = forward_kinematics(model, q)            # _kinematics_level = PLACEMENTS
J = compute_joint_jacobians(model, data)       # OK
J = get_frame_jacobian(model, data, fid)       # OK

data.q = new_q                                 # invalidates cache
J = compute_joint_jacobians(model, data)       # raises StaleCacheError — must call FK first
```

## Device and dtype

`forward_kinematics_raw` produces output on the same device and dtype as the
validated inputs. At the public boundary, `q` must be `float32` or `float64`
and must exactly match the model values' device and dtype; mismatches raise
`DeviceMismatchError` or `DtypeMismatchError` rather than being cast
implicitly. `Data` created by the pass inherits that device and dtype. Mixed
precision is outside the supported numerical contract because analytic
Jacobians are sensitive to it.

## What gets shipped

| Routine | Status |
|---------|--------|
| `forward_kinematics` | Live |
| `update_frame_placements` | Live |
| `compute_joint_jacobians` | Live (analytic, world frame) |
| `get_joint_jacobian` | Live |
| `get_frame_jacobian` | Live (LWA, LOCAL, WORLD reference frames) |
| `residual_jacobian` | Live (ANALYTIC / FINITE_DIFF / AUTO; FD is unbatched) |

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
- {doc}`solver_stack` — how `Problem` selects analytic and AD Jacobian blocks.
- {doc}`/conventions/contracts` §3 — accuracy guarantees for the
  analytic Jacobian path.
