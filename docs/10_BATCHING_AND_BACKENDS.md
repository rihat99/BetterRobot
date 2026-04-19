# 10 · Batching, Devices, and Backends

This document locks the shape/device conventions for the whole library and
lays out the future Warp backend path. The one rule: **`(B, [T,] ..., D)`
is the canonical shape and `torch.Tensor` is the canonical type** — users
never see anything else.

> Performance budgets for every hot path live in
> [14_PERFORMANCE.md](14_PERFORMANCE.md); this doc explains the
> mechanisms that meet them.

## 1. Shape convention

For every tensor field on `Model` or `Data` and every argument/return of
a public function:

```
Last dim:        the manifold / feature dim
                     3 for position
                     4 for quaternion
                     6 for twist / tangent / spatial wrench
                     7 for SE(3) pose
                     nq / nv for configurations and velocities
Second-to-last:  (optional) per-joint or per-link/per-frame dim
                     njoints / nframes / nbodies / n_caps / n_pairs
Second dim:      (optional) time axis T (trajectories only)
First dim:       batch axis B
```

A function that handles a single pose still accepts a leading batch of 1
— never a dropped batch axis. There is no "unbatched mode", no
`if x.dim() == 1:` branches in the hot path.

### Examples

| Object | Shape |
|--------|-------|
| Single SE3 pose | `(1, 7)` |
| Joint config, fixed base Panda | `(B, 7)` or `(B, T, 7)` |
| Joint config, G1 free-flyer | `(B, 7 + n_act)` |
| `forward_kinematics` output | `(B, njoints, 7)` |
| Spatial Jacobian | `(B, 6, nv)` |
| Self-collision residual | `(B, n_active_pairs)` |
| Trajectory of 64 knots on 128 batch | `(128, 64, nq)` |

### Why one convention forever

- PyRoki, brax, mjlab, mjwarp, IsaacLab all converged on leading-batch.
- Broadcasting in PyTorch Just Works when the convention is consistent.
- Writing a function that accepts `(B, T, ..., D)` is the same cost as
  writing one for `(..., D)` — as long as every transform along the way
  respects it.

## 2. Batching in practice

```python
# A user calling FK on 4096 random Panda configs
q = torch.rand(4096, model.nq, device="cuda") * (model.upper_pos_limit - model.lower_pos_limit) + model.lower_pos_limit
data = forward_kinematics(model, q, compute_frames=True)
poses = data.frame_pose_world[..., model.frame_id("panda_hand"), :]   # (4096, 7)
```

Inside `forward_kinematics`, the loop is over `model.topo_order` (a fixed
Python tuple), *not* over the batch. Per-joint transform calls operate on
slices `q[..., idx_q:idx_q+nq_j]` which keep the leading batch axis
intact.

## 3. Heterogeneous batches — `world_start`

Newton's and mjwarp's trick for batching robots with *different* topologies:
flat buffers with a `world_start[]` array that indexes into per-world
sub-ranges. Example — mixing a 7-DOF Panda with a 29-DOF G1 in a single
batched problem:

```
flat_q        (1 * 7 + 1 * 36,)    # 43 total
world_start   [0, 7, 43]           # Panda in [0:7], G1 in [7:43]
```

BetterRobot's v1 does **not** fully commit to this — the first-round shape
convention assumes one `Model` per `Data`. The extension point is:

```python
@dataclass
class WorldIndexing:
    world_start: tuple[int, ...]     # prefix sums into flat tensors
    model_ids:   tuple[int, ...]     # which Model each world belongs to
```

A future `MultiWorldData` will carry this and let residuals/costs address
per-world slices. We reserve the name now so the refactor is minimal when
we need it.

## 4. Device & dtype polymorphism

### `Model.to(device, dtype)`

```python
def to(self, device=None, dtype=None) -> Model:
    """Return a NEW Model with every tensor buffer moved to the given
    device and dtype. The topology (parents, names, indices, joint_models)
    is copied by reference — it does not live on-device.
    """
```

Why a new `Model` instead of mutation? Because `Model` is a frozen
`@dataclass`, and because sharing a `Model` across devices (CPU for
diagnostics, CUDA for solving) is a legitimate pattern.

### `Data` follows `q`

`Data` has no `.to()`. When you pass a `q` of dtype fp32 on `cuda:0`, the
returned `Data` has all fields on `cuda:0`/fp32. Device/dtype follow the
input, never a cached value.

### Mixed precision

`forward_kinematics(model, q.half())` should return an fp16 `Data`. The
core kinematics are pure tensor ops, so mixed precision works for free
as long as the pypose backend (today) and Warp backend (later) don't
secretly upcast.

## 5. `torch.compile`

Every hot path must be `torch.compile`-friendly:

- Loops over `model.topo_order` are static — unrolled cleanly.
- No Python branching on tensor values.
- No `.item()` calls in FK / Jacobian / RNEA / ABA / CRBA.
- Joint-type dispatch happens at compile time (the `joint_models[j]`
  resolution is a tuple lookup, not a tensor operation).

## 6. GPU support

Day-one GPU support is achieved by:

1. Keeping all model buffers on device (`model.joint_placements`, `.axes`,
   `.inertias`, `.lower_pos_limit`, …).
2. Writing every algorithm in pure PyTorch ops that already run on CUDA.
3. Never calling pypose functions that break the autograd graph on CUDA —
   `lie/_pypose_backend.py` is the gatekeeper for this.
4. Providing a `tests/test_gpu.py` that constructs models on CUDA, runs FK
   / residuals / a full IK solve, and asserts output equals the CPU
   reference to `1e-5`.

## 7. Warp backend roadmap

### Principles

- Users never import warp.
- Users never write a `@wp.kernel`.
- `torch.Tensor` in, `torch.Tensor` out — no `wp.array` leakage.
- `wp.Tape` is disabled (`set_module_options({"enable_backward": False})`).
- Every Warp kernel is wrapped in a `torch.autograd.Function` with an
  analytic (or RNEA-derivative) `backward`.

### The `WarpBridge` pattern (mjlab)

```python
# src/better_robot/backends/warp/bridge.py

class WarpBridge:
    """Bidirectional torch ↔ warp tensor shim.

    Stores per-tensor caches so we don't re-allocate `wp.array` handles
    across repeated calls with the same-shape inputs (cuRobo pattern).
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[torch.dtype, torch.device, tuple[int, ...]], wp.array] = {}

    def to_warp(self, t: torch.Tensor) -> wp.array:
        key = (t.dtype, t.device, tuple(t.shape))
        arr = self._cache.get(key)
        if arr is None:
            arr = wp.from_torch(t)
            self._cache[key] = arr
        else:
            arr.assign(wp.from_torch(t))         # overwrite in place
        return arr

    def from_warp(self, arr: wp.array) -> torch.Tensor:
        return wp.to_torch(arr)
```

Every Warp kernel call goes through the bridge. Because it caches `wp.array`
handles, repeated calls with the same shape are allocation-free — this is
what makes the Warp backend faster than torch_native for tight loops.

### Adaptive kernel dispatch (cuRobo)

Some Warp kernels specialise on discrete shape parameters (number of
collision spheres, trajectory horizon). The backend picks a kernel at
first call:

```python
# src/better_robot/backends/warp/kernels/collision.py

@cache_kernel
def sdf_kernel(n_spheres: int) -> wp.Kernel:
    if n_spheres <= 100:
        return _single_fused_kernel         # all pairs in one launch
    else:
        return _dual_kernel                 # per-body transform + per-pair
```

`@cache_kernel` stores the specialisation in a dict keyed on
`(shape params, dtype, device)`. Warp's native content-addressed cache
picks up the kernel object by hash and re-uses the compiled artefact
across Python processes — so cold-start after process restart is cheap
(see [14_PERFORMANCE.md §5](14_PERFORMANCE.md)).

### Directory layout

```
src/better_robot/backends/warp/
├── __init__.py            # enable_warp_backend() / disable_warp_backend()
├── bridge.py              # TorchArray / WarpBridge (mjlab pattern)
├── graph_capture.py       # @graph_capture context manager
├── kernels/
│   ├── fk.py              # @wp.kernel forward-kinematics kernel(s)
│   ├── jacobian.py
│   ├── rnea.py
│   ├── aba.py
│   └── crba.py
└── autograd.py            # torch.autograd.Function wrappers per kernel
```

### How it plugs in

1. A runtime switch:

   ```python
   import better_robot as br
   br.backends.set_backend("torch_native")  # default
   br.backends.set_backend("warp")          # Warp, when available
   ```

2. `kinematics/forward.py` is unchanged — it calls
   `backends.current().forward_kinematics(model, q)`. The torch_native
   backend implements it as described in 05; the Warp backend dispatches
   to `backends/warp/kernels/fk.py`.
3. `lie/se3.py` calls `backends.current().lie.se3.*`, so pypose → Warp is
   also a backend switch.

### CUDA graph capture — the one Warp feature we expose

Warp's `wp.capture_*` API (wrapped in mjwarp's `@event_scope`) yields enormous
speedups for repeated step calls. We expose it as a first-class PyTorch
context manager:

```python
import better_robot as br

@br.backends.graph_capture
def ik_step(x0, target):
    return br.tasks.solve_ik(model, {"panda_hand": target}, initial_q=x0).q
```

Underneath: when the Warp backend is active, the decorator builds a CUDA
graph; when torch_native is active, it is a no-op that just calls the
function. Users get the speedup without porting their code to Warp kernels.

### What we do **not** expose

| Warp feature | Exposed? | Why |
|--------------|----------|-----|
| `wp.launch` | No | behind `backends/warp/kernels/*` |
| `wp.kernel` | No | authored internally only |
| `wp.array`  | No | `torch.Tensor` only in the public API |
| `wp.Tape`   | No | `torch.autograd.Function` wrappers handle the backward |
| `wp.ScopedDevice` | No | `model.to('cuda')` |
| Content-addressed kernel cache | Yes, advanced escape hatch | for users writing their own kernels |

(See [BEST_PRACTICES.md §12.6](../reference_optim/design/BEST_PRACTICES.md#126-what-to-hide-what-to-expose).)

### Kernel hygiene

Adopt mjwarp's 5-block convention for every kernel:

```python
@wp.kernel
def fk_kernel(
    # Model:
    parents:          wp.array(dtype=int),
    joint_placements: wp.array(dtype=wp.transform),
    topo_order:       wp.array(dtype=int),
    # Data in:
    q:                wp.array(dtype=float),
    # In:
    # (none)
    # Data out:
    joint_pose_world: wp.array(dtype=wp.transform),
    # Out:
    # (none)
):
    ...
```

A custom AST lint rule (like mjwarp's) checks that every kernel follows
the convention. Kernels without it fail CI.

## 8. Testing the shape and device contract

Every function gets three pro-forma tests:

1. **Shape test** — pass `(1, nq)`, `(B, nq)`, `(B, T, nq)`; check
   every returned tensor has the expected leading batch shape.
2. **Device test** — pass `q` on CPU, then on CUDA (skipped if no GPU);
   check outputs live on the same device.
3. **Dtype test** — pass `q` as fp32 and fp64; check outputs match in
   dtype and are within tolerance of each other.

These live in `tests/test_contracts.py` and import every public symbol
from `better_robot.__init__`.
