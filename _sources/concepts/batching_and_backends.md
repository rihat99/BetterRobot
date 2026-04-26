# Batching and Backends

The whole library has one tensor convention: `(B, [T,] ..., feature)`,
where `B` is one or more leading batch axes, `T` is an optional time
axis (trajectories only), and `feature` is the semantic last axis.
A "single pose" is `(1, 7)`, not `(7,)`. There is no scalar fast
path that diverges from the batched one; there is no
`if x.dim() == 1:` branch in the hot path.

The reason is throughput. PyRoki, brax, mjlab, mjwarp, and IsaacLab
all converged on leading-batch tensors for the same reason: when the
loop is over `model.topo_order` (a fixed Python tuple of size
`njoints`) instead of over the batch axis, there is no Python work
to amortise across batch entries. `torch.compile` unrolls the
topology cleanly per `(shape, dtype, device)` triple. Broadcasting in
PyTorch Just Works as long as the convention is consistent. Writing
a function that accepts `(B, T, ..., D)` is the same cost as writing
one for `(..., D)` — *as long as every transform along the way
respects the convention*. The "as long as" is the discipline; the
contract test `tests/contract/test_hot_path_lint.py` enforces it.

The backend layer is the second half of the same story. Below the
math layer (`lie/`, `kinematics/`, `dynamics/`) is a `Backend`
Protocol whose default is `torch_native` — pure PyTorch, no extra
dependencies. A future Warp backend (in progress) will plug in
through the same Protocol, swap the FK / Jacobian / RNEA kernels
under the hood, and leave every call site identical because users
see `torch.Tensor` in and `torch.Tensor` out at every public
surface. The architectural core is **explicit `Backend` objects**
passed via `backend=` kwargs; `default_backend()` is convenience
sugar for the common case where you do not want to thread the
backend through every call.

## Shape convention

For every tensor field on `Model` or `Data` and every argument or
return of a public function:

```
Last dim:        the manifold / feature dim
                     3 for position
                     4 for quaternion
                     6 for twist / tangent / spatial wrench
                     7 for SE(3) pose
                     nq / nv for configurations and velocities
Second-to-last:  (optional) per-joint or per-link / per-frame dim
                     njoints / nframes / nbodies / n_caps / n_pairs
Second dim:      (optional) time axis T (trajectories only)
First dim:       batch axis B
```

A function that handles a single pose still accepts a leading batch
of 1 — never a dropped batch axis. There is no "unbatched mode," no
`if x.dim() == 1:` branches in the hot path.

### Examples

| Object | Shape |
|--------|-------|
| Single SE3 pose | `(1, 7)` |
| Joint config, fixed-base Panda | `(B, 7)` or `(B, T, 7)` |
| Joint config, G1 free-flyer | `(B, 7 + n_act)` |
| `forward_kinematics` output | `(B, njoints, 7)` |
| Spatial Jacobian | `(B, 6, nv)` |
| Self-collision residual | `(B, n_candidate_pairs)` |
| Trajectory of 64 knots on 128 batch | `(128, 64, nq)` |

## Batching in practice

```python
# A user calling FK on 4096 random Panda configs
q = torch.rand(4096, model.nq, device="cuda") * (model.upper_pos_limit - model.lower_pos_limit) \
                                              + model.lower_pos_limit
data  = forward_kinematics(model, q, compute_frames=True)
poses = data.frame_pose_world[..., model.frame_id("panda_hand"), :]   # (4096, 7)
```

Inside `forward_kinematics`, the loop is over `model.topo_order` (a
fixed Python tuple), not over the batch. Per-joint transform calls
operate on slices `q[..., idx_q : idx_q + nq_j]` which keep the
leading batch axis intact.

## Heterogeneous batches — `world_start`

Newton's and mjwarp's trick for batching robots with *different*
topologies: flat buffers with a `world_start[]` array that indexes
into per-world sub-ranges. Mixing a 7-DOF Panda with a 36-DOF G1 in
one batched problem looks like:

```
flat_q        (43,)       # 7 + 36
world_start   [0, 7, 43]  # Panda in [0:7], G1 in [7:43]
```

V1 does not commit to this fully — the first-round shape convention
assumes one `Model` per `Data`. The extension point is reserved:

```python
@dataclass
class WorldIndexing:
    world_start: tuple[int, ...]     # prefix sums into flat tensors
    model_ids:   tuple[int, ...]
```

A future `MultiWorldData` will carry this and let residuals / costs
address per-world slices. The names are reserved now so the refactor
is minimal when the need arrives.

## Device and dtype polymorphism

### `Model.to(device, dtype)`

```python
def to(self, device=None, dtype=None) -> Model:
    """Return a NEW Model with every tensor buffer moved to the given
    device and dtype. The topology (parents, names, indices,
    joint_models) is copied by reference — it does not live on-device.
    """
```

Why a new `Model` instead of mutation? Because `Model` is frozen,
and because sharing a `Model` across devices (CPU for diagnostics,
CUDA for solving) is a legitimate pattern.

### `Data` follows `q`

`Data` has no `.to()`. When you pass a `q` of dtype fp32 on
`cuda:0`, the returned `Data` has all fields on `cuda:0` / fp32.
Device and dtype follow the input, never a cached value.

### Mixed precision

`forward_kinematics(model, q.half())` should return an fp16 `Data`.
The core kinematics are pure tensor ops, so mixed precision works as
long as the active backend does not secretly upcast. fp16 is **not
supported** in the optim layer — analytic Jacobians are too sensitive
to mixed precision; cast up before solving.

## `torch.compile` discipline

Every hot path must be `torch.compile`-friendly:

- Loops over `model.topo_order` are static — they unroll cleanly.
- No Python branching on tensor values.
- No `.item()` calls in FK / Jacobian / RNEA / ABA / CRBA.
- Joint-type dispatch happens at compile time
  (`model.joint_models[j]` is a tuple lookup, not a tensor
  operation).

The lint rules in `tests/contract/test_hot_path_lint.py` enforce
these patterns mechanically; see {doc}`/conventions/performance` §3
for the full list.

## GPU support

Day-one GPU support is achieved by:

1. Keeping all model buffers on device (`model.joint_placements`,
   `axes`, `inertias`, `lower_pos_limit`, …).
2. Writing every algorithm in pure PyTorch ops that already run on
   CUDA.
3. Using the torch-native Lie backend, which is gradcheck-clean by
   construction and has no PyPose autograd issues to dodge.
4. Providing `tests/test_gpu.py` that constructs models on CUDA, runs
   FK / residuals / a full IK solve, and asserts output equals the
   CPU reference to `1e-5`.

## The Backend Protocol

```python
class LieOps(Protocol):
    """Bottom-layer SE3 / SO3 ops; plain torch tensors in and out."""
    def se3_compose   (self, a, b): ...
    def se3_inverse   (self, t): ...
    def se3_log       (self, t): ...
    def se3_exp       (self, v): ...
    def se3_act       (self, t, p): ...
    def se3_adjoint   (self, t): ...
    def se3_adjoint_inv(self, t): ...
    def se3_normalize (self, t): ...
    # SO3 mirror omitted

class KinematicsOps(Protocol):
    def forward_kinematics(self, model, q): ...
    def compute_joint_jacobians(self, model, joint_pose_world): ...

class DynamicsOps(Protocol):
    def rnea (self, model, data, q, v, a, fext=None): ...
    def aba  (self, model, data, q, v, tau, fext=None): ...
    def crba (self, model, data, q): ...
    def center_of_mass(self, model, data, q, v=None, a=None): ...

class Backend(Protocol):
    name: str
    lie:        LieOps
    kinematics: KinematicsOps
    dynamics:   DynamicsOps
```

Source: `src/better_robot/backends/protocol.py`.

### Capability matrix

| Backend | `lie` | `kinematics` | `dynamics` | Status |
|---------|-------|--------------|------------|--------|
| `torch_native` | full | full | RNEA / ABA / CRBA / CCRBA / centroidal live | **default** |
| `warp` | partial | partial | partial | experimental |

### Where the backend boundary lives

| Module | Calls backend? |
|--------|----------------|
| `lie/se3.py`, `lie/so3.py` | yes |
| `lie/tangents.py` | no — pure PyTorch math |
| `kinematics/forward.py` | yes (FK kernel) |
| `kinematics/jacobian.py` | yes (joint Jacobian assembly) |
| `dynamics/rnea.py` / `aba.py` / `crba.py` | yes |
| `spatial/*.py` | indirectly (via `lie/`) |
| `data_model/joint_models/*.py` | indirectly |
| Everything else | no |

`tests/contract/test_backend_boundary.py` AST-walks `src/` and
fails if any module outside `lie/`, `kinematics/`, `dynamics/`, or
`backends/<name>/` imports a backend implementation module. The
discipline is enforced mechanically.

## Per-call vs process-wide backend selection

```python
# Explicit per-call (the primary mechanism):
import better_robot as br
from better_robot.backends import get_backend

wb   = get_backend("warp")
T    = br.lie.se3.compose(a, b, backend=wb)
data = br.forward_kinematics(model, q, backend=wb)

# Process-wide default (convenience sugar):
br.backends.set_backend("warp")           # one-time configuration
```

Library-internal code never calls `set_backend` (a contract test
greps for it). User code calls it once at startup if at all.

The reasons the global is sugar, not the architectural core:

- `torch.compile` traces against captured globals; switching the
  global mid-trace silently invalidates compiled artefacts.
- Tests that compare two backends side by side need both live in
  the same process without monkey-patching.
- Nested libraries that themselves call `set_backend` would clash.

## CUDA graph capture

```python
import better_robot as br

@br.backends.graph_capture
def ik_step(x0, target):
    return br.tasks.solve_ik(model, {"panda_hand": target}, initial_q=x0).q
```

When the Warp backend is active, the decorator builds a CUDA graph;
when `torch_native` is active, it is a no-op that just calls the
function. Users get the speedup without porting their code to Warp
kernels. Today the body lives behind a placeholder that lands with
the Warp backend — the API is stable so user code does not need to
change later.

## Warp principles (the future kernel path)

When Warp ships:

- Users never import `warp`.
- Users never write a `@wp.kernel`.
- `torch.Tensor` in, `torch.Tensor` out — no `wp.array` leakage.
- `wp.Tape` is disabled; backward goes through
  `torch.autograd.Function` with a hand-written analytic adjoint.
- Every Warp kernel is wrapped in a `WarpBridge` that caches
  `wp.array` handles per `(dtype, device, shape)` — repeated calls
  with the same shape are allocation-free.
- Adaptive kernel dispatch picks the right specialisation at first
  call, keyed on `(shape params, dtype, device)`.

The Warp directory layout under `backends/warp/` mirrors the
pattern; the bridge, the autograd wrappers, and the graph-capture
context manager are the integration surface.

## Testing the shape and device contract

Every public function gets three pro-forma tests:

1. **Shape test.** Pass `(1, nq)`, `(B, nq)`, `(B, T, nq)`; check
   every returned tensor has the expected leading batch shape.
2. **Device test.** Pass `q` on CPU, then on CUDA (skipped if no
   GPU); check outputs live on the same device.
3. **Dtype test.** Pass `q` as fp32 and fp64; check outputs match
   in dtype and are within tolerance of each other.

These live in `tests/contract/` and import every public symbol from
`better_robot.__init__`.

## Sharp edges

- **Quaternions are scalar-last `[qx, qy, qz, qw]`.** Code that
  expects scalar-first is wrong everywhere it touches an SE(3).
- **Spatial Jacobians have linear rows on top of angular rows.** A
  `(6, nv)` Jacobian's first three rows are linear velocity; the
  bottom three are angular.
- **`torch.compile` recompiles per `(shape, dtype, device)`.**
  Cycling through many batch sizes triggers many compiles; the cache
  is `TORCHINDUCTOR_CACHE_DIR` (default
  `~/.cache/torch_inductor/better_robot`).
- **Mixed precision is rejected at the boundary.** The optim entry
  points raise on fp16 inputs. Cast up to fp32 before solving; cast
  back if you need to.
- **`set_backend` is one-time configuration.** Switching mid-trace
  silently invalidates compiled artefacts.

## Where to look next

- {doc}`lie_and_spatial` — the math layer that the backend
  Protocol routes.
- {doc}`/conventions/performance` — the budgets the techniques in
  this chapter defend.
- {doc}`/conventions/extension` §10 — recipe for adding a new
  backend.
