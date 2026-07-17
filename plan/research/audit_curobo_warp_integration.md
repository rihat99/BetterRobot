# Audit: How cuRobo Combines Warp, Hand-Written CUDA, and PyTorch

Auditor: read-only source audit, 2026-07-16, for the Warp-first plan revision.
Subject: extracted cuRobo source at `references/kin_dyn/curobo/` — identified as
**cuRoboV2, tag `v0.8.0-2-gca94158`, commit "cuRoboV2 research release"** (git describe).
This is the V2 rewrite, NOT the public v0.7.x that most external commentary describes.
Digest cross-checked: `references/design/curobo.md` (treated as unverified claims).

All file paths below are relative to
`/data3/rikhat.akizhanov/better/BetterRobot/references/kin_dyn/curobo/curobo/_src/`
unless prefixed otherwise. CUDA is broken on this box, so this is a static-reading
audit — no claim below is backed by a local run. "Verified in source" means the code
was read; performance numbers are all "claimed in digest/paper" and are flagged as such.

---

## 1. Compute inventory — who computes what

Verified in source, per major computation:

| Computation | Layer | Evidence |
|---|---|---|
| FK (poses + collision spheres + CoM), fused | **hand CUDA C++** (`.cuh`, NVRTC-JIT) | `curobolib/kernels/kinematics/kinematics_forward_kernel.cuh`; launched from `curobolib/cuda_ops/kinematics.py:156` |
| FK backward (∂L/∂q via topology cache) | **hand CUDA C++** | `curobolib/kernels/kinematics/kinematics_backward_kernel.cuh`; `curobolib/cuda_ops/kinematics.py:269` |
| Analytic Jacobian forward (in same FK kernel) | **hand CUDA C++** | template `kinematics_fused_jacobian_kernel<...>`, `curobolib/backends/cuda_core_backend/kinematics.py:95-99` |
| Jacobian backward (grad w.r.t. q of J-dependent losses) | **hand CUDA C++** | `curobolib/cuda_ops/kinematics.py:302` (`launch_kinematics_jacobian_backward`) |
| RNEA inverse dynamics fwd + VJP backward | **hand CUDA C++** | `curobolib/cuda_ops/dynamics.py:191` (fwd), `:291` (bwd); kernels in `curobolib/kernels/dynamics/` |
| Self-collision (map-reduce for big robots) | **hand CUDA C++** | `curobolib/cuda_ops/geometry.py:17` (`SelfCollisionDistance`), kernels in `curobolib/kernels/geometry/self_collision/` |
| L-BFGS two-loop step direction | **hand CUDA C++** (torch fallback) | `curobolib/cuda_ops/optimization.py:192` (`LBFGScu`); fallback when `opt_dim>=1024 or history>31`: `optim/gradient/lbfgs.py:180-187` |
| Wolfe line search (parallel step-size select) | **hand CUDA C++** (torch fallback) | `curobolib/cuda_ops/optimization.py:22` (`wolfe_line_search`); feasibility gate `optim/components/gradient_opt_core.py:126-129` |
| B-spline state transition (knots → pos/vel/acc/jerk) fwd+bwd | **hand CUDA C++** | `curobolib/cuda_ops/trajectory.py:349` (fwd), `:415` (bwd); used by `transition/fns_state_transition.py:16-20` |
| B-spline result interpolation (knots → dense waypoints) | **hand CUDA C++** | `curobolib/cuda_ops/trajectory.py:64` |
| PBA3D distance transform (ESDF propagation) | **hand CUDA C++** | `curobolib/kernels/parallel_banding/pba3d_kernel.cuh` |
| Pose-distance cost + analytic gradient | **Warp** | `cost/wp_tool_pose.py:438-695` (kernel factory), wrapped at `:698` |
| C-space bound/state/position costs | **Warp** | `cost/wp_cspace_state.py`, `cost/wp_cspace_position.py`, `cost/warp_bound_util.py` |
| Scene collision (cuboid/mesh/voxel SDF, unified) | **Warp** (type-generic via overloads) | `geom/collision/wp_collision_kernel.py:51-62` (overload registration), `:70` (kernel) |
| Swept-sphere collision + speed metric | **Warp** | `geom/collision/wp_sweep_collision_kernel.py`, `wp_speed_metric.py` |
| LM step for seed IK (JᵀJ+λI Cholesky solve, per-problem tile) | **Warp tile API** | `optim/util/levenberg_marquardt_step.py:127-198` (`wp.tile_cholesky`, `wp.launch_tiled`) |
| Pose value-type ops (transform points, multiply, inverse) | **Warp** (fwd + warp adjoint bwd) | `geom/transform.py` (8 autograd.Functions; adjoint launch at `:820-846`) |
| Linear trajectory upsampling (PRM/result paths) | **Warp** | `util/warp_interpolation.py:17` |
| Perception: TSDF integrate, decay, ESDF seed, raycast, depth filter, marching cubes, mesh SDF | **Warp** (~8.3k LOC, bulk of warp code) | `perception/mapper/kernel/wp_*.py`, `perception/mapper/esdf/kernel/` |
| MPPI / Evolution-Strategies particle update | **plain PyTorch** (+`torch.jit` helpers) | `optim/particle/mppi.py:615-651`; `optim/components/particle_opt_core.py` imports — no curobolib, no warp |
| LM accept/reject/damping/convergence logic | **plain PyTorch** (`torch.where`, branch-free) | `solver/seed_ik/seed_iteration_state_manager.py:171-215` |
| IK / TrajOpt / MPC solver loops, seeding, goal mgmt, result select | **plain PyTorch** | `solver/solver_ik.py`, `solver/solver_trajopt.py`, `optim/components/gradient_opt_core.py:275-355` |
| Cost aggregation / weighted sums / metrics | **plain PyTorch** | `rollout/cost_manager/cost_manager_robot.py` |
| PRM graph search | **CPU Python** (NetworkX) | `graph_planner/` (digest claim, import-verified) |

### Honest answer: where does the speed come from?

**The claim "cuRobo's acceleration comes from warp kernels" is wrong for the
performance core.** Verified: every unique heavy-math inner-loop primitive — fused
FK+spheres+Jacobian, FK backward, RNEA fwd/VJP, self-collision, L-BFGS step, line
search, B-spline evaluation/gradients, PBA3D — is hand-written CUDA C++ in
`curobolib/kernels/` (11,493 LOC of `.cuh/.h`), JIT-compiled by NVRTC via
`cuda-core` with **per-robot template specialization** (link count is a template
parameter: `kinematics_fused_kernel<{compute_com}, {compile_n_links}>`,
`curobolib/backends/cuda_core_backend/kinematics.py:95-99`).

What Warp provides in cuRoboV2 (all verified):
1. **User-editable cost kernels** (pose distance, c-space) — moved from V1 CUDA to
   Warp explicitly for modifiability (digest §10.3 quotes the paper; the code split
   is consistent with that rationale, the quote itself is unverifiable locally).
2. **Type-extensible scene collision** — one generic kernel + three `wp.func`
   overloads per obstacle type (`wp_collision_kernel.py:51-62`); adding an obstacle
   type touches no kernel code.
3. **The V2 seed-IK LM solver** — warp tile API for batched dense Cholesky. This is
   one genuinely performance-relevant Warp kernel (it exists because Warp's tile
   primitives make a per-problem JᵀJ solve ~70 lines; `levenberg_marquardt_step.py:146-198`).
4. **The whole perception stack** (~8.3k LOC) — TSDF/ESDF, where warp's `wp.Mesh`
   BVH and spatial types do real work.
5. **Pose value-type ops** for the user-facing `Pose` class.

And a third, easily-missed pillar: **MPPI is plain PyTorch** and the optimizers get
their throughput from **CUDA graph capture of the whole iteration loop** (§3), not
from kernels at all. The architecture is: hand-CUDA for the irreplaceable fused
primitives, Warp for extensible/leaf math, torch for orchestration, CUDA graphs to
erase the Python/launch overhead of the orchestration layer.

---

## 2. Boundary pattern — how kernels meet torch

### 2.1 Universal rule: `torch.autograd.Function` wraps every kernel

26 `torch.autograd.Function` subclasses in `_src` (grep count). No `wp.Tape`
anywhere (`grep -rn "wp.Tape"` → zero hits). Tensors in, tensors out at every
module boundary; warp arrays never escape a wrapper.

Three distinct backward strategies coexist (all verified):

**(a) Hand-written CUDA backward kernels** — kinematics, RNEA, self-collision,
B-spline. `KinematicsFusedFunction.backward` (`curobolib/cuda_ops/kinematics.py:218-342`)
launches `launch_kinematics_backward` + `launch_kinematics_jacobian_backward` and
returns a pre-allocated `grad_out` buffer. Decorated `@once_differentiable`; guards
`if ctx.needs_input_grad[0]`; even asserts 16-byte alignment of a grad buffer
(`kinematics.py:267-268`).

**(b) Gradient computed in the forward kernel, trivial torch backward** — the cost
kernels. The Warp pose-distance kernel writes BOTH cost and analytic gradient to
pre-allocated `out_position_gradient`/`out_rotation_gradient` buffers in forward
(`cost/wp_tool_pose.py:853`); `backward` is ~20 lines of torch that scales the saved
gradient by the incoming `grad_output` (`:857-891`). Same pattern for scene collision:
`geom/collision/wp_autograd.py:99-110` (`ctx.save_for_backward(buffer.gradient)`,
backward returns it, optionally scaled). Cheap because cost gradients are needed
every iteration anyway — forward and backward fuse into one launch.

**(c) Warp adjoint kernels, manually launched** — the Pose value-type ops only.
`geom/transform.py:820-846`: backward re-launches the same kernel with
`adjoint=True`, passing `wp.from_torch(x, grad=adj_x)` grad arrays and
`adj_inputs/adj_outputs`. This uses Warp's codegen'd adjoint **without** `wp.Tape`
(no global tape state, composes with torch autograd). Note the comment at
`transform.py:786-790`: warp adjoints reject 0-stride grad tensors, so callers must
materialize gradients — a real interop sharp edge they hit.

The digest's claim §12.2 "All Warp kernels … hand-written backward passes" is
**partially wrong**: pattern (c) uses warp-generated adjoints. Correct statement:
no `wp.Tape`, and no reliance on adjoints for the hot-loop cost kernels.

### 2.2 Zero-copy interop and stream discipline

- `wp.from_torch(tensor.detach().view(-1, k), dtype=wp.vec3)` everywhere — zero-copy
  aliasing, with dtype reinterpretation to warp vector types
  (`cost/wp_tool_pose.py:811-827`).
- Every warp launch passes the **torch current stream**:
  `get_warp_device_stream()` (`util/warp.py:95-128`) converts
  `torch.cuda.current_stream()` via `wp.stream_from_torch`. Docstring states why:
  "During CUDA graph capture, this returns the capture stream" — warp launches
  become graph-capturable because they're on torch's capture stream.
- The cuda.core backend does the same for raw kernels: `PyTorchStreamWrapper`
  exposing `__cuda_stream__` (`curobolib/backends/cuda_core_backend/kernel_cache.py:73-87`),
  launches take raw `tensor.data_ptr()` (`.../kinematics.py:110+`).
- Input validation forbids implicit fixups: `curobolib/cuda_ops/tensor_checks.py:1-48`
  raises on wrong device/dtype/non-contiguous, with an explicit rationale:
  "Calling `.contiguous()` as a fallback is unsafe under CUDA graphs because the
  conditional allocation may capture a no-op or a copy depending on tensor state
  at capture time."

### 2.3 Representative wrapper end-to-end (Warp pose distance)

`cost/wp_tool_pose.py` (914 LOC; note `cost/wp_torch_pose_dist.py` is a stale
near-duplicate — its `mark_non_differentiable` call references undefined names
`terminal_pose_axes_weight` at `wp_torch_pose_dist.py:847-848` and nothing imports
it; only `wp_tool_pose.ToolPoseDistance` is imported by `cost_tool_pose.py:15`).

1. **Kernel factory with compile-time constants**:
   `create_goalset_pose_distance_kernel_with_constants(num_goalset, rotation_method)`
   (`wp_tool_pose.py:438`) closes over Python ints and returns a fresh `@wp.kernel`
   — kernel specialization per configuration, mirroring the C++ template trick.
2. **Forward** (`:700-855`): `ctx.set_materialize_grads(False)`; ~40 lines of shape
   validation (`log_and_raise`); `wp.launch(kernel=warp_kernel, dim=b*h*num_links,
   inputs=[wp.from_torch(...)...], device=wp_device, stream=wp_stream,
   adjoint=False)`; kernel writes cost AND gradient into caller-provided
   pre-allocated out buffers; `ctx.mark_non_differentiable(...)` on the 10
   non-differentiable inputs; `ctx.save_for_backward(out_position_gradient,
   out_rotation_gradient)`; returns the out buffers.
3. **Backward** (`:857-891`, `@once_differentiable`): `pos_grad =
   out_position_gradient * grad_distance[:, :, 0::2].unsqueeze(-1)`; returns
   19-tuple, `None` for everything but the two pose inputs.
4. **One thread per (batch × horizon × link)**; the goalset min-reduction is a
   serial loop inside the thread (`:574-642`).

The hand-CUDA equivalent (`KinematicsFusedFunction`, §2.1a) is identical in shape:
buffers in, launch, save minimal ctx, kernel backward. The wrapper pattern is
backend-agnostic by design — `curobolib/backends/__init__.py` selects
`cuda_core` vs `pybind` behind an identical launch-function signature
(`backends/cuda_core_backend/kinematics.py:52-55`: "EXACT same signature as the
PyBind11 version").

---

## 3. CUDA graph capture

### 3.1 The mechanism: one generic `GraphExecutor`

`util/cuda_graph_util.py:13-221`, verified in full:

- Wraps any `capture_fn(*tensors) -> tensors`. Lazy: first call records, later
  calls `copy_` inputs into the captured input tensors (skipping same-`data_ptr`
  cases, `:89-97`) then `self._graph.replay()` (`:100-102`).
- Recording protocol (`:144-180`): `gc.collect()` + `torch.cuda.synchronize()`
  (flush pending graph destructions), clone inputs, **3 warmup runs on a side
  stream**, then `with torch.cuda.graph(graph, pool=mem_pool, stream=stream)`.
- `reset()` (`:198-207`) clears graph + tensors → next call re-records. That is the
  entire invalidation story: shape change ⇒ owner calls reset or makes a new
  executor.
- Global kill-switches: `runtime.cuda_graphs: bool = True`, `debug_cuda_graphs`,
  `debug_nan` (disables graphs) — `_src/runtime.py:33-73`.

### 3.2 What gets captured: whole optimizer inner loops, **including backward**

- Gradient optimizers (L-BFGS/SR1/CG): `_graphable_methods = {"_opt_iters",
  "_prepare_initial_iteration_state"}` (`optim/components/gradient_opt_core.py:46`),
  executors created in `finish_init()` after buffers are sized (`:149-161`),
  dispatched via `_dispatch` (`:247-251`). `_opt_iters` = `inner_iters` complete
  L-BFGS steps (line search → rollout → step direction) (`:334-355`).
- **The captured region contains full autograd**: the line-search cost/gradient
  evaluation does `x.detach().requires_grad_(True)` … `cost.backward(gradient=self._l_vec)`
  (`gradient_opt_core.py:463-479`) — FK forward, FK backward kernel, cost kernels
  and their backwards all replay inside one graph. This is why every backward must
  be allocation-free and fixed-shape.
- Particle optimizers: `_graphable_methods = {"_opt_iters"}`
  (`particle_opt_core.py:70,182`).
- Seed-IK LM: `_levenberg_marquardt_step_inner_iterations` captured
  (`solver/seed_ik/seed_ik_solver.py:604-626`) = `inner_iterations` LM steps
  (warp tile solve + FK + error calc + torch.where accept/reject).
- Rollout metrics paths (`rollout/rollout_robot.py:288,315`) and perception
  (ESDF integrate/compute, robot segmenter, pose refiners) each have their own
  executors.

### 3.3 Code structure that makes capture legal

- **Outer/inner split**: host-side control flow (convergence early-exit) only at
  outer-iteration boundaries, outside the graph
  (`gradient_opt_core.py:306-314`). The convergence check
  (`optim/components/best_tracker.py:109-117`) compares a GPU count against a
  Python float — an implicit host sync — but it runs once per `inner_iters`
  replays and is skipped entirely when `fixed_iters=True`.
- **Branch-free inner iteration**: accept/reject via `torch.where` on full state
  (`seed_iteration_state_manager.py:171-215`); best-tracking via masked copies;
  NaN checks only under `debug_nan` (which disables graphs).
- **Fixed launch dims**: e.g. gather-based ESDF seeding chosen over scatter
  precisely because launch dim is occupancy-independent (digest §6.5 —
  code-consistent, perf claims unverified).
- **Mini-batch padding for variable problem counts**: seed IK solves arbitrary
  batch sizes through a fixed-size mini-batch buffer, `copy_`-ing slices in and
  cloning results out, "in-place for CUDA graph stability"
  (`seed_ik_solver.py:640-695`); state buffers kept at stable addresses
  (`:598-601`).
- Streams: costs each run on their own CUDA stream with event sync at the end
  (`rollout/cost_manager/cost_manager_robot.py:61-65,285`) — legal inside capture
  because the graph records cross-stream dependencies.

---

## 4. Data layout — the "CudaRobotModel" pattern

### 4.1 Model = one dataclass of flat device tensors

`robot/types/kinematics_params.py:22-160` (`KinematicsParams`), all verified:

- `fixed_transforms (num_links,3,4)` fp32 — parent→child rest transform (3×4, not
  4×4, not quaternion).
- `link_map (num_links,)` int16 — parent index; `joint_map` int16 — joint per link;
  `joint_map_type` **int8** — `JointType` enum; `joint_offset_map` — scale/offset
  for mimic and negative-axis joints.
- CSR-packed topology caches: `link_chain_data`/`link_chain_offsets` (ancestor
  chain per link, for backward), `joint_links_data`/`joint_links_offsets` (links
  affected per joint, for Jacobian), `joint_affects_endeffector` (bool,
  `n_joints × n_tool_frames`, flattened) — the O(1) subtree-pruning mask.
- `link_level_data`/`link_level_offsets` (int16 CSR by BFS depth) +
  `max_level_width` — tree-level parallelism for RNEA (`:146-158`).
- Geometry rides along: `link_spheres (n_configs, num_spheres, 4)` (xyz,r),
  `link_sphere_idx_map`, `link_masses_com (num_links,4)`,
  `link_inertias (num_links,8)` — "padded to 8 floats for float4 alignment in CUDA
  kernels" (`:135-137`).
- Dtype discipline enforced at every launch: fp32 / int16 / int8 / bool / int32
  checked per-tensor (`curobolib/cuda_ops/kinematics.py:124-154`); device
  membership validated in `__post_init__` (`kinematics_params.py:183+`).
- Names (`joint_names`, `link_name_to_idx_map`, `mimic_joints` dict) stay on the
  Python side; nothing string-shaped reaches a kernel.

`Kinematics` (`robot/kinematics/kinematics.py:38-68`) is **not an `nn.Module`** —
a plain class holding config + a buffer dict. Model mutation is `copy_` in place:
`update_kinematics_config` requires identical tensor shapes (`kinematics.py:443-455`)
— consistent with graph stability.

### 4.2 Batching conventions

- Canonical config shape **`(batch, horizon, dof)`**; 1-D/2-D inputs are unsqueezed
  at the single public entry point (`kinematics.py:193-198`), then everything below
  is fixed-rank. Kernels receive `b_size = batch*horizon` flattened
  (`cuda_ops/kinematics.py:115`).
- Multi-env via an index tensor: `idxs_env (batch,)` int32 maps batch elements to
  sphere-configurations (`kinematics.py:97-99,133-134`); scene collision likewise
  takes `env_query_idx` per batch element with `max_n_obs` fixed slots per env
  (`wp_collision_kernel.py:79,123-126`) — padding + index, never ragged.
- Thread mappings (verified): pose cost = 1 thread per (batch·horizon·link)
  (`wp_tool_pose.py:494-505`); scene collision = 1 thread per
  (sphere × obstacle-slot) pair, sparse `wp.atomic_add` accumulation only on
  penetration (`wp_collision_kernel.py:96,112-163`); LM = 1 tile (64 threads) per
  problem (`levenberg_marquardt_step.py:127-138`); FK = single fused kernel for
  <100 spheres else 2-kernel split (`backends/cuda_core_backend/kinematics.py:88-99`,
  branch verified; the "4 threads per link" claim is digest/paper-only).

---

## 5. CPU story

**cuRobo is CUDA-only for everything that matters. Confirmed.**

- `DeviceCfg.device` defaults to `cuda:0` (`types/device_cfg.py:19`).
- The hand-CUDA layer cannot run on CPU by construction: both backends compile
  CUDA (`kernel_cache.py` NVRTC; pybind = compiled extensions) and the launch path
  calls `torch.cuda.current_stream(...)` unconditionally
  (`backends/cuda_core_backend/kinematics.py:84`). No CPU implementation of FK,
  Jacobians, RNEA, self-collision, L-BFGS, line search, or B-splines exists.
- The Warp layer is *theoretically* CPU-capable: `get_warp_device_stream` handles
  CPU tensors (returns `stream=None`, `util/warp.py:123-128`) and warp compiles
  kernels for CPU. In practice only the **Pose value-type ops** are exercised on
  CPU: exactly one test module parametrizes over `["cpu", "cuda:0"]`
  (`tests/conftest.py:82-95` fixture; sole consumer `tests/_src/geom/test_pose.py`).
  Other "cpu" test references are config-plumbing checks, not compute.
- `helper_cpu_math.h` exists in `curobolib/kernels/common/` but is support code for
  host-side math in bindings, not a CPU execution path.
- The only CPU compute paths: `ScipyOpt` (delegates to scipy, `optim/external/scipy_opt.py`)
  and the NetworkX PRM search.

**Implication for BetterRobot** (which needs CPU): cuRobo's warp layer shows the
shape of a CPU story — warp kernels launched on the torch device with `stream=None`
on CPU — but cuRobo never validated it beyond Pose ops, and its hand-CUDA core has
no fallback at all. A library that needs real CPU support must either (a) keep a
torch reference implementation per pass (cuRobo's implicit approach: it simply
doesn't support CPU), or (b) restrict itself to warp kernels (no raw CUDA, no
`torch.cuda.*` in launch paths) and CI-test the warp-CPU device. cuRobo's own
tensor checks would mostly pass on CPU (they check "same device", not "is cuda" —
`tensor_checks.py:36-38`), so the blocker is genuinely just the kernel layer.

---

## 6. Memory / warm-start patterns

All verified:

- **Caller-owned output AND gradient buffers**: `KinematicsFusedFunction.create_buffers`
  (`cuda_ops/kinematics.py:26-90`) allocates 6 forward outputs + 6 backward
  gradient buffers once; `Kinematics.update_batch_size` reallocates only when
  (batch, horizon) changes (`kinematics.py:88-99`). The autograd.Function takes
  the buffers as *arguments* — allocation policy lives with the owner, not the op.
- Collision uses a persistent `CollisionBuffer` zeroed per call
  (`geom/collision/wp_autograd.py:76`) with atomic accumulation across obstacle
  types into the same buffer (single kernel launch per obstacle *type*, not per
  obstacle).
- Optimizer state is a flat dataclass of pre-sized tensors
  (`OptimizationIterationState`), updated via `torch.where`/`copy_`; quasi-Newton
  history in `QuasiNewtonBuffers` with **buffer shifting for MPC warm start**
  (`optim/components/quasi_newton_buffers.py`, digest-claimed, import-verified).
- Batch-size changes go through explicit `update_num_problems` →
  `finish_init` re-hooks graph executors (`gradient_opt_core.py:143-161,569-599`)
  — resizing is a rare, explicit, non-hot-path event.
- Variable workloads are padded into fixed buffers rather than resized
  (seed-IK mini-batching, §3.3).
- No `.contiguous()` / no implicit casts in kernel paths — hard errors instead
  (`tensor_checks.py` module docstring) — so the allocator is silent during
  steady-state and graphs can't capture allocation branches.

---

## 7. What stays in PyTorch, and why

Verified split:

1. **All orchestration**: solver loops, multi-stage chaining, seeding, goal
   management, result ranking (`solver/`, `motion/`, `optim/multi_stage_optimizer.py`).
   Torch here is a *scheduling language*; CUDA graphs make its overhead vanish.
2. **Whole optimizers when they're just tensor algebra**: MPPI and ES are softmax /
   z-score / EMA updates — bandwidth-trivial, shape-static, fully expressible in
   torch (+`torch.jit` for fusion, `mppi.py:615+`). Nobody wrote an MPPI kernel
   because there's nothing to fuse beyond what capture already gives.
3. **Branch-free decision logic**: LM trust-region accept/reject, damping updates,
   best-tracking — `torch.where` chains (`seed_iteration_state_manager.py:171-215`).
4. **Autograd as the composition layer**: chain rule *between* kernels is torch's
   job; kernels supply analytic VJPs. `runtime.torch_compile` exists but defaults
   **False** (`runtime.py:21`) — torch.compile is peripheral, not load-bearing.
5. **Anything touching users' data structures**: JointState ops, Pose factory
   methods, config dataclasses, parsers.

Where they drew the line: **a computation gets a kernel when it is (a) a fused
multi-stage tree traversal (FK, RNEA, B-spline), (b) a reduction with shared-memory
structure (self-collision, line search, L-BFGS dot-product chains), or (c) a
pointwise cost whose gradient can be co-computed in forward.** Everything else —
including entire optimization algorithms — remains torch, wrapped in a CUDA graph.
The lesson: kernel boundaries follow *data-reuse structure*, not module boundaries.

---

## 8. LOC / maintenance cost

Measured with `wc -l` on this checkout (excludes `.git`, docs, examples):

| Layer | LOC |
|---|---|
| Hand CUDA C++ kernels (`curobolib/kernels/`, `.cuh/.h`) | 11,493 |
| PyBind bindings + launch `.cu/.cpp` (legacy backend) | 2,557 |
| cuda.core backend (Python launch/compile layer) | 3,654 |
| `cuda_ops` autograd.Function wrappers (Python) | 1,601 |
| Warp-kernel-bearing Python files, total | 14,026 |
| — of which perception mapper/estimation | 8,279 |
| — of which core robotics (costs, collision, transform, LM, interp) | ~5,700 |
| All `_src` Python (includes the warp files above) | 84,682 |
| Public API re-export layer | 977 |
| Tests | 70,229 |

Reading of these numbers:

- The **kernel + boundary layer is ~33k LOC ≈ 28%** of non-test code
  (14k CUDA C++ + 3.7k+1.6k launch/wrapper + 14k warp files). The digest's
  "59k Python, 12k Warp, 9k CUDA" (§intro) doesn't match this checkout's counts;
  treat digest LOC as stale.
- **Dual backends are a real tax**: every hand-CUDA kernel family has a pybind
  launcher (2.6k LOC) *and* a cuda.core config/launch module (3.7k LOC) with
  intentionally identical signatures. That's ~6.3k LOC that exists only to launch
  11.5k LOC of kernels two ways.
- The hand-CUDA core is small in file count (~40 `.cuh/.h` files) but dense:
  the FK backward + Jacobian backward alone justify the "analytic backward is the
  expensive part" warning — roughly half the kinematics kernel code is backward
  paths (`kinematics_backward_helper.cuh`, `kinematics_jacobian_backward_*`).
- 26 autograd.Function wrappers is the entire torch boundary — small and uniform.
  The pattern cost is per-kernel-family, not per-op (contrast PyposeWarp's 24
  per-op wrappers at 13.3k LOC for Lie ops alone; cuRobo's `geom/transform.py`
  covers its Pose ops in 1,723 LOC using warp adjoints instead of hand-written
  backwards).
- Test:code ratio near 1:1 (70k:85k) — the kernel layer is guarded by parity
  tests, which is the hidden maintenance cost of hand kernels.

---

## 9. Digest cross-check (claims vs source)

| Digest claim | Status |
|---|---|
| Three-layer Optimizer/Rollout Protocol composition | Verified (`optim/optimizer_protocol.py`, `rollout/rollout_protocol.py`) |
| `KinematicsFusedFunction` buffer pattern (§10.4) | Verified, incl. backward-buffer threading not mentioned in digest |
| "Warp for modifiable costs, CUDA for mature algorithms" split (§10.3) | Code-consistent; the paper quote and 34%→57% share numbers unverifiable locally |
| "No wp.Tape reliance; all Warp backward hand-written" (§12.2) | **Partially wrong**: `geom/transform.py` uses warp adjoints via `wp.launch(adjoint=True)`; correct that `wp.Tape` is never used |
| CUDA-only, Pose has CPU path (§13.2) | Verified precisely (one CPU+CUDA test module, Pose ops only) |
| Single/dual FK kernel dispatch at 100 spheres | Verified branch (`config_separate is None` path); perf rationale unverified |
| Per-cost CUDA streams (§8.3) | Verified (`cost_manager_robot.py:61-65,285`) |
| Graph capture of `_opt_iters` with `_graphable_methods` (§7.2) | Verified |
| LM seeder "adapted from Newton's tile-based Warp LM" | Tile-based warp LM verified; provenance unverifiable |
| LOC totals (§intro, §15) | Do not match this checkout; use §8 numbers above |
| All performance numbers (14× vs Newton, 61× self-collision, 99.6% IK…) | Paper claims only — nothing local to verify against |

Also found (not in digest): `cost/wp_torch_pose_dist.py` is dead code with a latent
`NameError` (`:847-848`); the live module is `cost/wp_tool_pose.py`.

---

## 10. Lessons for a Warp-first BetterRobot

1. **"Warp-first" ≠ "warp everywhere".** cuRobo's fast core is fused hand-CUDA;
   its warp layer is for extensible leaf math, costs, and perception. For
   BetterRobot, the analogous conclusion: the FK/RNEA whole-pass kernels are where
   fusion pays; write them as few, large, tree-structured kernels (warp can express
   them — cuRobo's choice of CUDA there is partly historical, V1 predates warp
   maturity, and partly template-specialization) — but budget for the possibility
   that a warp version leaves performance on the table vs a templated CUDA kernel.
2. **The wrapper contract matters more than the kernel language.** One uniform
   pattern: `autograd.Function`, caller-owned pre-allocated out+grad buffers,
   `set_materialize_grads(False)`, `once_differentiable`, launch on
   `torch.cuda.current_stream` (via `wp.stream_from_torch`), hard-fail tensor
   checks (never `.contiguous()` fallback). This is what makes kernels
   graph-capturable and autograd-composable, independent of warp vs CUDA.
3. **CUDA graph capture of whole optimizer inner loops — including
   `cost.backward()` — is the second performance pillar** and it constrains
   everything: fixed shapes, no allocation in steady state, branch-free inner
   iterations (`torch.where`), host syncs only at outer boundaries, padding+index
   instead of ragged/dynamic shapes, explicit resize events that re-record.
   Design the seams for this from day one; it is cheap if planned, brutal to
   retrofit.
4. **Model data layout**: one frozen dataclass of flat device tensors (int8/int16
   indices, CSR-packed topology caches, precomputed ancestor/affects masks,
   alignment-padded per-link payloads), names/dicts on the Python side, in-place
   `copy_` for value updates with shape immutability. This is exactly the
   ModelStructure/ModelValues split already planned — validated by cuRobo.
5. **Compute analytic gradients in the forward kernel for residual/cost terms**
   (pattern b) — it makes backward nearly free and keeps the backward graph
   trivially capturable. Reserve warp adjoints (pattern c) for low-traffic value
   types; reserve hand-written backward kernels (pattern a) for the whole-pass
   tree traversals where the adjoint has different parallel structure than the
   forward.
6. **CPU support cannot be inherited from this architecture** — it must be a
   first-class requirement: warp-CPU-tested kernels and/or torch reference passes,
   with no `torch.cuda.*` in launch paths. cuRobo demonstrates the mechanism
   (`stream=None` on CPU) but not the practice.
7. **Skip the dual-backend tax.** cuRobo pays ~6.3k LOC to launch its kernels two
   ways (pybind + cuda.core). A warp-first library gets runtime compilation and
   caching from warp itself; do not add a second kernel toolchain.
