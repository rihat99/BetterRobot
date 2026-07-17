# Audit: NVIDIA Warp as a Compute Backend + Kernel Architecture of Warp-Based Simulators

Auditor: read-only research subagent (Claude). Date: 2026-07-16.
Purpose: evidence base for the "Warp-first" revision of the BetterRobot redesign plan
(see `~/.claude/handoffs/warp_first_context.md`).

Sources (read-only, all under `references/`):

| Source | Version | Path prefix used below |
|---|---|---|
| Warp | 1.13.0.dev0 (git 53a7bf5, 2026-04-09) | `warp/` = `references/sim/warp/` |
| mujoco_warp (MJWarp) | 3.6.0 (git 36fc8be, 2026-04-07) | `mjw/` = `references/sim/mujoco_warp/mujoco_warp/` |
| Newton | 1.2.0.dev0 (git 8baee876, 2026-04-09) | `newton/` = `references/sim/newton/newton/` |
| Digests (unverified claims) | `references/design/{warp,mujoco_warp,newton,PyposeWarp}.md` | |

Method: every load-bearing claim below was checked against source or shipped docs;
digest claims are marked **VERIFIED** / **REFUTED** / **UNVERIFIED** where they matter.
**No runtime measurements were possible** — CUDA is broken on this box (driver too old
for installed torch) and warp is not installed in the repo venv. All performance
numbers quoted are the ones *shipped in Warp's own docs*, clearly labeled as such.

---

## 1. Torch interop

### 1.1 `wp.from_torch` — zero-copy aliasing, no synchronization

`warp/warp/_src/torch.py:190-312`. Mechanics:

- Wraps `t.data_ptr()` directly: `warp.array(ptr=t.data_ptr(), dtype=..., shape=..., strides=..., copy=False)` (`torch.py:298-307`). Keeps the tensor alive via `a._tensor = t` (`torch.py:310`). Strides are taken from `t.stride()` × ctype size (`torch.py:221`) — **non-contiguous tensors are supported** (strided warp arrays), except the *inner* dims when viewing as `wp.vec3`/`wp.mat33`/`wp.transform`: trailing dims must match the value type's shape and be contiguous, then get trimmed off (`torch.py:226-241`). So a `(B, 7)` float tensor aliases cleanly as a length-`B` `wp.array[wp.transform]`.
- **No stream/device synchronization of any kind** happens in `from_torch` or `to_torch` — the functions only do pointer/metadata work. Ordering correctness is entirely the caller's problem (see §9.1).
- `return_ctype=True` (`torch.py:285-295`) returns a raw `array_t(ptr, grad_ptr, ndim, shape, strides)` descriptor instead of a `wp.array` — the documented hot-path form. Warp's own interop benchmark (`docs/user_guide/interoperability.rst:532-548`): `from_torch` 5095 ms vs `return_ctype=True` 2113 ms vs passing the torch tensor straight into `wp.launch` (via `__cuda_array_interface__`, no grads) 2950 ms — i.e. **the Python-side wrapper construction is ~2.4× overhead, so cache the converted arrays or use ctype descriptors** (`interoperability.rst:471-516`).
- dtype map: fp16/32/64, ints, bool. **`torch.bfloat16` and complex are NOT supported** (`torch.py:124-138`). Unsigned >8-bit maps lossily onto signed (`torch.py:95-98`).

### 1.2 `wp.to_torch`

`torch.py:315-358`. CUDA: `torch.as_tensor(a, ...)` via `__cuda_array_interface__`, zero-copy (`torch.py:347-355`). CPU: bounces through `numpy.asarray(a)` (`torch.py:336-345`) — the digest (`warp.md` §4.2) calls this "a hidden copy on CPU". **REFUTED**: warp CPU arrays expose `__array_interface__` with the raw pointer (`types.py:3382-3407`), `numpy.asarray` builds a view, and `torch.as_tensor(ndarray)` shares memory — the numpy hop is zero-copy too. Struct-dtype arrays cannot be converted (`torch.py:332-334`).

### 1.3 Gradient interop — `wp.array.grad` ↔ `tensor.grad` aliasing

`from_torch(t, requires_grad=True)` (default when `t.requires_grad`):

- If `t.grad` exists → wrap it as the warp array's `.grad` (`torch.py:266-277`). One buffer, two views: **warp's tape accumulates directly into torch's `.grad` storage.**
- If `t.grad` is None → **warp allocates the grad itself and mutates the torch tensor**: `t.grad = to_torch(grad)` (`torch.py:279-283`). This side effect is real (digest **VERIFIED**) and is also a performance trap: PyTorch defers grad allocation, and when it later finds an externally-allocated `.grad` it does a **device-wide synchronization** — Warp ships an entire case study on this (`interoperability.rst:551-946`) with measurements: baseline 98.02 ms → 22.1-28.6 ms after fixing (`interoperability.rst:918-931`). Fixes: pass `requires_grad=False` and manage grads as plain output tensors (Solution A), `.detach()` inputs (B), or pre-allocate `t.grad = torch.empty_like(t)` with torch's allocator before handing to warp (C — required when the backward uses `wp.Tape`).

### 1.4 The documented `torch.autograd.Function` pattern

Warp ships the pattern as docs + example, **not as a reusable helper** (digest **VERIFIED**: no `warp.torch_function(...)`; grep of `warp/_src` finds no `autograd.Function`/`custom_op` helper, while JAX *does* get one — `warp/_src/jax_experimental/ffi.py` `jax_kernel`, `interoperability.rst:975-1000`).

Three documented variants:

1. **Manual adjoint relaunch** (canonical, `warp/examples/core/example_torch.py:36-66`; docs `interoperability.rst:262-353`): `forward` does `from_torch` → `wp.launch(kernel, inputs, outputs)` → `to_torch`; `backward` sets `ctx.z.grad = wp.from_torch(adj_z)` and relaunches the *same kernel* with `adjoint=True, adj_inputs=[...], adj_outputs=[...]`. Forward/backward pairing is entirely manual; warp arrays must be parked on `ctx` to stay alive.
2. **`wp.Tape` inside the Function** (`interoperability.rst:848-896`): record launches on a tape in `forward`, `ctx.tape.backward(grads={out: wp.from_torch(grad_output)})` in `backward`, read grads off the aliased `.grad` tensors, then null `ctx.tape` to break cycles. Requires pre-allocated torch-side grads (Solution C above).
3. **`torch.library.custom_op` (torch ≥ 2.4)** (`interoperability.rst:355-466`): register forward and backward as separate custom ops with `register_fake` shape functions and `register_autograd`. This is the **torch.compile-safe** form — custom ops are opaque callables that `torch.compile(fullgraph=True)` treats as graph nodes instead of graph-breaking (`interoperability.rst:360-363`; pre-2.4 autograd.Functions are excluded from compiler optimization, `interoperability.rst:351-353`).

### 1.5 DLPack

`warp/_src/dlpack.py` (480 LOC). Unlike `from_torch`, **`wp.from_dlpack` does perform producer-side stream synchronization**: it passes Warp's current stream to the producer's `__dlpack__(stream=...)` so the producer orders its work before Warp's (`dlpack.py:458-474`; `interoperability.rst:1795-1808`). Exporting via legacy `wp.to_dlpack` capsules **skips** sync for speed (`interoperability.rst:1810-1827`). So: dlpack = safe-by-default; `from_torch` = fast-by-default.

---

## 2. Autodiff (`wp.Tape` + adjoint codegen)

### 2.1 How it works

- Every kernel gets a forward **and** a backward entry point generated at codegen time by default (`docs/user_guide/differentiability.rst:8`; `wp.config.enable_backward` default True, `warp/config.py:179`). The backward kernel **replays the forward body to rebuild locals in registers** (each forward statement is re-emitted into a `body_replay` block, `codegen.py:1374-1391`) and then runs adjoint statements in reverse — intermediates are recomputed, not stored, so per-kernel memory cost is only the `.grad` buffer per `requires_grad` array (≈2× array memory), *not* per-SSA-value.
- `wp.Tape` (`warp/_src/tape.py:13-155`) records `(kernel, dim, inputs, outputs, device, ...)` per launch; `backward()` walks the list in reverse relaunching each kernel with `adjoint=True`, wiring `adj_inputs/adj_outputs` from each array's `.grad` (`tape.py:143-155`). Grad seeds: scalar `loss.grad.fill_(1.0)` (`tape.py:94`) or a `grads={array: seed}` dict (used for Jacobian rows). **Gradients accumulate; `Tape.zero()` is mandatory between passes** (`differentiability.rst:34-35`, `tape.py:279-292`).
- Only **one tape** may be active at a time (module-global `runtime.tape`, `tape.py:50-64`).
- The `inputs=`/`outputs=` split on `wp.launch` is *purely a tape/tooling convention* — args are concatenated positionally for the kernel (`context.py:7500-7506`). Mislabeling silently corrupts nothing in forward but matters for graph-visualization and overwrite tracking.
- Launching a kernel whose module/kernel has `enable_backward=False` inside a tape produces a **warning, not an error**, and wrong gradients (`tape.py:113-122`).

### 2.2 What is differentiable / what silently breaks

Verified caveat list (all from `differentiability.rst` unless noted):

| Construct | Status | Evidence |
|---|---|---|
| Elementwise math, vec/mat/quat/transform/spatial ops | differentiable; adjoints hand-written in native headers | `warp/native/quat.h:157+`, `spatial.h:867+` |
| `wp.copy` / `wp.clone` / `array.assign` | differentiable, recorded on tape | rst:137-235 |
| In-place `+=` / `-=` and `wp.atomic_add` (fixed index) | supported; adjoint accommodates accumulation | rst:133-135, 1368-1399 |
| In-place `*=` / `/=` | **silently wrong** backward | rst:1400-1401 |
| Local vec/mat/quat **component re-assignment** | one assignment per component only; re-`=` invalidates grads (`+=`/`-=` fine) | rst:1403-1409 |
| **Dynamic loops** (`range(n)` with runtime `n`) | **not replayed/unrolled in backward** → wrong grads whenever adjoints need loop intermediates or final local values (worked example produces `[32, 8, 2]` instead of `[4,4,4]`, and an `inf` case) | rst:1411-1604 |
| Static loops ≤ `max_unroll` (default 16, `config.py:226`) | unrolled → correct | rst:1479-1482 |
| `wp.atomic_add` with **data-dependent index** (scatter/compaction) | adjoint pairs wrong threads → wrong grads; needs `@wp.func_replay` to cache indices | rst:500-636 |
| Write-after-read across or within kernels (buffer reuse) | wrong grads; detect with `wp.config.verify_autograd_array_access=True` (which **disables kernel caching**) | rst:38-135, 1232-1355 (note at 1349-1350) |
| `wp.tile_cholesky` etc. | **no adjoint** ("computing the adjoint is not yet supported", `is_differentiable=False`) | `builtins.py:12488-12516` |

Real-world consequence of the dynamic-loop rule, from Newton's FK kernel: *"unroll for loop to ensure joint actions remain differentiable (since differentiating through a for loop that updates a local variable is not supported)"* — D6 joint axes are manually unrolled 3× (`newton/_src/sim/articulation.py:281-320`).

### 2.3 Escape hatches — when hand-written backward is needed

- `@wp.func_grad(f)` replaces the generated adjoint of a `@wp.func`; `@wp.func_replay(f)` replaces its forward replay; `@wp.func_native(snippet, adj_snippet, replay_snippet)` for raw CUDA/C++ (`differentiability.rst:370-424, 638-856`). Not supported for generic (`Any`-typed) functions (rst:422-424).
- `wp.grad(func)` evaluates a *single function's* gradient inline in the forward pass (one kernel, all partials at once) — good for analytic per-thread Jacobian rows; single-output functions only; does **not** participate in reverse AD except inside `@wp.func_grad` (`differentiability.rst:306-368, 858-1025`).
- Evidence for when projects opt out of autogenerated adjoints:
  - **Iterative/implicit solves**: Newton registers a **no-op grad for the Cholesky factorization** and an implicit-function-theorem adjoint for the triangular solve (solve again with adjoint RHS): `newton/_src/solvers/featherstone/kernels.py:1466-1477` (`adj_dense_cholesky` = "nop, use dense_solve to differentiate through (A^-1)b = x") and `kernels.py:1544-1580` (`adj_dense_solve`: `adj_b = A^-T adj_x`, `adj_A -= adj_b xᵀ`). This is the exact pattern BetterRobot would need for LM/GN inner solves.
  - **Scatter with atomics** → `func_replay` (docs example above).
  - **Perf/fusion**: PyposeWarp hand-writes all 24 backward kernels with `enable_backward=False` everywhere (digest `PyposeWarp.md` §3.4 — not re-verified here, source not in tree).
- `warp.autograd` module = **gradcheck tooling**, not a torch bridge: `jacobian`, `jacobian_fd`, `gradcheck`, `jacobian_plot` (`warp/_src/autograd.py`, `differentiability.rst:1028-1069`). Ship-quality differential-testing utility; use it as the test oracle for any hand-written adjoint.

### 2.4 Jacobians and second order

- Full Jacobians = **one `tape.backward(grads={out: e_i})` per output row**, zeroing between rows (`differentiability.rst:237-304`); batched block-diagonal trick with tiled seed vectors for multi-env (rst:273-304). Newton's IK does exactly this (`newton/_src/sim/ik/ik_objectives.py:431, 706, 1057`), with an `ANALYTIC`/`MIXED` mode to replace autodiff rows with hand-coded ones (`ik_common.py:16-27`).
- **No second-order autodiff, no create_graph-style double backward.** The adjoint relaunches (`tape.py:143-155`) are not themselves recorded on any tape; adjoint kernels have no adjoint-of-adjoint codegen; zero hits for "second order"/"double backward" in source and docs. `wp.grad` is forward-evaluated and explicitly does not flow gradients (rst:955-958). Digest claim **VERIFIED**. Consequence for BetterRobot: any `torch.autograd.Function` wrapping a warp kernel must set `once_differentiable` semantics or hand-code a double-backward — gradients *of* gradients (e.g. differentiating through an LM step's J via autograd-of-autograd) cannot be delegated to warp.

---

## 3. CPU support

### 3.1 What works

- Every kernel is compiled for `"cpu"` via the embedded LLVM/Clang (statically linked into the warp DLL; `warp/AGENTS.md` "Codebase Internals"; `Module.load` CPU branch `context.py:3117-3123` loads the `.o` through `runtime.llvm.wp_load_obj`). The kernel *language* (types, builtins, quat/transform/spatial math, mesh/BVH queries, generated adjoints, `wp.Tape`) is device-neutral — the same generated C++ has both forward and backward CPU entry points (`codegen.py:4277-4344`).
- Even basic **tile** ops work on CPU: `tile_cholesky`/`tile_matmul` unit tests run on `all_devices` including cpu (`warp/tests/tile/test_tile_cholesky.py:990-1034`, `test_tile_matmul.py:216-222`), with a no-MathDx fallback GEMM (`test_tile_matmul_no_mathdx.py`). Some tile-linalg tests are CUDA-only (`test_tile_cholesky.py:1041-1081`).
- `capture_if`/`capture_while` degrade gracefully on CPU to eager host-side branching (`context.py:8664-8704`).

### 3.2 The threading story: **there is none**

The generated CPU module entry point is a **plain serial for-loop over the entire launch grid in one host thread**:

```c
// codegen.py:4298-4316 (cpu_module_template_forward)
for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {name}_cpu_kernel_forward(dim, task_index, _wp_args);
```

Same for backward (`codegen.py:4322-4341`). `wp.launch` forces `block_dim = 1` on cpu (`context.py:7473-7474`), and the CPU launch path is a synchronous ctypes call (`context.py:7386-7387`). No OpenMP, no thread pool anywhere in `warp/native/warp.cpp` (grep). Warp's CPU device is effectively a **single-core scalar interpreter of the kernel grid** (Clang may auto-vectorize within a task, but there is no multi-core parallelism).

### 3.3 Verdict for BetterRobot

- CUDA-only features: **CUDA graphs** (`capture_begin` raises "Must be a CUDA device", `context.py:8473-8474`), streams/events, mempools, peer access/IPC, conditional graph nodes, MathDx-backed tile perf. Tiles *function* on CPU but with `block_dim=1` semantics that change `wp.tile()` behavior (`docs/user_guide/tiles.rst:463-503`).
- Realistic expectation: a warp CPU kernel beats *eager* torch on small batches only because it fuses a whole pass into one C++ loop with zero dispatch overhead; at large batch it loses to torch's multithreaded/AVX kernels, and it will generally lose to `torch.compile`d CPU code (which BetterRobot already measured at 5× on the raw FK pass) because compiled torch also fuses *and* threads. **Do not plan on warp-CPU as the performance CPU path; keep it at most as a correctness/parity path, with torch as the primary CPU implementation** — consistent with the handoff constraint "CPU capability must not be lost".

---

## 4. Native SE3 / spatial types

All value types are ctypes PODs usable in kernels and as `wp.array` dtypes, in h/f/d (fp16/32/64) variants:

- `wp.quat` — **[x, y, z, w], scalar-last** (`warp/native/quat.h:10-17` constructor order; `types.py:1506-1535`). **Matches BetterRobot's convention.**
- `wp.transform` — 7-vector `p (3) + q (4)` with `q` defaulting to `(0,0,0,1)` (`types.py:1537-1560`; `native/spatial.h:130`). **Layout `[tx,ty,tz,qx,qy,qz,qw]` — identical to BetterRobot's SE3 pose.** A `(B,7)` torch pose tensor aliases directly as `wp.array[wp.transform]` via `from_torch(t, dtype=wp.transformf)` (§1.1).
- `wp.spatial_vector` (6) / `wp.spatial_matrix` (6×6) (`types.py:1799-1833`). **Convention: angular first** — `spatial_top` returns the `w` (angular) part, `spatial_bottom` the `v` (linear) part (`spatial.h:60-67`), and `spatial_jacobian` writes rows `[w(3); v(3)]` (`spatial.h:1155-1166`). **This is the OPPOSITE of BetterRobot's `[lin, ang]` tangent/Jacobian convention** — any warp kernel boundary must document/absorb the swap (or BetterRobot uses its own packing and only borrows the math functions that don't care).

Builtin function surface (all with hand-written native adjoints — `quat.h:157+` "adjoint methods", `spatial.h:867+`):

- Quaternion: `quat_identity`, `quat_from_axis_angle` (`builtins.py:1730`), `quat_to_axis_angle`, `quat_from_matrix`, `quat_rpy`, `quat_inverse`, `quat_rotate`, `quat_rotate_inv`, `quat_slerp` (`builtins.py:1811`), `quat_to_matrix` (`builtins.py:1819`).
- Transform: `transform_identity`, get/set translation/rotation, `transform_multiply` (`builtins.py:2046`), `transform_point`/`transform_vector` (`builtins.py:2053-2091`), `transform_inverse` (`builtins.py:2101`).
- Spatial: `spatial_dot`, `spatial_cross`, `spatial_cross_dual` (`builtins.py:2223-2230`), `spatial_adjoint(R,S)` building the 6×6 Plücker transform (`spatial.h:1073-1096`), plus articulated-specific `spatial_jacobian` (`builtins.py:2261`, `spatial.h:1128`) and `spatial_mass` (`builtins.py:2277`).
- `mat33` etc. with `inverse`, `svd3`, `eig3`, `qr3` and full arithmetic.

Notable gaps vs BetterRobot's lie layer: **no SE3/SO3 `log`/`exp`** builtins (no `quat_log`, no `transform_log`) — the geodesic ops BetterRobot uses for pose residuals and retraction still have to be written as `@wp.func`s (mjwarp likewise rolls its own `math.py` quaternion helpers, e.g. `mul_quat`, `rot_vec_quat`, `axis_angle_to_quat` used in `mjw/_src/smooth.py:112-135`). What warp gives for free is the *value types + arithmetic + adjoints*, not a Lie-group library.

---

## 5. CUDA graphs

- API: `wp.capture_begin`/`wp.capture_end` (`context.py:8430-8537`), `wp.ScopedCapture`, `wp.capture_launch` (`context.py:8903`), debug export `capture_debug_dot_print` (`context.py:8540`). CUDA-only (§3.3). Graphs re-execute the recorded kernel/memory ops with near-zero CPU launch cost.
- **Conditional nodes**: `wp.capture_if` / `wp.capture_while` (`context.py:8633-8790`) put data-dependent control flow *inside* the graph. Requires toolkit+driver ≥ 12.4 (`assert_conditional_graph_support`, `context.py:8552-8563`); outside capture they fall back to eager readback-and-branch (`context.py:8674-8704`), so the same code runs uncaptured/CPU.
- **Constraints**: fixed kernel set/shapes per capture (a graph is a frozen recording); no host readbacks or `wp.synchronize` inside capture; `verify_cuda` incompatible (`context.py:8466-8467`). **Allocation during capture IS allowed** when mempool allocators are active — captured allocs become graph-owned and only exist after `capture_launch` (`docs/deep_dive/allocators.rst:190-206`). Module loading during capture is fine on driver ≥ 12.3 (`context.py:8459-8464`). Mempool-to-mempool copies across GPUs during capture fail unless mempool access is enabled (`allocators.rst:246-268`).
- **Joint torch+warp capture** is documented and supported *if both run on the same non-default stream*: create a torch stream and bridge it with `wp.stream_from_torch`, or push warp's stream into torch with `torch.cuda.stream(wp.stream_to_torch(...))` (`interoperability.rst:113-180`). Many torch ops are not capturable; warmup is required (`interoperability.rst:177-180`).

How the simulators use graphs:

- **mujoco_warp**: the entire Newton/CG constraint-solver iteration loop is a `wp.capture_while(nsolving, while_body=_solver_iteration, ...)` — the GPU itself iterates until the per-world convergence counter hits zero (`mjw/_src/solver.py:3326-3337`), with an eager Python-loop fallback for JAX interop (`solver.py:3338-3343`) and a driver check warning below CUDA 12.4 (`mjw/_src/warp_util.py:144-156`).
- **Newton diffsim examples**: capture the **whole tape record + `tape.backward` (forward AND backward of a multi-step rollout) in one graph**, then replay per training iteration: `with wp.ScopedCapture(): self.forward_backward()` … `wp.capture_launch(self.graph)` (`newton/examples/diffsim/example_diffsim_ball.py:115-148`). This requires all intermediate states to be pre-allocated (one `State` per substep — `self.states[t]`, `example_diffsim_ball.py:139-143`) so the graph has fixed buffers and the tape has no overwrites. Odd-substep state ping-pong must `assign` (copy) instead of swapping references under capture (`newton/_src/sim/state.py:131-155`).

This is the pattern BetterRobot's optimizer loops would want: **graph-capture one (forward+backward) evaluation, `capture_while` the solver iteration.** Note the tension: capture requires fixed shapes and pre-allocated everything — reinforcing the plan's ModelValues/workspace design.

---

## 6. Codegen / caching / dtypes

- `@wp.kernel` **parses the function's source text** (`inspect.getsourcefile` + `ast.parse`, `codegen.py:955-977`) — kernels must live in real `.py` files (warp's own `AGENTS.md` bans `python -c` kernels). No tracing path; kernels are written, not derived.
- **Content-addressed caching**: `ModuleHasher` (`context.py:1930-2124`) hashes source + arg-type codes + struct hashes + referenced constants + `wp.static` expressions + module options; artifacts (`.cpp/.cu/.o/.ptx`) live under `kernel_cache_dir/wp_<hash>` (`build.py:156-203`). `Module.load` returns the cached exec when the hash matches, recompiles when it changed (`context.py:3042-3052`). First-ever launch of a module compiles (NVRTC / Clang); subsequent runs hit the disk cache. Compile latency was **not measurable here** (no runtime); warp maintains dedicated cold-start compilation benchmarks (`docs/deep_dive/codegen.rst:103-104`), and mjwarp's digest claim of "seconds of cold-start JIT for a step()" remains **UNVERIFIED but plausible**.
- **Silent-staleness trap**: closure variables that aren't Warp functions/structs/constants are *not hashed* — "Users should wrap such values with `wp.static()` to make them visible" (`context.py:2069-2073`). A Python int baked into a kernel factory can go stale without recompile.
- **Dynamic shapes do NOT recompile.** Array shape/strides are runtime fields of `array_t` (`torch.py:289`), launch `dim` is a runtime argument — one compiled kernel serves all batch sizes and robot sizes. Recompilation triggers only on: source change, new generic-overload type signature (`Kernel.add_overload`, `context.py:812-856` — first launch of a new dtype combo stalls; pre-declare with `@wp.overload`, `docs/user_guide/generics.rst:96-132`), struct layout change, `wp.constant` change, module options, and **`block_dim`** (module hash keyed per block_dim, `context.py:2690`, execs keyed `(device.context, block_dim)`, `context.py:3043`). Tile *shapes* are compile-time constants — the tile path is where shape specialization lives.
- **Multi-process**: `clear_kernel_cache()` is not multi-process-safe (warp `AGENTS.md`); forked child processes can't reuse parent CUDA contexts (`docs/user_guide/limitations.rst:78-81`).
- **fp32 vs fp64**: virtually all builtins are generic over `Float` (fp16/32/64) — quat/transform/spatial ship `h/f/d` variants (`types.py:1799-1833`); `tile_cholesky` is fp32/fp64 (`builtins.py:12505-12507`); `atomic_add` fp16 needs sm_70+ (`limitations.rst:25-27`); **no bfloat16 anywhere** (`torch.py:134-137`). BetterRobot's fp32-primary/fp64-parity policy maps cleanly. Note both simulators run fp32-only in practice (mjwarp: "C MuJoCo tolerance was chosen for float64 architecture, but we default to float32 on GPU", `mjw/_src/io.py:182`).
- Module options worth planning for: `enable_backward`, `fast_math`, `fuse_fp` (FMA contraction — determinism knob), `lineinfo`, `max_unroll`, `mode=debug/release`, `block_dim`; per-kernel via `@wp.kernel(module="unique", ...)` (used by every mjwarp kernel factory).

---

## 7. Kernel architecture in mujoco_warp and Newton

### 7.1 mujoco_warp (the performance-first reference)

- **Granularity**: many small kernels per pass, orchestrated by thin Python launchers. `kinematics()` = 5 launches (branch FK, body mats, inertial frames, geoms, sites) (`mjw/_src/smooth.py:358-415`); `rne()` = 5 sub-stages (`smooth.py:1259-1274`). 232 kernels total across the package (digest count, spot-checked).
- **Thread mapping**: 2-D grid `(worldid, entityid)` almost everywhere — per-(world, branch) for FK, per-(world, body), per-(world, dof), per-(world, geom). Batch (`nworld`) is always the leading grid dim and leading array dim.
- **Tree dependencies — two mechanisms, both host-free:**
  1. **Down-tree (FK, cvel, cacc): branch-parallel, sequential-in-kernel.** A "branch" is a *complete root→leaf ancestor chain*, one per leaf body (`ancestor_chain` construction, `mjw/_src/io.py:241-258`). `_kinematics_branch` loops `for i in range(start, end)` over its chain, composing parent→child in registers (`smooth.py:70-143`). Interior bodies shared by several leaves are **recomputed redundantly by each leaf's thread** (identical values → benign duplicate writes) — redundant compute is traded for zero synchronization and a single launch. Same pattern for velocity/acceleration propagation (`_comvel_branch`, `smooth.py:2015+`; `_cacc_branch`, `smooth.py:1167-1184`).
  2. **Up-tree (reductions: subtree COM/mass, RNEA force back-propagation): level-order sweeps.** Bodies are grouped by depth into `m.body_tree: tuple[wp.array[int], ...]` (`io.py:234-239`); the launcher does **one `wp.launch` per depth level** in reverse order (`_rne_cfrc_backward`: `for body_tree in reversed(m.body_tree): wp.launch(...)`, `smooth.py:1236-1241`; `subtree_com`, `smooth.py:609-620`).
- **Joint-type dispatch**: plain runtime `if jnt_type_ == JointType.FREE/BALL/SLIDE/HINGE` chains on an int array *inside* the kernel (`smooth.py:83-135`). No per-joint-type kernels, no sorting by type. (Warp branches are cheap; warps in a world/branch mostly agree.)
- **Static specialization where it pays**: `@cache_kernel` factories close over compile-time config (solver flavor, cone type, sparse/dense, `nv`) and emit `@wp.kernel(module="unique", enable_backward=False)` specializations, memoized in a module-global dict (`mjw/_src/warp_util.py:122-141`; 10 factories in `solver.py`).
- **Per-world parameter broadcasting**: model fields that *may* be per-world are indexed `arr[worldid % arr.shape[0]]` so shape-(1,…) shared params and shape-(nworld,…) randomized params use the same kernel (`smooth.py:94, 108, 129`).
- **Differentiability: deliberately OFF everywhere.** `wp.set_module_options({"enable_backward": False})` at the top of every physics module (`smooth.py:41`, `solver.py`, `forward.py`, …) and README: "Differentiability via Warp is not currently available" (`mujoco_warp/README.md:76-77`). The pipeline is *not* an existence proof of differentiable warp robotics — it's an existence proof that a large warp codebase found autodiff not worth the memory/complexity.
- **Torch mixing: none.** Zero `import torch` in `mjw/_src` (grep). Pure warp + numpy; interop offered via JAX unroll contrib only.

### 7.2 Newton (the differentiable-but-slower reference)

- **FK granularity is coarse**: `eval_fk` launches **one thread per articulation**, which loops sequentially over *all* joints of that articulation in topological order (`newton/_src/sim/articulation.py:377-447` kernel; launch `dim=num_articulations`, `articulation.py:487-516`; the per-joint loop is `eval_single_articulation_fk`, `articulation.py:213+`). No branch- or level-parallelism. Fine for thousands of robots, terrible for few-robot/large-batch-q workloads unless the batch = articulations.
- **Jacobian**: same per-articulation threading; each thread walks `while j != -1` up the parent chain filling J rows from motion subspaces (`articulation.py:998-1050`) — the logic of the `spatial_jacobian` builtin (§4).
- **Joint dispatch**: runtime `if type == JointType.X` chains, one branch per type (`articulation.py:235-320`), with the D6 loop manually unrolled *specifically for autodiff correctness* (`articulation.py:281-283`).
- **Dynamics (Featherstone)**: composite H = JᵀMJ per articulation, then either (a) default: batched **single-thread dense Cholesky + solve per articulation** (`eval_dense_cholesky_batched` / `eval_dense_solve_batched`, one `wp.tid()` per articulation, `featherstone/kernels.py:1481-1496, 1573-1588`), or (b) opt-in `use_tile_gemm=True`: `wp.launch_tiled` per articulation with `wp.tile_matmul` + `wp.tile_cholesky`, optionally fused into one kernel (`create_inertia_matrix_cholesky_kernel`, `kernels.py:1358-1400`; dispatch `solver_featherstone.py:116-147, 668-700`).
- **Differentiability end-to-end: verified real, with hand-holding.** All solver workspaces allocated with `requires_grad=model.requires_grad` (`solver_featherstone.py:303-357`); the linear solve is made differentiable by *custom* grads (nop-Cholesky + implicit solve adjoint, §2.3), including an extra `joint_solve_tmp` buffer allocated only when `requires_grad` (`solver_featherstone.py:339-343`); diffsim examples run multi-step rollouts under `wp.Tape` with one pre-allocated `State` per substep and graph-captured forward+backward (§5). Only Featherstone/SemiImplicit/XPBD are differentiable; MuJoCo/VBD/MPM/Style3D solvers are not (`newton/solvers.py` feature matrix — digest, spot-consistent with the custom-grad evidence).
- **Torch mixing: none in core.** `import torch` appears only in kamino RL example glue that wraps warp kernels for torch tensors (`newton/_src/solvers/kamino/examples/rl/utils.py:17-45`). Gradients are `wp.Tape`-native; Newton ships **no** `torch.autograd.Function` bridge.
- **IK**: pure-warp LM and L-BFGS with pluggable Jacobian modes — `AUTODIFF` (tape backward per residual row with basis seeds, `ik_objectives.py:431, 706, 1057`), `ANALYTIC`, `MIXED` (`ik_common.py:16-27`, `ik_solver.py:234`).

### 7.3 The transferable pattern for BetterRobot

Both projects agree on the shape of the answer even where they differ on details:

1. **Whole-pass kernels, never per-op** — one launch (or a handful) per pipeline stage, sequential tree recursion *inside* the kernel body, everything in registers between joints.
2. **Batch = leading grid dim + leading array dim**; per-batch model params via stride-0/modulo indexing.
3. Tree parallelism menu: per-articulation serial (Newton — simplest, differentiable, coarse) → per-(world, root-to-leaf-branch) with redundant recompute (mjwarp — no sync, more parallelism) → level-order multi-launch for up-tree reductions (mjwarp). For BetterRobot's few-robots × huge-B IK/trajopt workloads, the batch dim itself provides the parallelism, so **Newton-style per-(batch-element) serial sweeps are likely sufficient and are the differentiable-safe choice** (static topo loop unrolled over `model.topo_order`, matching BetterRobot's existing FK structure).
4. Joint dispatch: runtime int `if` chains are the accepted practice in both codebases; compile-time specialization is reserved for solver-level config (mjwarp `cache_kernel`).
5. Differentiability must be designed in: no buffer reuse across a taped pass, workspaces pre-allocated with `requires_grad`, custom adjoints at linear-solve boundaries, dynamic loops avoided or pushed into `@wp.func`s.

---

## 8. Small dense linear algebra — where the boundary sits

- **Warp-native options**: (a) tile API — `tile_matmul`, `tile_transpose`, `tile_cholesky`, `tile_cholesky_solve`, `tile_lower_solve`, `tile_upper_solve`, `tile_diag_add`, `tile_fft` (`builtins.py:12105, 12488, 12693`; `docs/user_guide/tiles.rst:400-416`), backed by MathDx/cuBLASDx LTO on CUDA (needs CUDA ≥ 12.6.3 at build for full support, `tiles.rst:714-727`) with basic-op CPU fallbacks (§3.1); **`tile_cholesky` has no adjoint** (§2.2). (b) per-thread scalar loops (write your own factorization in a kernel — Newton default). (c) `warp.optim.linear` iterative solvers: `cg`, `bicgstab`, `gmres` over a `LinearOperator` (`warp/_src/optim/linear.py:21+`). (d) `warp.sparse` BSR/CSR mat-vec and mat-mat.
- **Where the projects put the boundary: entirely inside warp — neither calls torch/cusolver for anything.**
  - mujoco_warp: mass-matrix factorization = `wp.tile_cholesky` per world (`mjw/_src/smooth.py:1068-1104`), back-substitution = `wp.tile_cholesky_solve` (`smooth.py:2783-2811`), fused factorize+solve variant (`smooth.py:2861-2899`); Newton-solver Hessian JᵀDAJ assembled with `launch_tiled` shared-memory kernels and solved with a custom **blocked Cholesky built from tile primitives** for sizes beyond one tile (`mjw/_src/block_cholesky.py:22-60`).
  - Newton: hand-rolled per-articulation dense Cholesky in a single thread as the default, tile path opt-in (§7.2) — chosen so it can be *differentiable* via custom grads, which the tile Cholesky is not.
- Implication for BetterRobot's LM/GN: the normal-equation solve per batch element (nv×nv, tens of dofs) fits either pattern. If gradients must flow through the solver step (they do for BetterRobot), the Newton recipe — factorization treated as non-differentiable + implicit-diff adjoint on the solve — is the proven approach (`kernels.py:1466-1580`); a torch fallback (`torch.linalg.cholesky_ex` batched) remains reasonable at the boundary since the solve is one op on `(B, nv, nv)` tensors, but no warp project needed it.

---

## 9. Failure modes when torch and warp cohabit

### 9.1 Stream races (the silent one)

- `from_torch`/`to_torch` never synchronize (§1.1). Warp launches go to **warp's per-device stream**, created with `CU_STREAM_DEFAULT` — a *blocking* stream (`warp/native/warp.cu:2540`). Torch eager ops run on torch's current stream, which is the **legacy default stream** by default. Because legacy-default ↔ blocking-stream ordering is implicit in CUDA, the naive `torch op → wp.launch → torch op` pattern is ordered correctly *as long as torch stays on its default stream*. It breaks silently the moment torch code runs on a side stream (DDP, `torch.cuda.Stream`, graph capture, some compiled/AMP paths) — then you must bridge: `wp.stream_from_torch(torch.cuda.current_stream())` / `wp.stream_to_torch` (`torch.py:361-398`) or share one stream explicitly (docs pattern, `interoperability.rst:108-180`). DLPack path is the safe-by-default alternative (§1.5).
- Warp's own guidance hierarchy: default streams < `ScopedStream` (syncs on enter by default) < explicit stream args (`docs/deep_dive/concurrency.rst:598-622`).

### 9.2 The deferred-grad sync trap (the measured one)

Fresh `requires_grad=True` tensors without `.grad` passed to `from_torch` cause warp to allocate the grad and attach it; PyTorch then does **device-wide syncs** when it discovers externally-allocated grads — 4.3× slowdown on warp's own 300M-element benchmark, hit *every iteration* when training loops make fresh tensors (`interoperability.rst:551-946`). Fixes in §1.3. This is the single most likely perf bug in a naive BetterRobot warp bridge.

### 9.3 Grad double-accumulation

Warp's tape accumulates into the buffer aliased as `t.grad`; torch's engine *also* accumulates whatever `Function.backward` returns into `t.grad`. Mixing patterns (aliased grads *and* returned grads for the same leaf) double-counts. The shipped patterns avoid it by picking one channel: manual-adjoint examples return `wp.to_torch(ctx.x.grad)` on freshly zeroed warp-owned grads (`example_torch.py:50-66`); the tape variant reads `ctx.a.grad` and returns it, having pre-allocated it torch-side (`interoperability.rst:884-896`). A BetterRobot wrapper must standardize one convention and zero warp-side grads every call (`tape.zero()`, §2.1).

### 9.4 Allocators & lifetime

- Two pool allocators coexist (torch caching allocator, warp mempool) carving up the same GPU; neither documented issue nor coordination exists — expect fragmentation pressure at high memory use (inference from design; **not documented**, flagged as such).
- Lifetime is reference-glued: `from_torch` pins the tensor on the array (`torch.py:310`), ctype descriptors pin `._ref/._gradref` (`torch.py:292-293`), `to_torch` relies on `torch.as_tensor` holding the producer (`torch.py:347-351`), stream bridges pin each other (`torch.py:376, 396`). Dropping the wrapper objects while the consumer lives = use-after-free; keep conversions cached on the module/`ctx`.
- Fork-safety: CUDA contexts don't survive `fork` (`limitations.rst:78-81`); `clear_kernel_cache` not multi-process-safe (warp `AGENTS.md`).

### 9.5 Debugging story

Weak but tooled: errors reference generated C++/CUDA source in the kernel cache, `lineinfo`/`mode="debug"` module options, `wp.config.print_launches` (`context.py:7481-7482`), `verify_autograd_array_access` overwrite detector (disables caching, `differentiability.rst:1344-1350`), `wp.autograd.gradcheck/jacobian_plot` (§2.3), tape visualization (`tape.py:302+`), graph DOT export (`context.py:8540`). mjwarp found it necessary to build a custom AST linter for kernel signatures (`mjw/contrib/kernel_analyzer` — digest, dir confirmed present).

---

## 10. Digest-claim scorecard (deltas only)

| Claim (digest) | Verdict |
|---|---|
| `to_torch` on CPU is a hidden copy (`warp.md` §4.2) | **REFUTED** — numpy bounce is zero-copy (`types.py:3382`, §1.2) |
| No torch-autograd helper shipped; every project rewrites the wrapper (`warp.md` §4.5) | **VERIFIED**, with the addition that docs now ship a `torch.library.custom_op` pattern that is torch.compile-safe (`interoperability.rst:355-466`) — the digest missed this |
| No second-order / double backward (`warp.md` §5.4) | **VERIFIED** (§2.4) |
| mjwarp: `enable_backward=False` everywhere, 0 tapes (`mujoco_warp.md` §6) | **VERIFIED** (`smooth.py:41`, README:76-77) |
| mjwarp solver loop runs as CUDA-graph conditional (`mujoco_warp.md` §5) | **VERIFIED** (`solver.py:3326-3343`) |
| Newton diffsim = tape + Featherstone/SemiImplicit, no torch bridge (`newton.md` §7) | **VERIFIED** (§7.2) |
| Warp CPU kernels exist, so CPU path is "covered" (handoff assumption) | **NEEDS QUALIFICATION** — single-threaded serial loop, no multicore (§3.2, `codegen.py:4312-4315`) |
| PyposeWarp 13.3k LOC / 12× amplification for per-op kernels | **UNVERIFIED here** (source not in tree) but consistent with everything above; the per-op strategy contradiction with whole-pass kernels stands regardless |
| Kernel cold-compile costs "seconds" | **UNVERIFIED** (no runtime on this box); mechanism + disk cache verified (§6) |

---

## 11. Implications for the Warp-first plan (condensed)

1. **The M1 seam (ModelStructure flat index tensors + tensor-pytree ModelValues + pure whole-pass functions) is exactly what warp kernels consume** — mjwarp's `Model` is precisely flat index arrays + per-world value arrays, and both simulators pass them as long explicit kernel arg lists. Add to the seam: (a) fixed shapes per (model, batch) so graphs can capture; (b) `[ang, lin]` vs `[lin, ang]` adapter decision at the boundary (§4); (c) `(B,7)` pose tensors stay directly aliasable as `wp.transform` arrays (§4).
2. **Kernel boundary = one `torch.library.custom_op` (or autograd.Function) per whole pass** (`fk`, `fk+frame-jacobians`, `rnea`, residual+J stacks), backward = either warp adjoint kernel relaunch or `wp.Tape` recorded inside forward; never per-Lie-op. Pre-declare all dtype overloads at import (§6).
3. **Autograd budget**: warp gives first-order reverse only. BetterRobot needs grads w.r.t. q *and* model values — warp handles that (grads flow to any `requires_grad` array, including placements/inertias passed as inputs), but **no double backward** means any API promising `create_graph=True` must keep a torch reference path.
4. **CPU**: warp-CPU is correctness-only (single-threaded). Keep torch (+`torch.compile`) as the CPU implementation; the tensor-in/tensor-out seam makes the dispatch trivial.
5. **Linalg**: keep LM/GN solves in torch initially (`(B,nv,nv)` batched Cholesky is one op); if/when moved into warp, use Newton's nop-factorization + implicit-solve-adjoint recipe (`featherstone/kernels.py:1466-1580`), not `tile_cholesky` (no adjoint).
6. **Interop hygiene to bake into the bridge from day one**: convert once and cache warp views; `requires_grad=False` on all `from_torch` calls with explicit grad buffers (or pre-allocate `.grad` torch-side); bridge `torch.cuda.current_stream()` on entry; `tape.zero()` per call; hold references on `ctx`.
