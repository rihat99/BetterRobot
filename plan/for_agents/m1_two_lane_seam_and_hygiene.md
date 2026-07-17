# M1 — Structure, the Two-Lane Seam, and Hygiene: Agent Execution Instructions

> **Implementation log (2026-07-17):** completed on `dev`; Apache-2.0 and
> eager-CPU evidence were owner-approved. Final suite: 992 passed, 3 skipped.
> Key deviations: source LOC is +474, and Warp backward uses a Torch-recompute VJP.

> Read `plan/for_agents/README.md` first. It carries the standing rules
> (deletion ordering, honesty rules, the five-part kernel requirement,
> the GPU/CUDA caveat, test commands). This file does not repeat them.

> **Update (owner decision 2026-07-17 — branch strategy):** the redesign
> is implemented on a dedicated branch; BHF/BVR stay on the pre-redesign
> branch until the M4 migration. Shim-keeping instructions in this file
> are superseded: in T1.8, `costs/` and the `Data` aliases may be deleted
> outright, and the "both consumer repos still import cleanly" done-when
> no longer applies — instead, record every removed consumer-facing
> symbol for M4's migration table.

## Mission

M1 replaces the dead `backends/` Protocol layer with the real compute
seam BetterRobot needs for both PyTorch and Warp: a `ModelStructure`
(frozen topology as *both* Python static mirrors and flat device tensors)
plus a `ModelValues` tensor pytree, with the whole-pass functions
(`forward_kinematics_raw`, dynamics passes) re-signed as pure functions
over those types. On top of that seam we prototype the torch↔Warp bridge,
land the first real Warp kernel (a fused FK sweep) as an opt-in prototype,
freeze the execution-batch ABI that every later kernel and every batched
Model consumes, write the cross-cutting engineering contract (dtype,
quaternion, threading, differentiation, license), and clear a backlog of
hygiene debt: hoist per-call constants, widen the hot-path lint, delete
zero-consumer dead code, fix the un-buildable `ModelBuilder` kinds, add
extras packaging + CI, and cache `Data` per solver iterate. When M1 is
done the library no longer lies about its architecture, the seam is
proven consumable by an actual kernel, and the ABI/contracts that M2–M6
build on are frozen.

## Prerequisites

- **M0 must be complete** (`plan/for_agents/m0_truth_and_correctness.md`).
  In particular the θ=0 NaN-gradient fix (safe-`where` in
  `lie/_torch_native_backend.py`, `lie/tangents.py`,
  `JointSpherical.difference`) must have landed — the FK-kernel gradcheck
  in T1.3 and the bridge gradcheck in T1.2 both differentiate through
  exp/log at singular points and will produce NaNs without it.
- **Confirm the starting tree** before touching anything:
  - `find src/better_robot/backends -name '*.py' | xargs wc -l | tail -1`
    → **527 total** (10 files, incl. `backends/warp/bridge.py` (21 LOC),
    `backends/warp/__init__.py` (21 LOC), `backends/warp/kernels/__init__.py`
    (8 LOC)).
  - `uv run pytest tests/ -q` → **897 pass** (~52 s). This is the floor;
    it stays green through every task except where a task explicitly
    retires a contract test.
  - `grep -rn "graph_capture" src/` → hits only
    `backends/__init__.py:110` (the no-op function) and `:138` (its
    `__all__` entry). **There is no `graph_capture.py` file** — the plan
    text implies one; the surface is that function inside `backends/`.
  - `git log --oneline -1` shows commit `93b8c03` (`so3_inverse`
    per-call-constant fix) already landed — do not re-do it.

## Sizing & parallelism

Realistically **4–6 weeks solo** (the roadmap's earlier "~2 weeks"
estimate is retired). Effort per task, from `plan/04_roadmap.md` M1:

| Task | Effort | Depends on |
|------|--------|-----------|
| T1.1 seam + delete `backends/` | **L** | M0 |
| T1.2 bridge prototype | **M** | T1.1 (seam types), remote CUDA runner |
| T1.3 first Warp FK kernel | **M–L** | T1.1, T1.2 (bridge pattern) |
| T1.4 contract/docs migration | **S–M** | T1.1 |
| T1.5 engineering contract + LICENSE | **M** | — (parallel, owner STOPs) |
| T1.6 hoist per-call constants | **M** | T1.1 (axes into ModelStructure) |
| T1.7 extend hot-path lint | **S** | T1.6 (fixes land first) |
| T1.8 de-bloat zero-consumer surfaces | **S** | — (parallel) |
| T1.9 fix ModelBuilder kind mapping | **S–M** | — (parallel) |
| T1.10 extras packaging + CI | **S–M** | most others (lands near end) |
| T1.11 cache `Data` per iterate | **S** | — (parallel) |

**Ordered spine:** T1.1 → T1.2 → T1.3. T1.4 follows T1.1. T1.6 → T1.7
(pair; the lint must not be widened before the fixes land, or existing
violations fail CI). T1.10 lands near the end (CI green needs the tree
stable; `pyproject` is also touched by T1.2's torch floor and T1.5's
license — coordinate the edits).

**Parallel-safe** (hand to separate agents, independent of the spine):
T1.5, T1.8, T1.9, T1.11. T1.4 can begin its *inventory* early but its
final grep-clean acceptance needs T1.1 merged.

The ABI frozen in **T1.1** is consumed by
`m3_parametric_model_breadth.md` (which extends breadth *under* this ABI,
never changing it) and `m6_warp_fast_path_and_cuda_graphs.md`. The bridge
pattern from **T1.2** and the FK kernel from **T1.3** are picked up for
CUDA validation, benchmarking, and the default-on decision in M6.

---

## Tasks

### T1.1 — Design the two-lane seam, then delete `backends/`  [L]

**Goal / done-when (verbatim from roadmap):** the layer test is updated
and has no `backends` node; the parity suite is green; an **eager-CPU
non-regression benchmark is committed**; a layout test pins
`transformf`/`transformd` aliasing by **pointer and stride**; the ABI
tests (batched-q × unbatched-values, the converse, multi-axis, mismatch
errors) pass.

Expanded, concretely checkable:
1. `src/better_robot/backends/` no longer exists; `br.set_backend`,
   `br.default_backend`, `graph_capture`, and the `backend=` kwarg are
   gone from every signature (grep `backend=` in `src/` returns only
   unrelated matches, e.g. `build-backend`).
2. `lie/se3.py` / `lie/so3.py` call the implementation directly; the
   implementation module is renamed `lie/_torch_native_backend.py` →
   `lie/_impl.py` (per 03 §2).
3. `ModelStructure` exists with the **dual representation** (Python static
   mirrors + flat device tensors) and a consistency test between them.
4. `ModelValues` exists as a tensor pytree; SE3 poses alias
   `wp.transform` arrays zero-copy (tested by pointer+stride, gated on
   the `warp` extra being importable — skip cleanly if not).
5. The execution-batch ABI is **frozen** (flat `E`, per-input batch-index
   maps, in-kernel shared-value gradient reduction) and the four ABI test
   cases pass.
6. `forward_kinematics_raw` (and the dynamics passes) are re-signed as
   pure functions over `(ModelStructure, ModelValues, q, …) → tensors`.

**Current state (verified):**
- The dispatch chain is dead weight: `lie/se3.py:22` already imports
  `_torch_native_backend as _be` and calls it directly for constructors
  (`identity`, `from_*`), but dispatched ops route
  `_lie(backend).se3_compose(...)` through `default_backend()`
  (`se3.py:28-29,49`). `DynamicsOps` has zero call sites; `KinematicsOps`
  has two, both bouncing back into their own package (01 §2.1).
- The `*_raw` functions are **not yet true pass boundaries** (03 §2.7,
  codex #14). `forward_kinematics_raw` (`kinematics/forward.py:73`)
  consumes a Python `Model` and dispatches per joint through Python
  `JointModel` objects; `rnea` (`dynamics/rnea.py:71`) *mutates* `Data`
  in place — verified: it writes `data.joint_pose_world/joint_pose_local`
  (`:109-110`), `data._kinematics_level = 1` (`:111`, a raw int, not the
  enum), `data.tau` (`:205`), `data.joint_forces` (`:208`), etc.
- Structure the seam must expose (from the inventory audit §4): today
  `parents/children/subtrees/supports/topo_order`, `nqs/nvs/idx_qs/idx_vs`
  are **Python tuples**; `joint_models` is a tuple of frozen dataclasses
  whose discriminator is a **string** `kind`; joint axes live as
  module-level CPU tensors inside those objects; `frames` is a tuple of
  `Frame` dataclasses holding `(7,)` CPU tensors not moved by
  `Model.to()`. Already-flat device tensors: `joint_placements (njoints,7)`,
  `body_inertias (nbodies,10)`, the limit vectors, `gravity`, `q_neutral`.
- Warp layout facts that make the aliasing work (audit_warp_platform §4;
  codex_warp_review #13): `wp.quat` is scalar-last `[x,y,z,w]` and
  `wp.transform` is `[tx,ty,tz,qx,qy,qz,qw]` — **bit-identical to BR's SE3
  pose**. `wp.transform` is an fp32 alias; **fp64 needs `wp.transformd`**.
  Aliasing precondition: the trailing value-type dimension must be
  contiguous (inner stride); outer dims may be strided. `wp.spatial_vector`
  is **angular-first** — the OPPOSITE of BR's `[lin, ang]`; never
  reinterpret spatial memory across that boundary.

**Implementation plan:**
1. **Build `ModelStructure`** (new file, `data_model/model_structure.py`;
   apply the `file-naming` skill). Frozen dataclass carrying BOTH:
   - *Static mirrors* (what `torch.compile` specializes on): the existing
     Python tuples `topo_order/parents/nqs/nvs/idx_qs/idx_vs` and per-joint
     int kind codes — kept as tuples/ints, because the measured 5× compile
     win came from static unrolling of exactly these (03 §2.1
     static-topology caveat; codex_warp_review #2).
   - *Flat device tensors* (what kernels index): int8 joint-kind codes,
     int32 parent/index tables, CSR topology caches, packed joint axes,
     built once from the mirrors at construction (03 §2.4).
   - A **consistency test** asserting the two representations encode the
     same tree (parents, kinds, idx maps agree element-for-element).
   Do **not** delete the static form (codex #2, #14): a tensor-only torch
   recursion is a different algorithm with no benchmark behind it. Keep
   the dual form; drop it only if a later prototype shows a tensor-only
   torch FK compiles clean *and* does not regress.
2. **Build `ModelValues`** (`data_model/model_values.py`): a tensor pytree
   of `joint_placements`, `body_inertias`, `frame_placements` (stack the
   per-`Frame` `(7,)` tensors into `(nframes,7)` + a `(nframes,)`
   parent-joint index on `ModelStructure` — this also fixes the
   `Model.to()`-forgets-frames bug), limits, `q_neutral`. Poses are stored
   `[...,7]` scalar-last so they alias `wp.transformf`/`wp.transformd`
   zero-copy. `Model` remains the user-facing pairing of structure+values
   (03 §5); this M1 task builds the split and re-signs the passes, but the
   *public* `with_values(...)` / batched-value breadth is **M3** — do not
   build batched-value coverage here beyond the ABI tests.
3. **Freeze the execution-batch ABI** (03 §2.4, codex #3 — this is an ABI,
   not breadth, so it is frozen *now*, not in M3). At the seam, flatten any
   public batch `(B...,·)` into one flat execution batch of size `E` (the
   broadcast of the q batch with every value batch). Kernels/passes see
   `E` plus a **per-input batch-index map** for value arrays that were not
   expanded — a shared value is passed once and indexed, never physically
   repeated `E` times. The reverse rule is part of the ABI: the gradient
   of a value shared across `E` is **reduced inside the kernel path**
   (atomically or by segment), never returned expanded for torch to
   re-reduce. Write this as a small documented helper
   (`flatten_execution_batch` / `unflatten`) plus the reverse-reduction
   contract, and freeze the signatures.
4. **Re-sign the passes as pure functions.** Change
   `forward_kinematics_raw` and the dynamics passes to consume
   `(ModelStructure, ModelValues, q, …)` and **return** fresh tensors
   instead of mutating `Data`. `rnea` currently mutates `Data`
   (`rnea.py:109-208`) — return the FK poses and force/tau tensors and let
   the caller populate `Data`. Keep the public `forward_kinematics(model,
   q)` wrapper and `Data` for backward compatibility; internally it calls
   the pure raw pass then fills `Data`.
5. **Delete `backends/` entirely** (all 10 files, 527 LOC), including the
   `backends/warp/` stub tree — `bridge.py`, `__init__.py`, `kernels/`.
   Retarget `lie/se3.py`/`so3.py` to import the renamed `lie/_impl` and
   call it directly; delete the `_lie()` helper, the `backend=` kwargs, the
   `default_backend` import. `forward_kinematics`/`compute_joint_jacobians`
   already call their `_raw` siblings — verify no lingering
   `default_backend`/`set_backend`/`graph_capture` references anywhere
   (`grep -rn 'set_backend\|default_backend\|graph_capture\|from ..backends\|from .backends' src/`).
   The `backends/warp/` stub's *continued presence* is used by
   `m6_warp_fast_path_and_cuda_graphs.md` as a tripwire that M1 never
   landed — its deletion here clears that tripwire.
6. **Update the layer-DAG contract test** (`tests/contract/test_layer_dependencies.py`):
   remove the `"backends": 0` entry from `LAYER_RANK` (line 25). Leave the
   `costs` node in place — merging `costs/` into `optim/` is **M2c**, not
   M1. (Note: the `utils` rank-1 entry becomes dead after T1.8 deletes
   `utils/`; harmless, but remove it in whichever task lands second.)

**What to test:**
- `tests/contract/test_layer_dependencies.py` — updated, still green, and
  asserts there is no `backends` layer. Add a positive assertion that
  `src/better_robot/backends` does not exist.
- `tests/data_model/test_model_structure.py` (new): the dual-representation
  **consistency test** — for the Panda and the SMPL-like body, the flat
  device tensors (kind codes, parents, idx maps, packed axes) decode back
  to the Python static mirrors exactly.
- `tests/data_model/test_layout_aliasing.py` (new, `@pytest.mark.skipif`
  warp not importable): a `(B,7)` fp32 pose tensor aliased as
  `wp.array[wp.transformf]` shares `data_ptr()` and matching strides;
  same for fp64 via `wp.transformd`. **Assert pointer AND stride
  equality, not just values** (codex #13).
- `tests/data_model/test_execution_batch_abi.py` (new): the four ABI
  cases — (a) batched-q × unbatched-values, (b) batched-values ×
  unbatched-q (the converse), (c) multi-axis batches, (d) mismatched
  batch shapes raise a clear error. Include a gradient-reduction check: a
  value shared across `E` receives one correctly-summed gradient, not an
  expanded one.
- The **pinocchio-parity suite** (`tests/test_pinocchio/`) stays green
  through the pass re-signing — it is the safety net for this refactor.
- **Eager-CPU non-regression benchmark** (committed): a benchmark
  definition file under `benchmarks/` (hardware, dtype, robot, batch
  shapes, warmup, statistics — per README rule 4) measuring eager-torch
  FK/RNEA latency before vs after the seam change. Eager torch is the
  supported performance floor (03 §2.5); the dual representation exists
  precisely so this does not regress. **Produce the numbers and STOP for
  owner review** — do not silently accept a regression.

**Pitfalls / do-not-forget:**
- Keep BR's `[lin, ang]` spatial ordering; do not adopt warp's
  angular-first `spatial_vector` layout anywhere in `ModelValues`.
- The `rnea` `_kinematics_level = 1` raw-int write (`rnea.py:111`) is a
  latent invariant bug — when re-signing, use the enum, don't propagate
  the raw int.
- `Model.to()` must move `frame_placements` (the old bug: it moved 15
  fields but not `frames`). Fold this into the ModelValues stacking.
- Consumer imports that must keep working after the delete:
  `better_robot.lie.se3` / `lie.so3` / `lie.tangents.hat_so3` (BHF + BVR,
  many files), `better_robot.data_model.model.Model` (BVR),
  `better_robot.kinematics.forward_kinematics` (BVR). None import
  `backends` — verified — so the delete is consumer-safe, but re-run the
  import checks in T1.8's done-when after this lands.
- The `AUTO`/analytic Jacobian FD fallback and residuals still consume
  `Model`/`Data`; re-sign the *hot passes* (FK, dynamics), not the whole
  library. The full Problem/provider redesign is **M2a**.
- The frozen residual-kernel ABI (raw `r` + canonical `(E,dim,nv)` J) is
  **not built here** — it is M6. But the ABI you freeze in step 3 is what
  that later kernel J-layout plugs into; keep the flat-`E` convention
  consistent with the residual ABI sketch in 03 §2.2.

---

### T1.2 — Bridge prototype before bridge promises  [M]

**Goal / done-when (verbatim):** the prototype passes gradcheck +
gradgradcheck (via the torch-lane recompute path) + compile/fake-tensor
tests, and is written up as the normative pattern. **A CUDA runner is
secured (remote is fine) — or this item stays open.**

**Current state:** there is no bridge — `backends/warp/bridge.py` was a
21-line stub that raises, and it is deleted in T1.1. No
`torch.autograd.Function` or `torch.library.custom_op` exists anywhere in
`src/` (grep). The first draft's "~40-line helper with caller-owned
preallocated buffers" was **refuted** by codex round 2: writing into a
caller-provided output buffer makes the op non-functional, and torch
**rejects** `register_autograd` for non-functional custom ops
(reproduced against `torch/_library/custom_ops.py:597`:
`Cannot register autograd formula for non-functional operator …`). Warp's
own documented pattern is a **pair of functional custom ops** — one
forward, one backward, each allocating its own outputs, each with a fake
(shape-only) registration.

**Implementation plan:**
1. Build **one FK-shaped functional custom-op pair** (not a reusable
   40-line helper — schemas/shape-fns are per-pass). Forward op:
   `wp.from_torch(..., requires_grad=False)` on inputs, launch the FK
   kernel on **torch's current stream** via `wp.stream_from_torch`,
   allocate and return fresh output tensors. Backward op: a second
   functional custom op with its own fake registration. Register both
   with `torch.library.custom_op` + `register_fake` + `register_autograd`
   (torch ≥ 2.4 — the compile-safe route).
2. Prove **fp32 and fp64** (dispatch `wp.transformf` vs `wp.transformd`).
3. Prove **gradients to q AND to shared model values** with correct
   cross-batch reduction (the ABI's in-kernel reduction from T1.1).
4. Prove **`torch.compile(fullgraph=True)`** traces the op via the fake
   registration without graph-breaking.
5. **Second order:** the registered backward detects grad-enabled backward
   (`torch.is_grad_enabled()` inside the backward) and recomputes the VJP
   **differentiably via the torch lane** from saved inputs — warp has no
   adjoint-of-adjoint (codex #5). `gradgradcheck` runs **through the
   public API**, not only against the torch lane directly.
6. **Launch-on-current-stream + graph replay** — replay is CUDA-only, so
   it runs only on the secured runner.
7. Settle the two design choices the prototype exists to settle
   (03 §2.3): (a) functional ops allocating from torch's graph-pool-aware
   allocator vs a functional outer op wrapping private mutating launch
   ops; (b) `torch.library.custom_op` vs plain `autograd.Function`
   (cuRobo's choice) where compile-safety is not needed. **Record the
   decision with the evidence** — this write-up becomes the normative
   pattern doc consumed by `m6_warp_fast_path_and_cuda_graphs.md`.
8. Bump the torch floor: `pyproject.toml:12` `torch>=2.1.0` → `torch>=2.4`.
9. Bake in the interop hygiene rules from day one (audit_warp_platform
   §11.6): always `requires_grad=False` at `wp.from_torch` with
   torch-side grad buffers (the deferred-`.grad` sync is a measured 4.3×
   slowdown); layout preconditions checked up front with torch-lane
   fallback (a hard error only *inside* an active capture — never a silent
   `.contiguous()`); `tape.zero()` per call if a tape is used; hold warp
   array refs on `ctx`.

**What to test:**
- `tests/warp/test_bridge_prototype.py` (new, `@pytest.mark.skipif` warp
  not importable, run in the dedicated warp-extra CI job on warp-CPU):
  gradcheck (q and shared values, fp32 and fp64), gradgradcheck through
  the public API, `torch.compile(fullgraph=True)` + fake-tensor tracing,
  parity vs the torch lane.
- On the **remote CUDA runner only**: current-stream launch ordering and
  CUDA-graph replay parity.

**Pitfalls / do-not-forget:**
- **This item STAYS OPEN if no CUDA runner is secured.** Say so
  explicitly in your report. Warp-CPU cannot exercise streams, capture,
  graph replay, GPU atomics, or races (codex #8) — the bridge's most
  load-bearing rules. Securing a remote CUDA runner is an **M1 action
  item**, not an M6 luxury (README rule 7).
- Do NOT promise "one helper written once" — schemas, shape functions,
  saved context, and backward returns are pass-specific.
- Grad double-accumulation: pick ONE gradient channel (warp-owned grads
  returned by backward, OR aliased `t.grad`) and zero warp-side grads
  every call. Mixing double-counts (audit_warp_platform §9.3).

---

### T1.3 — First Warp kernel: the fused FK sweep  [M–L]

**Goal / done-when (verbatim):** parity + gradcheck versus the torch lane
pass, including branched trees and >16-joint chains, via warp-CPU in CI.
**Prototype status until CUDA validation** (in M1 if the runner exists,
else M6). The GPU benchmark and the default-on decision happen in M6.
Ships opt-in.

**Current state:** FK is a serial Python loop over `topo_order`
(`forward.py:73-135`), ~2 composes + 1 `joint_transform` per joint,
carrying ~4 ms fixed per-call overhead (B=1 costs the same as B=256). No
Warp kernel exists.

**Implementation plan:**
1. **Thread mapping — Newton-style, not mujoco_warp:** one thread per
   batch element, a **serial topological joint loop inside the kernel**,
   joint-kind dispatch as `if kind ==` chains over int8 codes (from
   `ModelStructure`). This fits BR's workload (few robots, huge batches);
   mujoco_warp's branch-parallel recompute does not (audit_warp_platform
   §7.3). Kernel lives in a real `.py` file (`kinematics/_warp_kernels.py`
   or similar — warp parses source files, no generated strings).
2. **Choose the adjoint strategy from the 03 §2.3 table and RECORD it**
   in the design table (the kernel-author checklist in 03 §9). Options:
   (a) warp's generated adjoint with **named stored intermediates** (warp
   does not replay dynamic-loop locals — a documented silent-wrong-grad
   trap); (b) a hand-written VJP kernel (a reverse tree sweep); (c) static
   specialization (unroll bounds at codegen — replay-safe). FK evidence
   does **not** generalize to RNEA/ABA/CRBA — each re-decides in M6.
3. **Differentiable-input matrix = q, joint placements, frame placements**
   (the fused pass consumes all three — this kernel is "FK + frame
   placements"). Guarantee first-order gradients to each through both
   lanes; an FK pass owes no inertia gradient (the "every leaf" slogan is
   retired — codex #6).
4. **Ships opt-in:** the lane is selected by a plain `if` at the FK call
   site keyed on device/dtype/layout/warp-availability; unsupported
   layouts **fall back to the torch lane, never raise** (03 §2.1). Turning
   it on must not change public behavior. No registry, no Protocol.

**Warp gradient traps to apply in review** (audit_warp_platform §2.2):
dynamic loops are not replayed in backward (produced `[32,8,2]` instead of
`[4,4,4]` in warp's own example); in-place `*=`/`/=` are silently wrong
(use `+=`/`-=`); local vec/mat/quat **component re-assignment** invalidates
grads (one write per component); data-dependent `atomic_add` pairs the
wrong threads. And the ABI rule: `wp.from_torch(..., requires_grad=False)`
always, grads managed torch-side. Warp has **no SO3/SE3 log/exp builtins**
— FK uses `quat_from_axis_angle` / `transform_multiply` (which exist), so
the Lie-log/exp port is not needed for *this* kernel (it is an M6 task for
pose residuals and retraction).

**What to test:**
- `tests/warp/test_fk_kernel_parity.py` (new, warp-CPU CI job): parity vs
  `forward_kinematics_raw` (torch lane) — fp32/fp64, several batch shapes,
  both base types; and **branched trees + chains >16 joints** (past warp's
  default `max_unroll=16`, where a dynamic-loop adjoint would silently
  break). Use the Panda (branched: two fingers) and a synthetic >16-joint
  serial chain and the SMPL-like body.
- `tests/warp/test_fk_kernel_gradcheck.py` (new): gradcheck vs the torch
  lane for q, joint placements, and frame placements, at singular points
  (θ=0, needs the M0 fix), on branched and long chains, plus the
  shared-value gradient-reduction case.
- The torch lane stays the oracle — its parity suite stays green.

**Pitfalls / do-not-forget:**
- **Prototype status only.** Warp-CPU parity marks it a prototype; it
  exercises none of streams/capture/atomics/races. Production requires
  CUDA validation on a real GPU runner (M1 if secured, else M6);
  default-on additionally requires the M6 benchmark. Never claim GPU
  validation from this dev box (CUDA is broken here).
- The GPU benchmark and default-on switch are **M6**, not here. Do not
  wire the kernel on by default.
- Cross-reference: this kernel gets its CUDA validation, GPU benchmark,
  and default-on decision in `m6_warp_fast_path_and_cuda_graphs.md` §2.

---

### T1.4 — Contract/docs migration for the inversion  [S–M]

**Goal / done-when (verbatim):** no doc or contract test contradicts the
two-lane design; a grep for `backend=` in docs/ comes back clean.

**Current state (verified stale claims):**
- `CLAUDE.md:41` — the dependency rule marked **"never violate"** still
  leads with `backends → lie → …`, the exact node M1 deletes.
- `docs/concepts/batching_and_backends.md:29-30` calls **explicit
  `Backend` objects passed via `backend=` kwargs** the "architectural
  core"; `:23` says every math layer is a `Backend`; `:168-245` is a whole
  "Backend Protocol" section; `:235-236` show `se3.compose(a, b,
  backend=wb)` and `forward_kinematics(model, q, backend=wb)`.
- `docs/conventions/performance.md:293` lists `backends/` as a layer with
  "Explicit `backend=` kwargs"; `:140-189` is a "CUDA graph capture"
  section whose `:154-155` claims **"replay nukes the grad tape"** and
  capture is opt-in because it interacts with autograd — but the M2/M6
  design deliberately **captures the backward** (03 §2.6). It also
  references the deleted `@graph_capture` at `:143,146`.
- `plan/01_assessment.md` has stale lines the codex sweep flagged (#14):
  `:157` says the `*_raw` functions are "already the needed tensor-only
  seam" (03 §2.7 corrects: they still consume Model/JointModel and mutate
  Data); `:158`/around says mujoco_warp/Newton wrap kernels in
  `torch.autograd.Function` (Newton has no torch bridge in core);
  GPU-profiling references point at M5 where it is now M6.

**Implementation plan:**
1. `CLAUDE.md:41` — rewrite the dependency rule to the target DAG
   (`lie → spatial → data_model → (kinematics, dynamics) → residuals →
   costs → optim → tasks → viewer`; `io → data_model`;
   `collision ∥ kinematics`), and describe the two-lane model (Warp is not
   a layer; kernels live beside their torch counterparts). Remove the
   `backends →` prefix. (Do not merge `costs` into `optim` in the text —
   that is M2c.)
2. Rewrite `docs/concepts/batching_and_backends.md` — retire the Backend
   Protocol / `backend=` framing; replace with the ModelStructure/
   ModelValues seam and the two-lane `if`-at-call-site selection. Apply
   the `sphinx-docs` and `diataxis-docs` skills (this is a Concepts page —
   keep it explanatory, not how-to).
3. `docs/conventions/performance.md` — remove the `backends/` row and the
   `backend=` claim (`:293`); fix the capture section (`:140-189`) to say
   capture records **forward + backward together** (cuRobo/Newton pattern,
   03 §2.6), and remove the `@graph_capture` references. Correct
   `:154-155`.
4. Fix the stale `plan/01_assessment.md` lines noted above (these are plan
   docs, but the roadmap item names them explicitly).
5. Retire the layer-DAG test's `backends` node — already done in T1.1;
   verify here it is gone.
6. State the **torch-version policy** in `pyproject` (the `torch>=2.4`
   floor from T1.2) and note it in the docs.

**What to test:**
- `grep -rn "backend=" docs/ --include='*.md'` on the **source** tree
  (exclude `docs/_build/`) returns clean; then **regenerate the HTML**
  (`make -C docs html` or the repo's documented build) so `docs/_build/`
  stops carrying the stale `backend=` strings.
- Every doc snippet executes in CI (the executable-docs job from T1.10).
- No contract test references `backends`.

**Pitfalls / do-not-forget:**
- The `backend=` grep hits `docs/_build/html/...` (generated). Do not
  hand-edit generated HTML — fix the `.md` sources and rebuild.
- `CLAUDE.md` and per-package `CLAUDE.md` files have a verified drift
  history (6/10 sampled doc claims wrong). If your other tasks invalidate
  a `CLAUDE.md` claim, fix it in the same change (standing rule 2).

---

### T1.5 — Cross-cutting engineering contract + LICENSE  [M]

**Goal / done-when (verbatim):** short written policies are checked in; a
LICENSE file exists.

**Current state:** no top-level `LICENSE`. Policies exist only implicitly
and inconsistently (e.g. an fp16-rejection exception class that claims but
does not enforce; `Model.meta` retains builder IR + resolver objects).

**Implementation plan:** write short checked-in policies (03 §9), each
before the API it constrains freezes:
1. **Dtype & numerics:** fp32 primary, fp64 supported, fp16/bf16
   explicitly rejected (warp has no bf16); accumulation/factorization
   dtypes; dtype-dependent tolerances + Taylor cutoffs; TF32 stance;
   "preserve input dtype" as a tested invariant.
2. **Quaternion double cover & continuity:** how `q ~ −q` is handled in
   priors, temporal residuals, interpolation, and better_human pose
   params (hemisphere alignment or sign-invariant metrics); the convention
   near the log-map discontinuity at θ = π.
3. **Threading & reentrancy:** concurrent solves must not share mutable
   state (evaluation-local caching, deeply-immutable `Model`); CUDA-stream
   and multiprocessing-spawn behavior.
4. **Serialization:** a versioned structure/values `state_dict`,
   `map_location`, an explicit pickle stance; `Model.meta` must not leak
   builder IR / resolver objects into a checkpoint.
5. **Differentiation contract:** which gradients are guaranteed (q, model
   values, residual params, solver hyperparameters), to what order,
   through which paths (unrolled vs implicit), and behavior on
   non-converged solves. Feed the differentiable-input matrix from T1.3.
6. **Compile lifecycle:** which dims are dynamic; graph-cache keying;
   cold-start expectations (~31 s first FK compile); phase-change = zero
   weights vs separate compiled programs; the warp-side questions
   (kernel-cache location/versioning in CI, first-call codegen latency,
   CUDA-graph re-record triggers).
7. **LICENSE — present options, then STOP for the owner.** Draft the
   choice with its consequences and **do not pick silently**:
   - **Apache-2.0** — carries NOTICE-file obligations; required-compatible
     for porting **jaxopt / mujoco_warp / newton** code (all Apache-2.0).
   - **MIT** — simplest; **pyroki is MIT**.
   - Note that porting Apache-2.0 code under an MIT project needs care
     (keep the Apache NOTICE and attribution). Keep a source ledger
     distinguishing algorithm *reimplementation* from *copied* code.
   **Produce the options write-up and STOP for the owner decision** before
   adding the `LICENSE` file (README rule 8; evidence/decision-gated).

**What to test:** the policies are prose (checked into `docs/conventions/`
or a `CONTRACT.md`), not code — but the fp16/bf16 rejection and the
"preserve input dtype" invariant get real tests (the latter overlaps M0's
dtype-preservation fix). Verify `LICENSE` exists only after the owner
picks.

**Pitfalls / do-not-forget:** this task is largely parallel and blocks no
code — but the LICENSE decision **gates any code porting** (jaxopt LM
numerics land in M2b; pyroki vectorization in M3). Do not port before the
license is chosen. Threading/serialization decisions constrain M2a's
provider-cache design — write them before M2a freezes.

---

### T1.6 — Hoist per-call constants  [M]

**Goal / done-when (verbatim):** zero tensor-constructor/`.to()` calls
inside FK/RNEA loops, enforced by lint.

**Current state (verified):**
- Pose residual: `weight = r.new_tensor([self.pos_weight]*3 +
  [self.ori_weight]*3)` per call at `residuals/pose.py:65` (in `__call__`)
  and `:102` (in `.jacobian()`) — an allocation each call. **Note for
  M2/M6:** these two lines PRE-WEIGHT both `r` and `J`, whereas the frozen
  residual-kernel ABI (03 §2.2) expects the kernel to return the **RAW**
  residual with robust weighting on the torch side. Do **not** resolve
  that tension here — flag it as a cross-reference to
  `m6_warp_fast_path_and_cuda_graphs.md` (and the M2 residual redesign).
  For M1, just hoist the constant tensor construction out of the per-call
  path (precompute the weight vector once at residual construction).
- `torch.eye` per iteration: `optim/optimizers/gauss_newton.py:53`
  (`H = JtJ + self.eps * torch.eye(nv, …)`) and
  `optim/optimizers/levenberg_marquardt.py:96`
  (`H = JtJ + state.damping * torch.eye(nv, …)`). Preallocate the identity
  once outside the loop.
- Per-joint axis `.to()` calls: `revolute.py:21` (transform), `:29-31`
  (subspace, three scalar writes), `:39` (velocity); `prismatic.py:19,26,37`;
  `helical.py:32,48` — and helical *also* has `float(self.pitch)` at
  `:36,:50-52` (a Python-scalar sync, relevant to T1.7's lint). These
  module-level CPU axis tensors get `.to(device, dtype)` per joint per FK/
  Jacobian/dynamics call — a real H2D copy on CUDA. Pack the axes (and kind
  codes) into flat device tensors on `ModelStructure` (T1.1) so the loop
  indexes an already-on-device array.
- 6×6 spatial inertias: `spatial/inertia.py:179` builds `torch.eye` inside
  `_to_6x6`, re-derived per joint per call in `rnea.py:167`, `aba.py:121`,
  `crba.py:60`, `centroidal.py:152`. Precompute a `(njoints,6,6)` buffer on
  `ModelStructure`/`ModelValues` **for static inertias only** — when
  `body_inertias` is a live parametric tensor, derive the 6×6 once per
  evaluation context so gradients stay attached (03 §5).

**Implementation plan:** hoist each constant to build time / loop entry;
route joint axes and kind codes through the flat `ModelStructure` device
tensors from T1.1 (this work doubles as the kernel-layout prep for T1.3).
Precompute the constant 6×6 inertia buffer for static values; keep the
per-evaluation derivation path for parametric values.

**What to test:**
- The extended hot-path lint (T1.7) is the enforcement — it must show zero
  `new_tensor`/`torch.eye`/`.to()`-in-loop hits across FK/RNEA after this
  lands.
- Parity suite + optimizer convergence tests stay green (hoisting must not
  change numbers).
- A micro-check that the precomputed 6×6 inertia buffer matches the
  per-call `_to_6x6` output bitwise for static inertias, and that
  parametric inertias still flow gradients.

**Pitfalls / do-not-forget:** the pose-residual pre-weighting note above —
do not "fix" the ABI mismatch here. Parametric inertia gradients must not
be broken by caching (the cache is for *static* values only).

---

### T1.7 — Extend the hot-path lint  [S]

**Goal / done-when (verbatim):** the lint catches the fixed patterns when
they are re-introduced.

**Current state (verified):** `tests/contract/test_hot_path_lint.py`
AST-walks `WATCHED = ("kinematics", "dynamics", "optim/optimizers")` and
bans `.item()`, `.cpu()`, `torch.{zeros,ones,empty,full,rand,randn}`
*inside a Python loop*, and `if x.dim() == N`. A line can exempt itself
with `# bench-ok: <reason>`.

**Implementation plan:** extend the lint to also flag, on tensor
operands: `bool(` / `float(` calls (forced host syncs — the free-flyer
FK breaker and the optimizer per-iteration syncs), `new_tensor` (the pose
residual allocation), and `torch.eye` (the optimizer/inertia allocations).
Widen `WATCHED` to add `residuals` and `lie`. Keep the `# bench-ok`
exemption mechanism.

**What to test:** add a **negative test** that re-introducing each newly
banned pattern (a `float(x)`, a `new_tensor`, a `torch.eye`, a `bool(x)`)
in a watched file is *caught* by the lint — i.e. assert the lint fails on
a crafted snippet. This is the done-when: the lint must catch reintroduction.

**Pitfalls / do-not-forget:** land this **after** T1.6 — turning on the
widened lint before the constants are hoisted fails CI on the existing
violations. `float(`/`bool(` appear legitimately on Python scalars (e.g.
`float(self.pitch)` in helical); the lint must target tensor operands or
the watched hot-loop bodies, and cold-path uses take `# bench-ok`. The M0
free-flyer `bool()` fix (opt-in debug check) should already be in place;
this lint prevents its reintroduction.

---

### T1.8 — De-bloat zero-consumer surfaces ONLY  [S]

**Goal / done-when (verbatim):** `wc -l` drops by ≥1k and both consumer
repos still import cleanly.

**Current state (verified):** delete only surfaces with **zero** consumer
references anywhere in the workspace. `costs/` and the deprecated `Data`
aliases stay as **shims** (BHF imports `costs.stack.CostStack` and uses
`Data.oMi` today).

Verified deletion candidates (all confirmed zero-consumer or safely
retired):
- `utils/` — `batching.py`, `broadcasting.py`, `logging.py`, `testing.py`
  (+`__init__.py`); grep confirms **zero imports** from the rest of
  `src/`. Delete. (Also drop the `utils` node from `LAYER_RANK` if T1.1
  didn't.)
- The residual **registry** (`get_residual`, zero callers) — delete the
  registry only. **Do NOT delete `residuals/` or `residuals.base`** — BHF
  imports `better_robot.residuals` and `residuals.base.ResidualState`
  (`optimize_motion.py:211,216`).
- `kinematics/chain.py` (`get_chain` → `NotImplementedError`, 461 bytes)
  and `data_model/indexing.py` (`build_name_to_id` → `NotImplementedError`,
  562 bytes) — both stubs; verify truly zero-consumer
  (`grep -rn 'get_chain\|build_name_to_id\|kinematics.chain\|data_model.indexing'
  src/ ../BetterHumanForce ../BetterVideoReconstruction`) before deleting.
  (Note: `chain.py` is at `kinematics/`, not `data_model/` as some audit
  text implies.)
- `graph_capture` — the no-op function at `backends/__init__.py:110` (it
  leaves with `backends/` in T1.1; verify no other reference survives,
  including in `docs/conventions/performance.md` — see T1.4).
- The **IR schema handshake** — `IR_SCHEMA_VERSION` constant at
  `io/ir.py:75`, the `schema_version` field at `:93`, and the reject check
  at `build_model.py:232-235`. Nothing pickles IRs. Remove the constant,
  field, and check. *(Plan doc says `io/ir.py:72`/`:93`; code now shows the
  constant at `:75` and the field at `:93`.)*
- `rich` dependency — `pyproject.toml:14` `rich>=13`, never imported
  (`grep -rn 'import rich\|from rich' src/` → none). Remove from core deps.
- `dynamics/action/` — `action.py` (55) + `differential.py` (104) +
  `integrated.py` (112) + `__init__.py` (21) = **292 LOC**. **Park in a
  branch** (`git branch experimental/dynamics-action` then remove from
  `main`), not delete — a DDP/iLQR solver may revive it. Fix the false
  "implemented" signal in `CLAUDE.md`'s feature list.
- The `retarget` stub — `tasks/retarget.py:41` raises
  `NotImplementedError("see docs/concepts/tasks.md §4")`; it is frozen
  into the 26-symbol public API (`__init__.py:80`, `EXPECTED` in
  `test_public_api.py`). Remove `retarget` from `__all__` and from
  `EXPECTED`; **unfreeze the 26-symbol API contract** (relax the strict
  `actual == EXPECTED` / `len == 26` assertions in
  `tests/contract/test_public_api.py` to a subset/expectation that stays
  unfrozen until 1.0, per 03 §8). Keep `SE3` and `ModelBuilder` exports.

**Shims to keep (NOT delete):**
- `costs/` + `costs.stack.CostStack` (BHF `optimize_motion.py:210`) — thin
  shim until the M2c `costs/`→`optim/` migration.
- The deprecated `Data.oMi` (and the other `Data` aliases) — BHF
  `tools/robot_motion/playback.py`, `motion.py` — shim until M2/M4.

**What to test:**
- `wc -l` before/after over `src/` drops by **≥ 1000** lines.
- Both consumer repos still import cleanly — exact commands:
  ```
  cd /data3/rikhat.akizhanov/better/BetterHumanForce && \
    uv run python -c "import better_robot as br; \
      from better_robot.io.builders import JOINT_NAMES, build_kinematic_tree_model, make_smpl_like_model; \
      from better_robot.costs.stack import CostStack; \
      from better_robot.residuals.base import ResidualState; \
      from better_robot.optim.optimizers.gauss_newton import GaussNewton; \
      from better_robot.optim.kernels.huber import Huber; \
      from better_robot.optim.kernels.cauchy import Cauchy; \
      from better_robot.tasks.trajectory import Trajectory; \
      from better_robot.lie import se3, so3; from better_robot.lie.tangents import hat_so3; \
      print('BHF imports OK')"
  cd /data3/rikhat.akizhanov/better/BetterVideoReconstruction && \
    uv run python -c "import better_robot as br; \
      from better_robot.lie import se3, so3; \
      from better_robot.data_model.model import Model; \
      from better_robot.io.builders import build_kinematic_tree_model; \
      from better_robot.kinematics import forward_kinematics; \
      print('BVR imports OK')"
  ```
- The full BR suite stays green (minus the retired frozen-API assertions).

**Pitfalls / do-not-forget:**
- Deletions land **with or after** their replacement (standing rule 1);
  everything here is genuinely zero-consumer or a stub, so it deletes now
  — but the grep-both-consumers check is mandatory before each delete.
- The `no_legacy_strings` contract test and the `Data` alias shims are
  retired in **M2/M4**, not here (03 §8 lists them but they still guard
  live consumers) — leave them.

---

### T1.9 — Fix the ModelBuilder kind mapping  [S–M]

**Goal / done-when (verbatim):** every builder method produces a loadable
model.

**Current state (verified in audit; re-verify the `IRError`s):**
- `ModelBuilder` is the class at `io/parsers/programmatic.py:67`. (The
  `io/builders/` *package* — `build_kinematic_tree_model`, `JOINT_NAMES`,
  `make_smpl_like_model` — is a **different, working** surface that BHF/BVR
  import; do not confuse the two. Reconcile the naming in your write-up so
  the executor doesn't touch `io/builders/`.)
- `ModelBuilder.add_helical(...)` emits `kind="helical"`, which
  `build_model._kind_to_joint_model` rejects → `IRError: Unknown joint
  kind 'helical'` (`JointHelical` is imported at `build_model.py:15` but
  has no case).
- `add_joint(kind=<JointModel instance>)` pushes the instance's `.kind`
  string (e.g. `"revolute_rx"`), also rejected → `IRError: Unknown joint
  kind 'revolute_rx'`. So the documented "pass a JointModel" contract
  fails for every axis-aligned revolute/prismatic class, helical,
  composite, and mimic.
- `JointComposite` and `JointHelical` are constructible but unreachable
  from any parser/builder.

**Implementation plan:**
1. Verify today's failures first:
   `uv run python -c "from better_robot.io.parsers.programmatic import ModelBuilder; …"`
   reproducing `add_helical` and `add_joint(kind=JointRX())` raising
   `IRError` — this is your red test.
2. Map the missing kinds in `build_model._kind_to_joint_model`: add a
   `helical` case (carrying axis + pitch through the IR), and make
   `add_joint(kind=<instance>)` carry the instance (or its axis-aligned
   kind string) through the IR so every axis-aligned revolute/prismatic,
   helical, composite, and mimic-base class round-trips to a loadable
   model. (Mimic joints still **reject at build** per M0 — do not
   silently accept them.)
3. **Wire or remove `JointComposite`** — decide: either give it a builder
   path + `build_model` case, or delete it as dead API. State the choice.

**What to test:**
- `tests/io/test_builder_kinds.py` (new): every `ModelBuilder` method
  (`add_revolute`/`add_prismatic`/`add_helical`/`add_fixed`/`add_free_flyer`/
  `add_spherical`/`add_joint(kind=<each JointModel class>)`, …) builds a
  loadable model and FK runs on it. Parametrize over kinds.
- If `JointComposite` is wired: a build+FK test; if removed: assert it is
  gone from the public surface.

**Pitfalls / do-not-forget:** `kinematic_tree.py:171,189` calls the
private `b._push_joint(...)` directly, bypassing the guard-railed API — if
you change the builder's kind handling, keep that internal path working
(or route it through the fixed public method). Do not break the
`io/builders/` package that BHF/BVR depend on.

---

### T1.10 — Extras packaging + CI  [S–M]

**Goal / done-when (verbatim):** a fresh install pulls only
torch+numpy+yourdfpy; CI is green.

**Current state (verified):** `pyproject.toml` core `dependencies`
(`:11-19`) currently include, as **hard deps**, `torch>=2.1.0` (→ 2.4 per
T1.2), `numpy>=2.0`, `rich>=13` (removed in T1.8), `yourdfpy>=0.0.14`,
`mujoco>=3.1` (`:16`), `trimesh>=4.0.0` (`:17`), `viser>=0.2.0` (`:18`),
`robot_descriptions>=1.0.0` (`:19`). There is **no `.github/`** — no CI
exists.

**Implementation plan:**
1. Move the heavy deps into extras (03 §10): `viewer` = viser; `io-mjcf` =
   mujoco; `meshes` = trimesh; `demos` = robot_descriptions; **`warp`** =
   warp-lang (the GPU fast lane). Core `dependencies` becomes exactly
   `torch>=2.4`, `numpy>=2.0`, `yourdfpy>=0.0.14`. The core stays fully
   functional and importable without any extra (03 §2.5); `warp` is
   promoted to a hard dep only if/when kernels become default-on for CUDA
   (not in M1).
2. Guard the now-optional imports in `src/` so a core-only install imports
   cleanly (viewer/mjcf/mesh/demo code raises a clear "install the X
   extra" message on use, not on import). Check
   `tests/contract/test_optional_imports.py` — extend it to assert the new
   optional boundaries.
3. Set up CI (`.github/workflows/`): a job running `uv run pytest tests/`,
   a job running the **executable docs** (every snippet — see T1.4 and
   03 §8), and a job running the **committed benchmark definitions** (the
   eager-CPU non-regression from T1.1 + any others). Add a **dedicated
   warp-extra job** on warp-CPU with a persistent kernel cache and small
   bounded parity cases (codex #12) — this is where T1.2/T1.3's warp tests
   run.

**What to test:**
- Fresh-install check (committed as a CI step):
  `uv pip install .` in a clean env, then
  `python -c "import better_robot; ..."`, and assert
  `mujoco`/`trimesh`/`viser`/`robot_descriptions`/`warp` are **not**
  importable (a `pip list` / import-guard test) — only torch+numpy+yourdfpy
  pulled.
- `tests/contract/test_optional_imports.py` green with the new extras.
- CI is green end-to-end.

**Pitfalls / do-not-forget:** lands near the **end** of M1 — CI green
requires the tree stable, and `pyproject` is edited by T1.2 (torch floor)
and T1.5 (license classifier) too; make those edits cohere. Do not add
`warp` as a hard core dep. The warp-CPU CI job must not make the default
`pytest tests/` job depend on warp being installed (the core suite runs
without it).

---

### T1.11 — Cache `Data` per iterate in the current solver  [S]

**Goal / done-when (verbatim):** `solve_ik` does at most one FK per
iteration plus one per accepted step.

**Current state (verified mechanism):** `LeastSquaresProblem.residual(x)`,
`.jacobian(x)`, `.gradient(x)`, and `.jacobian_blocks(x)` each call
`self.state_factory(x)` (`optim/problem.py:42,54,84,102`), and `solve_ik`'s
`_state_factory` (`tasks/ik.py:212-213`) runs a **fresh**
`forward_kinematics(model, x, compute_frames=True)` — a new `Data` — every
call. In the LM loop (`levenberg_marquardt.py`), each iteration calls
`problem.jacobian(state.x)` (1 FK) and, after `problem.step`, evaluates the
trial residual `problem.residual(x_new)` (1 FK); plus `SolverState.from_problem`
does 1 residual eval at `x0` (1 FK). Over 5 iterations that is ~1 + 5·2 =
**~11 FK calls** — verified against the audit's "11 FK in a 5-iteration
solve". Nothing is cached between residual and Jacobian at the *same* `x`.

**Implementation plan:** add an **evaluation-local** memo so that
`residual(x)` and `jacobian(x)` (and `gradient`/`jacobian_blocks`) at the
same iterate share one `Data`. Simplest stopgap: cache the last
`(x, state)` pair in the `LeastSquaresProblem`, reusing the cached `state`
when the *same* tensor object (identity, or an equal `data_ptr`+version)
is passed. Because the LM loop reuses the accepted `x_new` as the next
iterate's `state.x`, this yields **≤ 1 FK per iteration + 1 per accepted
step**. Keep it evaluation-local — no persistent version-keyed cache
across the whole solve (03 §3 constraint 2: cached autograd graphs leak
memory / break second backward). This is explicitly a **stopgap** until
the M2a `RobotStateProvider` DAG replaces it — leave a `TODO(M2a)`.

**What to test:**
- `tests/tasks/test_solve_ik_fk_count.py` (new): monkeypatch / spy on
  `forward_kinematics` and assert the call count over a fixed-iteration
  `solve_ik` is ≤ `n_iterations + n_accepted_steps + 1` (down from ~11 for
  a 5-iteration solve). Assert the solution matches the pre-cache result
  within tolerance (caching must not change numbers).
- Existing IK convergence + parity tests stay green.

**Pitfalls / do-not-forget:** the cache MUST be evaluation-local. Do not
key it on a mutable "values version" (tensors mutate invisibly — the very
bug `Data` documents). Do not extend the cache lifetime across accepted
steps except via the detached accepted-state artifact. This is a
throwaway that the M2a provider DAG deletes — do not over-engineer it.

---

## Milestone acceptance checklist

- [ ] `src/better_robot/backends/` is gone (all 527 LOC, incl. the
      `backends/warp/` stub); no `backend=` kwarg, `set_backend`,
      `default_backend`, or `graph_capture` survives in `src/`.
- [ ] `lie/_torch_native_backend.py` renamed to `lie/_impl.py`; facades
      call it directly.
- [ ] `ModelStructure` (dual representation, consistency-tested) and
      `ModelValues` (tensor pytree, frames as a table) exist; the hot
      passes are re-signed as pure functions; `rnea` no longer mutates
      `Data` in the raw path.
- [ ] The execution-batch ABI is frozen; the four ABI tests pass;
      shared-value gradients reduce in-kernel.
- [ ] The layer-DAG test has no `backends` node and is green; the
      **pinocchio-parity suite is green**.
- [ ] The `transformf`/`transformd` aliasing layout test pins pointer +
      stride.
- [ ] The eager-CPU non-regression benchmark is committed; the numbers
      were reviewed by the owner (no silent regression).
- [ ] The bridge prototype passes gradcheck + gradgradcheck (torch-lane
      recompute) + compile/fake-tensor tests and is written up as the
      normative pattern. **CUDA runner secured, OR T1.2 is reported STILL
      OPEN.**
- [ ] The FK Warp kernel passes warp-CPU parity + gradcheck (branched
      trees, >16-joint chains, singular points); its adjoint strategy is
      recorded in the design table; it ships opt-in and is marked
      **prototype** (no CUDA validation on this box).
- [ ] `torch>=2.4` floor in `pyproject`; docs/`CLAUDE.md`/plan lines no
      longer contradict the two-lane design; `grep backend=` in docs
      sources is clean and HTML rebuilt.
- [ ] Engineering-contract policies checked in; **LICENSE decision made by
      the owner** and `LICENSE` file present.
- [ ] Zero tensor-constructor/`.to()` in FK/RNEA loops; the widened
      hot-path lint (bool/float/new_tensor/torch.eye across
      kinematics/dynamics/optim/residuals/lie) catches reintroduction.
- [ ] `wc -l` over `src/` dropped ≥ 1000; both consumer import-checks pass;
      `dynamics/action/` parked in a branch; `retarget` removed from the
      API; the 26-symbol contract unfrozen.
- [ ] Every `ModelBuilder` method produces a loadable model
      (helical + JointModel-instance kinds); `JointComposite` wired or
      removed (state which).
- [ ] Extras (`viewer`/`io-mjcf`/`meshes`/`demos`/`warp`) split out; a
      fresh install pulls only torch+numpy+yourdfpy; CI green (tests +
      executable docs + benchmarks + warp-CPU job).
- [ ] `solve_ik` does ≤ 1 FK per iteration + 1 per accepted step.

## Out of scope

- **Batched-value breadth across all passes** — that is **M3**
  (`m3_parametric_model_breadth.md`); M1 freezes the ABI and re-signs the
  passes but only builds ABI-level coverage, not full value-batched FK/
  RNEA/CRBA.
- **The public `Model.with_values(...)` API, the frame table breadth, the
  exhaustive `.to()`, the mimic reduced-coordinate map** — all **M3**.
  M1's mimic handling stays "reject at build" (M0).
- **CUDA validation, GPU benchmarks, kernel default-on decisions,
  CUDA-graph capture, the Warp Lie-log/exp `wp.func` library, RNEA/ABA/
  CRBA/residual/collision kernels** — all **M6**
  (`m6_warp_fast_path_and_cuda_graphs.md`). M1 ships exactly one opt-in
  prototype FK kernel.
- **The Problem/VarSpec/Values redesign, the provider DAG, tangent-space
  autograd everywhere** — **M2a**. T1.11's `Data` cache is a stopgap the
  M2a providers delete.
- **Batched second-order solvers, the capture-ready `update`, the real
  bounded LM** — **M2b**. Do not touch optimizer control flow here beyond
  hoisting constants (T1.6) and caching `Data` (T1.11).
- **Merging `costs/` into `optim/`, deleting the `costs/` shim, deleting
  the `Data` aliases and `no_legacy_strings` test** — **M2c/M4**. They
  guard live consumers; keep the shims.
- **The residual-kernel raw-`r`/pre-weighting reconciliation** — flagged
  in T1.6, resolved in **M2/M6**, not here.

## References

- `plan/04_roadmap.md` §M1 (items 1–11) — the authoritative done-when text
  reproduced verbatim above.
- `plan/03_architecture.md` §1 (target layer diagram), §2.1–2.7 (two-lane
  model, boundary table, bridge, layout/ABI, CPU policy, capture,
  anti-goals), §5 (ModelStructure/ModelValues), §9 (engineering contract),
  §10 (packaging/de-bloat).
- `plan/01_assessment.md` §2.1 (backends deletion evidence), §2.4
  (constants/hot-path syncs), §2.5 (dead code + consumer-import caveat).
- `plan/research/audit_core_architecture.md` §2.1 (call-chain, inverted
  imports), §2.6 (per-call H2D copies), §2.9 (ModelBuilder failures),
  §2.11 (legacy machinery), R1–R11 (recommendations + suggested order).
- `plan/research/audit_compute_pass_inventory.md` §2.2 (tree-scan kernel
  candidates), §4 (Model tensor layout: what is flat vs Python), §5
  (sync-point inventory).
- `plan/research/audit_warp_platform.md` §1 (torch interop rules), §2.2
  (gradient traps), §4 (native SE3/transform layout, no log/exp), §3
  (warp-CPU is single-threaded correctness-only), §7.3 (Newton-style
  per-batch-element mapping), §11 (implications).
- `plan/research/audit_curobo_warp_integration.md` §2 (autograd.Function
  boundary + stream discipline), §3 (CUDA-graph capture pattern), §4
  (CudaRobotModel flat-tensor layout), §10 (lessons: warp-first ≠ warp
  everywhere).
- `plan/research/codex_warp_review.md` — the binding corrections:
  #1 functional custom-op PAIR (T1.2), #2 dual representation (T1.1),
  #3 ABI frozen in M1 (T1.1), #4 explicit per-kernel adjoint strategy
  (T1.3), #5 second-order via torch-lane recompute (T1.2), #6
  differentiable-input matrix (T1.3), #8 remote CUDA runner is an M1
  action item (T1.2/T1.3), #12 eager-CPU baseline + warp-CI job
  (T1.1/T1.10), #13 pointer+stride aliasing test (T1.1), #14 the
  docs/contract sweep (T1.4).
- Sibling files: `m2a_variable_blocks_and_slice.md` (consumes the seam +
  θ=0 fix; deletes T1.11's stopgap), `m3_parametric_model_breadth.md`
  (extends the ABI's breadth), `m6_warp_fast_path_and_cuda_graphs.md`
  (CUDA-validates the bridge + FK kernel, resolves the residual raw-`r`
  ABI). The `backends/warp/` stub deleted in T1.1 is m6's tripwire that
  M1 landed.
- Consumer repos (read-only): `/data3/rikhat.akizhanov/better/BetterHumanForce`
  (live imports: `costs.stack.CostStack`, `residuals.base.ResidualState`,
  `optim.optimizers.gauss_newton.GaussNewton`, `optim.kernels.{huber,cauchy}`,
  `io.builders.*`, `tasks.trajectory.Trajectory`, `lie.*`, `Data.oMi`) and
  `/data3/rikhat.akizhanov/better/BetterVideoReconstruction` (`lie.se3/so3`,
  `data_model.model.Model`, `io.builders.build_kinematic_tree_model`,
  `kinematics.forward_kinematics`).
