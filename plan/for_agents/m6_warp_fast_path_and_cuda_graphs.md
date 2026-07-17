# M6 — Warp Fast Path & CUDA Graphs: Agent Execution Instructions

> **Implementation log (2026-07-17):** The CUDA gate failed (zero devices;
> driver unavailable), so all GPU/kernel/capture/benchmark tasks remain open.
> A conservative dense Torch-lane implicit `solve` subset landed; its remaining
> contract gaps are recorded in `m6_results.md`.

> Read `plan/for_agents/README.md` first. It carries the standing rules
> (deletion ordering, honesty rules, the five kernel requirements, test
> commands). Standing rules 4 (committed benchmark definitions), 6 (every
> warp kernel's five requirements), 7 (CUDA broken on the dev box), and 8
> (evidence-gated decisions stay with the owner) govern every task below.

## Mission

Make the GPU fast lane real. M1 built the two-lane seam, the bridge
pattern, and one prototype FK kernel; M2 built capture-ready batched
solvers. M6 is where those investments pay out on a real CUDA box:
committed GPU baselines, the warp kernel build-out in boundary-table
order (03 §2.2), CUDA-graph capture of solver inner loops (a third of
cuRobo's speed story), the implicit-diff `solve()` (the flagship
differentiability feature, torch lane), and honest external benchmarks
against cuRobo and a JAX-class library. The "should we do Warp" decision
is already made (03 §2). What stays evidence-driven — and stays with the
owner — is each kernel's **default-on** switch, per (device, dtype).

## Prerequisite gate: a real CUDA GPU box

**Nothing GPU-specific in this milestone can be validated on the current dev
box.** Reverified 2026-07-17 after the dependency refresh: Torch is
`2.13.0+cu126` (`torch.version.cuda == "12.6"`) but reports
`torch.cuda.is_available() == False` and zero devices; `nvidia-smi` cannot
communicate with an NVIDIA driver. Warp `1.15.0` is installed and, with an
isolated writable cache, reports toolkit 12.9 but no CUDA driver/device.
Warp-CPU (embedded Clang, single-threaded serial grid loop —
`audit_warp_platform.md §3.2`) compiles kernels and adjoints and is a
legitimate **correctness/parity vehicle for prototyping**, but it
exercises none of: streams, `wp.stream_from_torch` bridging, CUDA-graph
capture/replay, GPU atomics for shared-value gradient reduction,
block/grid races, or any performance property. Per standing rule 7 and
03 §2.5, warp-CPU results mark a kernel **prototype**, never production,
never a performance claim.

Securing a remote CUDA runner was an **M1 action item** (roadmap M1
item 2: "A CUDA runner is secured (remote is fine) — or this item stays
open"). Before starting anything here, verify the runner exists and
works. On the GPU box, all of the following must pass:

```bash
nvidia-smi                                # GPU model + driver visible; record both
uv run python -c "import torch; assert torch.cuda.is_available(); \
    print(torch.version.cuda, torch.cuda.get_device_name(0))"
uv run python -c "import warp as wp; wp.init()"   # prints a CUDA device
# smoke: one warp kernel launch on cuda + one torch CUDA-graph capture/replay
# of a trivial matmul (torch.cuda.CUDAGraph) — both must run cleanly.
```

If any of these fail, **stop and report to the owner** — do not attempt
to "partially validate" on CPU and do not soften the prototype label.
Record GPU model, driver, CUDA toolkit, torch version, warp version, and
kernel-cache location in the benchmark definition (T6.1); every
committed number carries this header.

## Prerequisites (milestone outputs this file relies on)

- **M1** (`m1_two_lane_seam_and_hygiene.md`): the ModelStructure /
  ModelValues seam with the dual representation; the **frozen
  execution-batch ABI** (flat `E`, per-input batch-index maps, in-kernel
  shared-value gradient reduction — 03 §2.4); the bridge prototype
  (functional custom-op pair with fake registrations, torch ≥ 2.4 floor)
  written up as the normative pattern; the prototype FK kernel with
  warp-CPU parity in CI; the committed eager-CPU non-regression
  benchmark; the layout test pinning `transformf`/`transformd` aliasing
  by pointer and stride. Confirm: the M1 FK kernel's tests pass under
  the warp extra; grep for the bridge write-up doc; the ABI tests
  (batched-q × unbatched-values, converse, multi-axis, mismatch) exist
  and pass.
- **M2a** (`m2a_variable_blocks_and_slice.md`): the
  differentiation-contract decisions that constrain `solve()` implicit
  diff (T6.11), and the custom-residual author guide (capture
  eligibility rules build on it).
- **M2b** (`m2b_batched_second_order_solvers.md`): batched LM/GN on
  `init_state/update/run` with the **capture-readiness checklist**
  satisfied (fixed input buffers via `copy_`, branch-free logic,
  `cholesky_ex` info-mask fallback as fixed tensor work, host syncs only
  at `run` boundaries). T6.9 is the certification of that checklist —
  it was accepted in M2b by inspection, it is **proven** here.
- **M3** (`m3_parametric_model_breadth.md`): value-batched breadth under
  the M1 ABI; the betas-parameterized SMPL-scale skeleton (used as a
  baseline robot in T6.1); the vectorized `integrate`/`difference`
  torch rewrite (the bar for T6.8); value-batched coverage tests for the
  M1 FK kernel.
- **M4** (`m4_consumer_packs_and_migration.md`): the torch capsule
  self-collision residual (the oracle for T6.6) — or the decision that
  `collision/` was cut, which shrinks T6.6 to nothing. Check which
  happened before planning T6.6.
- **M5 is NOT a prerequisite** (sparse trajectory structure is
  orthogonal; see Out of scope).

## Sizing & parallelism

Roadmap label: **L** — and per the codex review (finding 7), treat the
kernel build-out as **multi-month unless evidence shows otherwise**.
Two maintained implementations per kernelized pass is the standing cost.

Ordered spine (each blocks the next):

1. T6.0 (GPU gate) → blocks everything.
2. T6.1 (baseline) → blocks ALL kernel work. No kernel task starts
   before the baseline is committed — it is the bar every kernel must
   clear.
3. T6.2 (Lie wp.func library) → blocks T6.5, T6.7, T6.8 (anything
   using log/exp/right-Jacobians in-kernel). T6.3/T6.4 need only warp's
   transform/quat builtins and may run in parallel with T6.2.
4. T6.3 (FK graduation) → T6.4 (Jacobians) → T6.5 (pose residual) →
   T6.6 (collision) → T6.7 (dynamics, LAST). This is the boundary-table
   priority order: consumer IK throughput is the demand signal;
   FK/Jacobian + one residual + collision land **before** any dynamics
   kernel.

Parallel branches (independent of the kernel spine once T6.1 exists):

- T6.9 (CUDA-graph capture) — captures the **torch-lane** solver loop;
  needs M2b + the GPU box, not the warp kernels. Run it early: it is
  the biggest single end-to-end win (cuRobo evidence).
- T6.11 (implicit-diff `solve()`) — pure torch lane, CPU-developable,
  needs only the M2a/M2b APIs. Can even start before the GPU box is
  confirmed.
- T6.10 (formulation benchmarks) — the frax-style torch variant is
  torch-lane work; its comparison against warp kernels completes only
  after T6.3/T6.7.
- T6.12 (external benchmarks) — the **definition** is written at
  milestone start (before any measurement); the measurement runs last,
  after T6.9 and the kernel spine.

## Per-kernel default-on discipline (applies to every kernel task)

- Every kernel ships **opt-in** behind the M1 lane-select `if` (keyed on
  device, dtype, layout support, warp availability — 03 §2.1). It
  becomes the default for a specific (device, dtype) combination only
  after the owner reviews its committed benchmark against the compiled
  torch lane (03 §2.7). Produce the evidence, **stop for owner review**,
  never flip a default silently (standing rule 8).
- Turning a kernel default-on must not change public behavior:
  unsupported layouts fall back to the torch lane silently (hard error
  only inside capture — T6.9), results match within the stated parity
  tolerances, gradients flow per the pass's differentiable-input matrix.
- Every kernel lands with all five of standing rule 6: (a) an
  adjoint-strategy entry in the design table (03 §2.3 — options:
  generated adjoint with named stored intermediates / hand-written VJP
  kernel / static specialization); (b) parity vs the torch lane
  (fp32/fp64, several batch shapes, both base types); (c) gradcheck vs
  the torch lane including singular points, branched trees, chains >16
  joints (warp's `max_unroll` default is 16 — dynamic-loop adjoints are
  the documented silent-wrong-gradient trap); (d) a warp-CPU parity run
  in CI; (e) a committed benchmark against the compiled torch lane.
- Apply the §9 kernel-author gradient-trap checklist in review of every
  kernel: dynamic loops, in-place `*=`/`/=`, vector-component
  re-assignment, data-dependent `atomic_add`, cross-launch buffer
  aliasing (`audit_warp_platform.md §2.2` has the verified trap table).
- Interop hygiene, always: `wp.from_torch(..., requires_grad=False)`
  with grads managed torch-side (the deferred-grad trap is a measured
  4.3× slowdown — `audit_warp_platform.md §9.2`); launch on torch's
  current stream via `wp.stream_from_torch`; `transformf` for fp32,
  `transformd` for fp64; never reinterpret `[lin, ang]` memory as
  `wp.spatial_vector` (warp is **angular-first** — 03 §2.4); kernels
  live in real `.py` files beside their torch counterparts (e.g.
  `kinematics/_warp_kernels.py`) — warp parses source, no generated
  strings; pre-declare dtype overloads with `@wp.overload` at import so
  first-launch codegen stalls are predictable.

## Tasks

### T6.0 — GPU environment provisioning & verification  [S]

**Goal / done-when:** the gate checklist above passes on the runner; a
short committed note (inside the T6.1 benchmark definition) records
hardware, driver, toolkit, torch/warp versions, and the warp
kernel-cache directory used by CI; the `warp` packaging extra from M1
installs cleanly there (`uv sync --extra warp` or the M1-decided
equivalent).

**Current state:** dev box CUDA broken (verified — see gate section);
`tests/bench/baseline_cuda_l40.json` is a `_status: "PLACEHOLDER"` file
whose name presumes an L40 runner that does not exist; the bench README
references a `docs/claude_plan/accepted/12_regression_and_benchmarks.md`
that is **not in the tree** and a "self-hosted L40 baseline" procedure
with no CI behind it (no `.github/` existed pre-M1; M1 item 10 was to
create CI). Treat those files as aspirational scaffolding, not fact.

**Implementation plan:** run the gate checklist; wire the GPU runner
into the M1 CI as a dedicated job (warp extra, persistent kernel cache,
bounded parity cases — 03 §2.5); fix or replace the stale bench README
claims and rename `baseline_cuda_l40.json` to match the actual GPU (or
generalize the naming to `baseline_cuda_<gpu>.json`).

**What to test:** the CI job runs the M1 warp-CPU parity suite AND a
first CUDA smoke test (one FK kernel launch on `cuda`, parity vs torch
lane at one batch shape) green.

**Pitfalls:** warp kernel caches are content-addressed but
`clear_kernel_cache()` is not multi-process-safe; give the CI job a
persistent, per-runner cache dir. Do not let the smoke test silently
run on CPU — assert `torch.cuda.is_available()` inside it.

### T6.1 — Baseline FIRST: committed GPU/CPU benchmarks  [M]

**Goal / done-when:** committed benchmark **definitions** and committed
**results** exist for: (1) FK batched sweeps, (2) RNEA batched sweeps,
(3) end-to-end solver iterations (single IK and M2b batched IK), each on
Panda-scale (7-dof fixed base) and SMPL-scale (the M3 betas skeleton:
spherical joints + free-flyer), each **before and after**
`torch.compile`, on CPU-eager (the supported floor), CPU-compiled,
GPU-eager, and GPU-compiled; **cold-start costs are part of the
results**: `torch.compile` first-call latency, warp module first-launch
codegen latency (measured here for the first time — the audits could
not), and CUDA-graph record time (fed by T6.9). This baseline is the bar
every kernel must clear. **No kernel work starts before it exists.**

**Current state:** `tests/bench/` exists with pytest-benchmark
micro-benches (`bench_forward_kinematics.py`, `bench_jacobian.py`,
`bench_lie.py`, `bench_solve_ik.py`) — Panda-only, CPU-only in
practice, both baseline JSONs are placeholders. The only measured
compile number in the plan: codex probe, CPU, fixed-base Panda B=256 —
eager ~2.91 ms → steady compiled ~0.269 ms (~10.8×), **first compile
~31.5 s, amortized only after ~12k calls**
(`plan/research/codex_plan_review.md:186`). That number predates the M1
seam rewrite; re-measure, do not quote it as current.

**Implementation plan:**

1. Write the benchmark definition file first (suggested:
   `tests/bench/definitions.md` plus per-case fixtures) — hardware
   header from T6.0, dtypes (fp32 primary, fp64 where the parity suite
   demands), batch sizes `B ∈ {1, 16, 256, 4096}` (B=1 is mandatory —
   it is the known weak spot of one-thread-per-element kernels), warmup
   count, iteration count, statistic (median + IQR), timer discipline
   (CUDA events or `torch.cuda.synchronize()` around walls; never time
   an unsynchronized launch), peak memory
   (`torch.cuda.max_memory_allocated`).
2. Extend `tests/bench/` with the missing cases: RNEA sweeps, SMPL-scale
   model (from M3), free-flyer Panda variant, compiled-lane variants,
   batched-IK end-to-end (fixed iteration budget so timings compare).
3. Record cold starts explicitly as separate benchmark entries, not
   footnotes: first `torch.compile` call per (model, shape); first
   `wp.launch` per (module, dtype, device).
4. Populate the CPU and GPU baseline JSONs from one known-good run each,
   following the bench README's bump procedure (fixing its stale
   references while there, per standing rule 2).

**What to test:** the benchmark suite itself runs green in the CI GPU
job (`uv run pytest tests/bench/ -m bench --benchmark-only`); the
committed JSONs parse and are non-placeholder; the M1 eager-CPU
non-regression benchmark still passes (no CPU regression is a milestone
acceptance criterion).

**Pitfalls:** `torch.compile`'s cold start must never happen implicitly
in a timed region of another case (isolate compile warmup);
`q_neutral` for Panda violates joint-4 limits — clamp before solver
benchmarks (repo CLAUDE.md note); use fixed seeds for IK targets so
successive runs are comparable; commit raw JSON + definition, never a
bare ratio in a doc.

### T6.2 — The wp.func Lie library (explicit prerequisite, not an assumption)  [M]

**Goal / done-when:** a small `@wp.func` library exists providing
SO3/SE3 `exp`, `log`, and right-Jacobian / right-Jacobian-inverse math
usable **inside** warp kernels, in fp32 and fp64, with **validated
adjoints**: gradcheck vs the torch lane passes at and near the singular
points (θ=0, θ=1e-9, θ near π, quaternion double-cover boundary) on
both devices (warp-CPU + CUDA). Warp has **no** SO3/SE3 log/exp
builtins — `quat_from_axis_angle`, `quat_to_axis_angle`,
`transform_multiply`, `quat_rotate`, `quat_slerp` exist, log/exp do not
(`audit_warp_platform.md §4`, verified against warp source). Every
downstream kernel that retracts, differences, or evaluates a pose
residual depends on this task.

**Current state:** the torch-lane source of truth is
`src/better_robot/lie/_torch_native_backend.py` — `so3_exp` (:145),
`so3_log` (:163), `se3_exp` (:259), `se3_log` (:290), Taylor-stitched
via `torch.where` against a θ² cutoff — and
`src/better_robot/lie/tangents.py` — `_so3_jac_coefficients` (:76),
`right_jacobian_so3` (:105), `right_jacobian_inv_so3` (:122),
`right_jacobian_se3` (:184), `right_jacobian_inv_se3` (:204). Two
things will have changed by M6 and must be ported in their **post-M0**
form: the θ=0 NaN-gradient fix (safe-`where` idiom) and the
dtype-dependent Taylor cutoff (today the code has a single
`_TAYLOR_THETA2 = 1e-8` at `_torch_native_backend.py:26`; M0 makes it
per-dtype). Note `so3_inverse` (:138) was already fixed post-audit to
avoid a per-call constant tensor — the audit's per-call-constant table
row is partially stale.

**Implementation plan:**

1. Create `src/better_robot/lie/_warp_funcs.py` (real file — warp
   parses source; `lie` layer placement keeps the DAG clean; the
   `file-naming` convention applies). Implement as `@wp.func`s over
   warp value types: `so3_exp(wp.vec3) -> wp.quat`,
   `so3_log(wp.quat) -> wp.vec3`, `se3_exp(spatial 6-vec as two
   wp.vec3) -> wp.transform`, `se3_log(wp.transform) -> (vec3, vec3)`,
   plus the right-Jacobian 3×3 / 6×6 builders (as `wp.mat33` /
   two-block forms; keep BR's `[lin, ang]` ordering — do NOT use
   `wp.spatial_vector`, it is angular-first).
2. Port the Taylor stitches. In warp, use real `if/else` branches on
   the θ² cutoff rather than compute-both-and-blend: only one branch
   executes, so the torch-lane NaN-pollution problem does not arise —
   but the branch condition must not be loop-carried, and each branch
   must remain single-assignment per local (the component-reassignment
   trap). Match the post-M0 dtype-dependent cutoffs exactly so parity
   holds at the stitch boundary.
3. Provide fp32/fp64 via warp generics or explicit `@wp.overload`s;
   pre-declare both at import.
4. Validate adjoints with `warp.autograd.gradcheck` / `jacobian` (the
   ship-quality differential-testing tool — `audit_warp_platform.md
   §2.3`) AND against the torch lane through tiny launch-harness
   kernels (a `wp.func` cannot be launched directly — write one
   elementwise test kernel per function).

**What to test:** new test module (follow M1's warp test layout;
suggested `tests/warp/test_lie_funcs.py`): value parity vs
`lie/_torch_native_backend.py` over dense samples of θ ∈ [0, π]
including exact 0, 1e-9, cutoff±ε, π−1e-6 (fp32 atol 1e-6 for values;
fp64 1e-12); gradcheck at the singular points in both dtypes; the
round-trip property `log(exp(ω)) == ω` for ‖ω‖ < π; double-cover:
`so3_log(q) == so3_log(-q)` behavior matches the torch lane's
hemisphere convention (§9 quaternion policy from M1). Runs on warp-CPU
in CI and on CUDA in the GPU job.

**Pitfalls:** this is a `wp.func` library for use *inside* whole-pass
kernels — it must NOT grow torch-facing per-op custom-op wrappers
(PyposeWarp's 13.3k-LOC anti-pattern, 03 §2.7). Do not "improve" the
math while porting; the torch lane is the oracle and any divergence is
a parity failure. `se3_log`'s V⁻¹ coefficient has its own stitch
(`_torch_native_backend.py:297-304`) — port it too, not just the SO3
one.

### T6.3 — FK kernel graduation: CUDA validation, GPU benchmark, default-on decision  [M]

**Goal / done-when:** the M1 FK kernel (prototype: warp-CPU parity
only) passes its full parity + gradcheck matrix **on CUDA** — fp32/fp64,
batch shapes incl. B=1 and multi-axis, both base types, branched trees,
>16-joint chains, gradients to q AND joint placements AND frame
placements (the M1 differentiable-input matrix), value-batched cases
from M3, shared-value gradient reduction (unbatched placements ×
batched q — the reduction happens in-kernel, never expanded for torch
to re-reduce); its committed GPU benchmark against the compiled torch
lane exists at all baseline batch sizes; the default-on evidence packet
is delivered to the owner with a stated margin. The kernel is
**production** after CUDA validation; it becomes **default** for
(cuda, fp32) etc. only on the owner's sign-off.

**Current state:** today's torch FK is `forward_kinematics_raw`
(`kinematics/forward.py:73`) — a serial Python loop over `topo_order`
(~4 ms fixed overhead; B=1 costs the same as B=256 — boundary table
row 1). M1 replaced the seam and landed the kernel per the Newton-style
mapping: one thread per execution-batch element, serial topological
loop, int8 kind dispatch, with a recorded adjoint strategy. The old
`backends/warp/` stub (`bridge.py` raising NotImplementedError) was
deleted in M1 — if you still see it, M1 did not land; stop.

**Implementation plan:** (1) run the M1 test matrix on CUDA; fix what
breaks (stream bridging and grad-buffer management are the likely
suspects — they were untestable on CPU). (2) Add the CUDA-only tests M1
could not have: capture-compatibility smoke (the kernel launches inside
a `torch.cuda.graph` region next to torch ops — this is a T6.9
dependency), measured first-launch codegen latency (a baseline entry).
(3) Run the committed benchmark; if the generic kernel loses to
compiled torch at any committed batch size, that is a *finding, not a
failure* — record it and consider T6.10's static specialization before
recommending default-off. (4) Write the default-on evidence packet
(benchmark table + parity/gradcheck summary + adjoint-strategy entry)
and stop for owner review.

**What to test:** everything above is the test. Existing torch-lane
suites (pinocchio parity, `tests/kinematics/`) stay green — the kernel
is opt-in until the owner flips it, and after flipping, the full test
suite runs with the kernel enabled on the GPU job as well.

**Pitfalls:** batch shapes with expanded (stride-0) values cannot alias
zero-copy — the ABI's batch-index map handles them; test that path
explicitly, don't `contiguous()` it away. Don't benchmark with grad
buffers attached unless the case says so — separate forward-only and
forward+backward entries.

### T6.4 — Jacobian kernels, fused with FK where profitable  [M]

**Goal / done-when:** a warp kernel produces frame Jacobians (and/or
the joint-Jacobian table) matching `get_frame_jacobian` /
`compute_joint_jacobians` in LOCAL_WORLD_ALIGNED / WORLD conventions
respectively; the fused-vs-separate decision (one FK+J kernel vs FK
kernel + J kernel) is made by committed benchmark, not taste; all five
kernel requirements met; default-on evidence to owner.

**Current state:** torch lane: `_compute_joint_jacobians_raw`
(`kinematics/jacobian.py:30`) — tree propagation `J[j] = J[parent]` +
per-joint column block writes; `get_frame_jacobian` (:129). cuRobo
fuses FK+spheres+Jacobian into one hand-CUDA kernel (its choice is
partly historical/NVRTC-driven — `audit_curobo_warp_integration.md
§10.1`); Newton threads per-articulation and walks the parent chain
per row.

**Implementation plan:** (1) prototype both mappings against the M1
seam: (a) extend the FK kernel to also write J columns during the sweep
(loop-carried state grows: world axis + origin per ancestor — re-decide
the adjoint strategy, FK's entry does not transfer); (b) a separate
kernel consuming the FK kernel's pose output. (2) Output ABI: canonical
contiguous `(E, 6, nv)` for frame Jacobians (`(E, njoints, 6, nv)` for
the table) — this is the same layout T6.5's residual ABI consumes; pin
it with a layout test. (3) Benchmark both against the compiled torch
lane and each other; commit; owner decides fused/separate and
default-on.

**What to test:** parity vs torch lane for both conventions and both
base types (free-flyer J has the 6-dof base block — test it); gradcheck
of a scalar function of J w.r.t. q (differentiating *through* a
returned Jacobian must route via the torch-lane recompute rule — see
Pitfalls); branched tree + >16-joint chain; batch matrix as T6.3.

**Pitfalls:** the Jacobian is itself a derivative — a user calling
`.backward()` on a J-dependent loss needs the registered backward to
recompute via the torch lane when grad-enabled backward is detected
(03 §2.3 second-order rule); `gradgradcheck` runs through the public
API. Don't let the kernel-produced J silently disagree with the
LWA-vs-WORLD conventions in repo CLAUDE.md — the parity test IS the
convention test.

### T6.5 — ONE residual kernel with gradients-in-forward: pose first  [M]

**Goal / done-when:** a warp pose-residual kernel computes, in a single
forward launch, the **raw** residual `r` (E, 6) and the canonical
contiguous `(E, dim, nv)` Jacobian; torch-side backward is one multiply
(cuRobo pattern (b) — `audit_curobo_warp_integration.md §2.1`);
robust-kernel weighting and IRLS stay torch-side so nothing is
double-weighted and robust row-grouping stays visible (codex finding 9);
committed benchmark vs the compiled torch lane's **analytic-Jacobian**
path; default-on evidence to owner.

**Current state:** torch lane `PoseResidual`
(`residuals/pose.py:58-104`): residual = `se3.log(inverse(target) ∘
T_ee)`, analytic Jacobian = `Jr⁻¹(r) @ R_eeᵀ-rotated J_world`, verified
~39× faster than FD. **Honesty note on the roadmap's "erases the
FD/jacrev Jacobian cost":** for pose, the torch lane already has an
analytic Jacobian — the committed benchmark bar is that path, compiled;
the FD/jacrev erasure argument applies to residuals without analytic
blocks. Also: today's torch residual **pre-weights** `r` and `J` with
`pos_weight/ori_weight` (`pose.py:65-66, 102-104`) while the frozen ABI
says the kernel returns **raw** `r` (plan 03 §2.2; code currently
pre-weights — M2's residual redesign resolves where weights live).
Match the M2 semantics exactly; if M2 kept per-residual static weights
inside the residual, the kernel applies the same weights to both `r`
and `J` identically, and IRLS/robust weights still stay outside.

**Implementation plan:** (1) kernel consumes the FK/Jacobian kernel
outputs (frame pose + frame J) or fuses with them — decide by the T6.4
benchmark topology; per-thread work: compose, `se3_log` (T6.2),
`right_jacobian_inv_se3` (T6.2), one 6×6 by 6×nv multiply. (2) The
Jacobian is computed **inside the forward kernel** analytically — do
not rely on warp's generated adjoint here at all; the backward custom
op is a torch einsum `grad_r · J`. (3) Differentiable-input matrix:
q (via J), target pose (decide and record — consumers optimizing
targets need it; the analytic form gives it nearly free), weights (out;
they are torch-side). (4) Wire into the M2 residual protocol as the
warp lane of the pose residual item; batched-IK end-to-end benchmark
entry showing the per-iteration Jacobian cost drop.

**What to test:** parity of `r` and `J` vs the torch analytic path
(fp32/fp64, both base types, batch matrix); gradcheck of
`sum(loss(r))` w.r.t. q AND target through the public residual API at
singular points (r → 0, i.e. target == current pose — the Jr⁻¹ Taylor
branch); an IRLS integration test: one LM solve with Huber on the warp
lane matches the torch lane within stated tolerances (proves no
double-weighting); the `(E, dim, nv)` layout pinned by stride assertions.

**Pitfalls:** `r.new_tensor([...])` per call in today's torch residual
(`pose.py:65`) is an H2D sync the M1/M2 hoisting should have removed —
if it survived, fix the torch lane first or the benchmark flatters the
kernel. Keep robust row-group semantics: the kernel returns per-residual
raw rows; grouping metadata stays in the torch-side residual item.

### T6.6 — Collision warp-led: capsule self-collision  [M]

**Goal / done-when:** a warp capsule-capsule self-collision kernel
(segment-segment closest distance over the model's collision pairs,
gradients-in-forward, same `(E, dim, nv)`-style ABI as T6.5 for its
residual form) lands with parity against **M4's torch reference** — the
torch implementation ships first/alongside per the boundary table
("green-field describes the design freedom, not permission to skip the
oracle"); feeds the M4 `SelfCollisionResidual`; five kernel
requirements; default-on evidence to owner.

**Current state:** `collision/` is 100% stubs today
(`collision/closest_pts.py` — `point_to_segment`, `segment_to_segment`
both `raise NotImplementedError`; `pairs.py`, `robot_collision.py`
likewise). M4 either ported the old capsule mode as a torch residual or
cut the package (`m4_consumer_packs_and_migration.md`) — **check which
before starting; if collision was cut, this task is void and the
boundary-table row is updated to say so.**

**Implementation plan:** (1) torch oracle from M4: batched
segment-segment distance + per-pair residual with analytic gradient.
(2) Warp kernel: one thread per (execution-batch element, collision
pair); inputs are the FK kernel's world capsule endpoints (fusion with
FK is a benchmark question, same method as T6.4); outputs per-pair
distance/violation + gradient w.r.t. the two endpoint positions,
chain-ruled to q via the frame Jacobians torch-side or in-kernel
(decide by benchmark; record in the adjoint table). (3) No
data-dependent `atomic_add` in the gradient path (the scatter trap) —
per-pair outputs are dense, reduction is torch-side.

**What to test:** parity vs the M4 torch reference (distance values,
gradient direction at near-contact and deep-penetration
configurations); gradcheck at the parallel-segments degenerate case
(the classic segment-segment singularity — the torch oracle defines the
tie-break); an IK-with-collision end-to-end test on a humanoid model
(SMPL-scale) where the solve avoids self-intersection.

**Pitfalls:** parallel-segment closest points are non-unique — the
kernel must implement the same tie-break as the torch oracle or parity
fails intermittently. Distances have a `sqrt` at 0 — clamp per the
torch oracle's epsilon policy.

### T6.7 — Dynamics kernels LAST: RNEA → CRBA/centroidal → ABA  [L]

**Goal / done-when:** warp kernels for RNEA, CRBA + centroidal, ABA,
each with its **own re-decided adjoint strategy** (they carry far more
loop-carried state than FK: RNEA's forward (v, a, f) sweep + backward
force transport; ABA's three passes with per-joint (IA, pA, D⁻¹);
CRBA's O(depth²) ancestor walks — FK evidence does not generalize,
codex finding 4); five kernel requirements each; committed benchmarks
against BOTH the compiled Featherstone torch lane AND T6.10's
ancestor-mask torch variant; default-on evidence per kernel to owner.

**Current state:** torch lane: `rnea` (`dynamics/rnea.py:71`), `aba`
(`dynamics/aba.py:36`, per-joint `linalg.inv` — 1×1 for revolute),
`crba` (`dynamics/crba.py:29`, nested ancestor `while` walk), `ccrba`
(`dynamics/centroidal.py:111`). All Python tree loops today; M1/M3
hoisted the per-call 6×6 inertia rebuilds
(`spatial/inertia.py` `_to_6x6` per joint per call) into precomputed
static-value buffers — parametric inertias derive per evaluation
context (03 §5).

**Implementation plan:**

1. **Order by consumer demand:** RNEA first (feeds M4's contact-force
   task), then CRBA + centroidal (BHF force pipelines), ABA last (its
   main consumer, `dynamics/action/`, was parked in a branch in M1 —
   confirm demand before spending the effort; if none, deliver the
   evidence and let the owner defer it).
2. Before each kernel, check T6.10's result: if the compiled
   ancestor-mask torch variant already meets the consumer's throughput
   need at their DOF scale, present that to the owner — a kernel that
   cannot beat the *best* torch lane is not written (the bar is the
   best committed torch number, not today's loop).
3. Adjoint strategy per kernel, recorded in the design table before
   coding: RNEA's reverse sweep is the classic case for a hand-written
   VJP kernel (cuRobo hand-writes it; ~half its kinematics kernel code
   is backward machinery — that is the honest price tag); stored
   intermediates (write out per-joint v, a, f during forward) make the
   generated adjoint viable; static specialization is the third option.
   Prototype cheapest-first, gradcheck decides.
4. Differentiable-input matrix per pass: RNEA — q, v, a, joint
   placements, **inertias**, gravity; CRBA — q, placements, inertias;
   ABA — q, v, tau, placements, inertias. Inertia VJPs were explicitly
   priced into M6 (codex finding 6) — they are not optional.
5. ABA's per-joint solves: scalar reciprocal in-kernel for 1-dof
   joints; for the 6×6 free-flyer block use Newton's proven recipe —
   nop-grad factorization + implicit-function-theorem adjoint on the
   solve (`references/sim/newton/newton/_src/solvers/featherstone/
   kernels.py:1466-1580`, verified) — never `wp.tile_cholesky` (no
   adjoint).

**What to test:** pinocchio-parity for each pass through the warp lane
(the `tests/test_pinocchio/` suite parametrized over the lane);
gradcheck w.r.t. every input in the pass's matrix, incl. inertias, on
branched trees and >16-joint chains, both base types; `tau` round-trip
`rnea(q, v, aba(q, v, tau)) ≈ tau` on the warp lane; warp-CPU parity in
CI; committed benchmarks at Panda, SMPL, and the T6.10 DOF sweep points.

**Pitfalls:** spatial algebra convention — BR `[lin, ang]` vs warp
spatial builtins angular-first; use two `wp.vec3`s and BR's own cross
products (T6.2 library), or an explicit tested conversion around any
warp spatial builtin — never raw reinterpretation (03 §2.4).
`compute_*_derivatives` stays torch-lane autograd (boundary table) —
do NOT try to make warp adjoints serve it; but verify the derivative
APIs still work when the underlying pass runs on the warp lane (they
must route through the torch lane — that routing is part of the lane
select, test it).

### T6.8 — `integrate`/`difference`: kernel only on benchmark evidence  [S]

**Goal / done-when:** a committed benchmark compares M3's vectorized
torch `integrate`/`difference` rewrite (grouped by joint kind — no
longer a serial tree scan) against a candidate warp kernel sketch; a
kernel is built **only if** it beats the vectorized torch rewrite on
that benchmark; otherwise the decision "torch stays" is recorded in the
boundary table with the numbers. Expected outcome per 03 §2.2: torch
stays.

**Current state:** today `Model.integrate/difference`
(`data_model/model.py:160-190`) is a per-joint Python loop; M3 item 6
vectorizes it by kind grouping. The old boundary-table draft committed
this to warp; codex finding 9 reversed that to benchmark-gated.

**Implementation plan:** benchmark the M3 rewrite on GPU at the
baseline batch matrix (it sits on every solver step); only if a >
stated-margin win is plausible from the profile (e.g. launch-bound at
small B), prototype a one-thread-per-element kernel using T6.2's
exp/log and run the comparison; deliver numbers to owner either way.

**What to test:** if no kernel: nothing new beyond the committed
benchmark. If kernel: full standing-rule-6 matrix like every other
kernel (retraction singular points come from T6.2's tests).

### T6.9 — CUDA-graph capture of solver inner loops  [M–L]

**Goal / done-when:** a `GraphExecutor`-style utility (cuRobo's
pattern, verified at `references/kin_dyn/curobo/curobo/_src/util/
cuda_graph_util.py:13-221`) records `inner_iters ×` M2b `update` —
**including the autograd backward** — with warmup-then-record
discipline, replays until shapes change, and re-records on an explicit
resize event; proven by an actual **capture/replay-vs-eager parity
test** on GPU (per-element solution and status equality within stated
tolerances after N replayed iterations vs N eager iterations), never by
checklist alone; a committed benchmark demonstrates the end-to-end
batched-IK win (eager loop vs captured replay, fixed iteration budget,
several B); inside capture, the warp-lane layout fallback becomes a
**hard error**.

**Current state:** M2b's `update` satisfies the structural
capture-readiness checklist (fixed input buffers via `copy_`,
branch-free `torch.where` logic, `cholesky_ex` info-mask fallback as
fixed tensor work, no host syncs in `update`, syncs only at `run`
boundaries) — accepted there by inspection, certified here. cuRobo's
verified recording protocol: `gc.collect()` + `torch.cuda.synchronize()`
(flush pending graph destructions), clone inputs, **3 warmup runs on a
side stream**, `with torch.cuda.graph(graph, pool=mem_pool,
stream=stream)`, then replay with `copy_` into captured inputs
(skipping same-`data_ptr`) — and it captures full autograd:
`cost.backward()` runs inside the graph
(`audit_curobo_warp_integration.md §3`).

**Implementation plan:**

1. Implement the executor in `optim/` (suggested
   `optim/graph_executor.py`; do not resurrect the M1-deleted
   `backends/graph_capture` API). Surface: wrap a
   `capture_fn(*tensors) -> tensors`; lazy record on first call;
   `reset()` clears graph + buffers (shape change ⇒ owner resets or
   makes a new executor). CUDA-only; on CPU or when disabled it calls
   the function eagerly — one code path for users.
2. Wire it around `inner_iters × update` in `run` (outer loop keeps
   host-side early-exit at replay boundaries, cuRobo's outer/inner
   split; `fixed_iters=True`-style option skips even that sync).
3. Warp-lane interplay: warp launches are capturable because M1's
   bridge always launches on torch's current stream — during capture
   that IS the capture stream (cuRobo documents exactly this). Add the
   in-capture hard error to the lane select. Draft wording:
   `RuntimeError: better_robot: input '<name>' has an unsupported
   layout for the warp lane and a CUDA graph capture is active; the
   silent torch-lane fallback is disabled during capture because a
   hidden .contiguous() copy would be baked into the recorded graph.
   Make the input contiguous before capture, or solve without graph
   capture.` (cuRobo's `tensor_checks.py` documents the identical
   rationale.)
4. Custom-residual eligibility: document (in the M2a author guide) the
   rules — torch ops only or capture-tested kernels, fixed shapes, no
   host syncs, no data-dependent Python branching; an ineligible
   residual disables capture with a warning, never a corrupt graph.
5. Measure and commit graph-record time (a T6.1 cold-start entry) and
   replay-vs-eager per-iteration time.

**What to test:** `tests/optim/test_graph_capture.py` (GPU job only):
(a) parity — batched IK, 128 targets, N iterations captured vs eager,
per-element q and status within M2b's stated tolerances; (b) re-record
on resize — change B, assert a new record happens and results stay
correct; (c) replay stability — 100 replays with changing inputs via
`copy_`, no drift, no allocator growth after warmup
(`torch.cuda.max_memory_allocated` plateau); (d) the in-capture layout
hard error fires (feed a non-contiguous input under capture, assert
the exact error); (e) a warp-lane kernel inside the captured region
(FK from T6.3) replays correctly — the mixed torch+warp capture proof.

**Pitfalls:** warmup MUST run on a side stream that then syncs
(cuRobo's protocol) — warming up on the capture stream corrupts state;
autograd inside capture requires all backward buffers pre-allocated
and stable — the M2b deferred-candidate-Jacobian trick must not
allocate on first replay; convergence early-exit reads a GPU scalar —
keep it strictly at replay boundaries or behind `fixed_iters`;
`wp.capture_while` is OUT of scope — warp only recognizes captures it
started itself (mixed torch/warp *external* capture via warp's
conditional nodes is undemonstrated; 03 §2.6).

### T6.10 — Formulation benchmarks: ancestor-mask torch variant; generic vs specialized warp mappings  [M]

**Goal / done-when:** two committed benchmark studies with owner
decisions attached:
(1) **frax-style ancestor-mask RNEA/CRBA as a torch-lane variant** —
propagation as `ancestor_mask @ X` matmuls + einsums, no recursion
(`references/design/frax.md §5`, port from
`references/kin_dyn/frax/frax/core/robot.py:913-998`) — benchmarked
against the compiled Featherstone torch lane and (when built) the T6.7
warp kernels at **25–160 DOF** (synthetic chains/trees via
`br.ModelBuilder`, plus Panda, SMPL, G1), eager+compiled, CPU+GPU,
baseline batch matrix.
(2) **serial-generic vs static-specialized warp mappings** — the
generic one-thread-per-element kernel vs a per-robot statically
specialized codegen variant (fixed loop bounds → unrolled → adjoint
replay-safe; mjwarp's `cache_kernel` factory pattern,
`references/sim/mujoco_warp/mujoco_warp/_src/warp_util.py:122-141`) on
FK first, RNEA if built — with **B=1 latency** as a first-class column
(the known weak spot). Static specialization is the sanctioned escape
when generic kernels miss their target (03 §2.1) — **never a second
kernel language**.

**Current state:** the ancestor-mask formulation is O(N²)–O(N³) FLOPs
but pure dense contractions — frax validates it against pinocchio;
`frax.md` §10.1 predicts it ports almost verbatim to torch einsums and
carries `(B...)` trivially. No mask exists on ModelStructure yet
(`audit_compute_pass_inventory.md §4` — `supports`/`subtrees` are
Python tuples; M1 added flat topo tensors, check whether an ancestor
mask was included).

**Implementation plan:** (1) build `ancestor_mask (njoints, njoints)`
bool from ModelStructure (static, cached); implement `rnea`/`crba`
mask-variants as internal alternates behind the same pass signature —
they are torch-lane *candidates*, not replacements: swapping the
default torch algorithm is an owner decision requiring the pinocchio
parity suite green on the variant. (2) Benchmark per the committed
definition; find the DOF/batch crossover points. (3) For the warp
mapping study, generate the specialized module per robot via a kernel
factory keyed on (robot hash, dtype) with warp's content-addressed
cache; measure codegen cost per specialization as part of the result
(it is the price of the escape hatch). (4) Deliver both studies to the
owner: they gate T6.7 scope (see T6.7 step 2) and the default-lane
choice at humanoid scale.

**What to test:** mask-variant parity vs Featherstone torch lane
(rtol 1e-5 fp32) across joint kinds × base types × batch shapes;
gradcheck of the mask variant (it is shallow dense autograd — frax's
bet); specialized-kernel parity+gradcheck equals the generic kernel's.

**Pitfalls:** don't extrapolate the 160-DOF result from 25-DOF numbers
— the O(N³) CRBA einsum cliff is real (`frax.md` §9); per-robot
specialization multiplies kernel-cache entries — bound the cache and
document eviction in the CI job; the mask variant must keep gradients
to inertias/placements (it is still the torch lane — full autograd).

### T6.11 — Implicit-diff `solve()` (torch lane, flagship)  [L]

**Goal / done-when:** `solve()` (the M2 `run` entry) is differentiable
w.r.t. declared external parameters via **implicit differentiation** —
a `torch.autograd.Function` whose backward solves the optimality system
(jaxopt `root_vjp` pattern) instead of unrolling the iteration —
implementing exactly the contract decided in M2a. The retracted "~100
lines" estimate stays retracted: the real contract covers **manifolds**
(the optimality system lives in the tangent space; VJPs route through
the tangent-space autograd helper), **active bounds** (KKT conditions
at stalled-at-bounds elements: gradients through the active-set
projection, per the M2a-decided semantics), **robust kernels** (the
fixed point is of the IRLS-weighted system), **singular systems**
(damped/least-squares fallback in the backward solve + a per-element
grad-quality status), and the **optimized-vs-external parameter split**
(which inputs get implicit grads: targets, weights, model values —
declared, not inferred). Torch lane only; works on CPU; second-order
composition documented (the backward's linear solve is itself
differentiable torch — state what is and is not supported).

**Current state:** M2a decided the differentiation contract questions
(state shape, per-element statuses, what non-convergence means for
gradients) precisely so this implementation would not be blocked —
read that contract first (`m2a_variable_blocks_and_slice.md`); the
sketch is 03 §4 "Implicit differentiation". Reference:
`references/design/jaxopt.md` (root_vjp / fixed-point VJP machinery).

**Implementation plan:** (1) wrap `run` in an autograd.Function; save
converged `values*` + external params; backward: assemble the
optimality residual `F(values*, params) = 0` (tangent-space gradient of
the robustified cost, with active-bound rows handled per contract),
solve `(∂F/∂values)ᵀ λ = -grad_out` with the M2 linear solvers
(`cholesky_ex` + damping fallback), return `(∂F/∂params)ᵀ λ`. (2)
Per-element handling: non-converged elements get the contract-defined
behavior (zero grads + status flag, or best-effort with a warning —
whatever M2a fixed; never silent garbage). (3) Expose as an opt-in
argument (e.g. `solve(..., differentiate="implicit")`); the default
stays non-differentiable-through-the-loop (unrolled differentiation is
NOT offered unless M2a's contract says otherwise — it holds graphs for
every iteration).

**What to test:** `tests/optim/test_implicit_diff.py`: (a) implicit
grads vs unrolled-autograd grads on a small well-conditioned problem
(few iterations so unrolling is exact), stated tolerances, fp32 and
fp64; (b) grads w.r.t. target poses on a Panda IK solve match
FD-of-the-solve directionally (loose FD tolerance — the solve is
noisy); (c) a bounds-active element behaves per the contract (test the
contract, not a wish); (d) a robust-kernel (Huber) solve's implicit
grads match unrolled on an outlier problem; (e) non-converged elements
produce the contracted status/grad behavior; (f) batched: per-element
grads for 16 independent problems match 16 individual solves.

**Pitfalls:** the optimality system uses the **robustified** gradient
(M0's IRLS-consistency fix carries through — accepting on ρ means the
fixed point is of ρ, not raw L2); manifold blocks: the system is in
tangent coordinates at `values*` — mixing state-space and tangent-space
Jacobians here is the classic silent-wrong-gradient bug; do not reuse
the forward solve's last factorization blindly — it may be the damped
`JᵀJ + μI`, and the implicit system wants the undamped (or
contract-stated) operator at the solution.

### T6.12 — External reference benchmarks: cuRobo and a JAX-class library  [M]

**Goal / done-when:** committed, caveat-stated comparisons: (1)
**batched-IK throughput vs cuRobo** on a comparable Panda-class
problem; (2) **FK/dynamics sweeps vs a JAX-class library** (mjx and/or
pyroki) where an apples-to-apples definition exists — same robot, same
outputs, same precision, or the case is dropped with the mismatch
documented. The benchmark **definition is committed at milestone start,
before any measurement**, containing: hardware, batch sizes (including
B=1), robot, target set + seeds, fixed iteration budget, success
tolerance (position/orientation thresholds), collision on/off,
precision, warm/cold timing, memory. **A gap vs cuRobo is EXPECTED and
stated openly** — BR deliberately gives up per-robot NVRTC
specialization and 11.5k lines of hand-written CUDA
(`audit_curobo_warp_integration.md §1`); if the generic kernels miss
the target, the sanctioned escape is T6.10's warp static
specialization, never a second kernel language. Committed numbers with
stated caveats — **no bare ratios anywhere** (standing rule 4).

**Current state:** an in-tree cuRobo checkout exists at
`references/kin_dyn/curobo/` — but it is **cuRoboV2, tag
`v0.8.0-2-gca94158`, a research release**, not the public v0.7.x most
published numbers describe (`audit_curobo_warp_integration.md` header).
All cuRobo performance numbers in the digests are paper claims — nothing
was locally verified. No comparison harness exists.

**Implementation plan:** (1) write and commit the definition file
(suggested `tests/bench/external/definition.md`) FIRST — get owner
sign-off on it before measuring (it decides what "comparable" means:
same success tolerance, same iteration budget or same wall-clock
budget — state which and why; run both if cheap). (2) Install cuRobo in
a **separate venv** on the GPU box (it needs its own CUDA build chain);
prefer the in-tree V2 for provenance, fall back to the public release
if V2 does not build — record exactly which commit/wheel ran. (3) Run
BR's best configuration (warp kernels default-on where the owner
approved, graph capture on, compiled torch lane elsewhere) and worst
honest configuration (eager torch lane) — publish both so the
architecture's spread is visible. (4) JAX-class: pyroki for IK-shaped
comparison, mjx for FK/dynamics sweeps; match precision (fp32) and
count JIT compile time as their cold start, symmetrical to ours. (5)
Write the results doc with per-case caveats (solver differences,
collision model differences, seed sensitivity) and deliver to owner.

**What to test:** the harness itself is committed and re-runnable
(`tests/bench/external/`, excluded from default CI, run on the GPU
job on demand); a smoke test asserts the harness produces the committed
JSON schema.

**Pitfalls:** do not tune BR on the benchmark targets and report the
tuned number as default behavior — report default-config and
tuned-config separately; cuRobo's IK success criterion (its own
position/rotation thresholds) must be mapped to BR's, not assumed
equal; memory numbers must separate allocator pool growth from true
working set (`torch.cuda.memory_stats`, warp mempool stats); B=1
latency will likely be BR's worst column — publish it anyway.

## Milestone acceptance checklist

- [ ] GPU gate passed and recorded; CI GPU job runs warp-CPU parity +
      CUDA suites green (T6.0).
- [ ] Baseline benchmark definitions + results committed (FK/RNEA
      sweeps, solver iterations, Panda + SMPL scale, eager + compiled,
      CPU + GPU, cold starts incl. measured torch-compile, warp
      codegen, and graph-record costs) — and it landed before any
      kernel work (T6.1).
- [ ] `wp.func` Lie library: fp32/fp64, value parity + adjoint
      gradcheck vs torch lane at singular points, warp-CPU + CUDA
      (T6.2).
- [ ] Every landed kernel (FK, Jacobian, pose residual, collision,
      each dynamics pass) has all five standing-rule-6 artifacts: an
      adjoint-strategy design-table entry, parity, gradcheck (singular
      points, branched trees, >16-joint chains), warp-CPU CI run,
      committed GPU benchmark (T6.3–T6.7).
- [ ] Per-kernel committed benchmarks exist; every **default-on**
      kernel beats the compiled torch lane by its stated margin on the
      committed benchmark, and each default flip has an owner sign-off
      recorded per (device, dtype) (all kernel tasks).
- [ ] Pose-residual kernel ABI verified: raw `r` + contiguous
      `(E, dim, nv)` J; IRLS/robust weighting torch-side; the
      no-double-weighting integration test passes (T6.5).
- [ ] `integrate`/`difference` decision recorded with benchmark
      evidence — kernel only if it beat M3's vectorized torch rewrite
      (T6.8).
- [ ] Capture/replay matches eager within stated tolerances (actual
      parity test, incl. a warp kernel inside the graph); re-record on
      resize works; in-capture layout fallback hard-errors with the
      documented message; the end-to-end batched-IK capture win is a
      committed benchmark (T6.9).
- [ ] Formulation studies committed: ancestor-mask vs Featherstone vs
      warp at 25–160 DOF; serial-generic vs static-specialized incl.
      B=1; owner decisions recorded (T6.10).
- [ ] Implicit-diff `solve()` implements the M2a contract: manifolds,
      active bounds, robust kernels, singular systems, parameter split;
      implicit-vs-unrolled gradcheck green (T6.11).
- [ ] External benchmarks: definition committed before measurement;
      cuRobo + JAX-class results committed with caveats; no bare
      ratios; expected-gap statement present (T6.12).
- [ ] Torch-lane parity suite (incl. `tests/test_pinocchio/`) green
      throughout — on every intermediate commit, not just the last.
- [ ] **No CPU regression**: the M1 committed eager-CPU non-regression
      benchmark still passes at milestone end.
- [ ] Docs/CLAUDE.md truth pass for everything this milestone changed:
      lane defaults, capture behavior, cold-start expectations, the
      boundary table's decided rows (standing rule 2).

## Out of scope

- **`wp.capture_while` / warp conditional graph nodes** — out until
  mixed torch/warp *external* capture is demonstrated (03 §2.6); warp
  only recognizes captures it started itself.
- **Hand-written CUDA C++** — anti-goal, permanently (03 §2.7). The
  escape hatch for missed targets is warp static specialization
  (T6.10), never a second kernel language.
- **Per-op Lie kernels / torch-facing wrappers for T6.2's wp.funcs** —
  the PyposeWarp anti-pattern. The Lie facade (`se3.compose`, …) stays
  torch.
- **Moving JᵀJ assembly, Cholesky/lstsq, robust kernels, or optimizer
  control flow into warp** — boundary table says torch;
  `wp.tile_cholesky` has no adjoint; graph capture already erases the
  launch overhead.
- **Analytic Carpentier–Mansard dynamics derivatives** — separate
  roadmap item; `compute_*_derivatives` stays torch-lane autograd.
- **Batched LBFGS** — its own later milestone (03 §4).
- **Sparse-trajectory kernels / banded solvers on warp** — M5 owns
  trajectory structure (`m5_sparse_trajectory_structure.md`); M6 does
  not kernelize it.
- **fp16/bf16 kernels** — bf16 does not exist in warp; fp16 is
  rejected until the §9 dtype policy designs it.
- **Promoting the `warp` extra to a hard dependency** — packaging/owner
  decision, only if/when kernels are default-on for CUDA (03 §10).
- **Retargeting, new residual families, viewer work** — M4 territory.

## References

- `plan/04_roadmap.md` — M6 section: the items and done-when this file
  expands; the standing kernel rule.
- `plan/03_architecture.md` §2.1–§2.7 (two-lane model, boundary table,
  bridge + adjoint strategies + differentiation contract, layout/ABI,
  CPU policy, capture pattern, anti-goals), §4 (optimizer stack,
  implicit diff), §7 steps 5–7, §9 (warp kernel policy — the normative
  checklist).
- `plan/research/audit_warp_platform.md` — what warp can/cannot do:
  interop mechanics, adjoint traps, no Lie builtins, no double
  backward, warp-CPU is single-threaded, graph APIs. Primary evidence.
- `plan/research/audit_curobo_warp_integration.md` — where cuRobo's
  speed actually comes from; GraphExecutor; wrapper patterns (a/b/c);
  gradients-in-forward; the hard-error-under-capture rationale; LOC
  budgets. Primary evidence.
- `plan/research/audit_compute_pass_inventory.md` — all 53 passes,
  loop structures, sync inventory, flat-metadata needs.
- `plan/research/codex_warp_review.md` — the binding corrections
  (bridge, ABI, adjoint safety, second-order routing, differentiable
  inputs, capture criteria, external-benchmark definition).
- `references/design/curobo.md`, `warp.md`, `mujoco_warp.md`,
  `newton.md` — digests (treat claims as unverified where the audits
  flagged them); `frax.md` §5/§10 — the ancestor-mask formulation for
  T6.10; `pyroki.md`, `jaxopt.md` — T6.11/T6.12 references.
- Source evidence verified for this file:
  `references/kin_dyn/curobo/curobo/_src/util/cuda_graph_util.py`
  (GraphExecutor), `references/sim/newton/newton/_src/solvers/
  featherstone/kernels.py:1466-1580` (nop-Cholesky + implicit solve
  adjoint), `references/sim/warp/docs/user_guide/interoperability.rst`
  (custom-op pattern, deferred-grad trap),
  `references/kin_dyn/frax/frax/core/robot.py:913-998` (mask RNEA/CRBA).
- Sibling instruction files: `m1_two_lane_seam_and_hygiene.md` (seam,
  ABI, bridge, FK kernel this milestone graduates),
  `m2a_variable_blocks_and_slice.md` (differentiation contract,
  author guide), `m2b_batched_second_order_solvers.md` (capture-ready
  `update` certified here), `m3_parametric_model_breadth.md` (SMPL
  skeleton, vectorized integrate/difference, value-batch coverage),
  `m4_consumer_packs_and_migration.md` (collision torch reference,
  consumer parity benchmarks).
