# study_backend — fastest batched kinematics/dynamics with one implementation

Inputs: `scan_backend.md` (repo audit), reference summaries in
`/home/rikhat.akizhanov/better/BetterRobot/references/design/` (curobo.md, newton.md,
mujoco_warp.md, pyroki.md, brax.md). Budget note: no web search was run (token budget).
Facts about torch.compile / Warp / pinocchio that come from prior knowledge, not from a
source read here, are tagged **[PK]**; every one of them is re-checked by an experiment
in section 2. Numbers tagged **[EST]** are my estimates from the cost model below.

## 0. Cost model (derived from this repo's own numbers)

- Eager torch on GPU is CPU-bound at these sizes. Measured: the batched Jacobian pass is
  52 launches and 0.842 ms on mhr203 b256 (`plan/for_agents/02_results.md:34`), i.e.
  **~16 us per aten op**; eager matrix FK on the 203-joint model is ~2 ops/joint and
  9.5 ms (`docs/CHANGELOG.md:22-25`), i.e. **~23 us per op**. Time is flat from batch 1 to
  4096 (`scan_backend.md` section 3). So for eager: `t ≈ 20 us × (#ops)` until tensors
  reach ~50-100 MB, after which `t ≈ bytes moved / ~0.8 TB/s` (RTX 6000 Ada ~960 GB/s).
- Therefore the only two levers are: (1) cut op count so it does not grow with njoints,
  (2) cut per-op cost (CUDA graphs / fusion / one fused kernel). Everything below is
  judged by these two numbers: ops per call, and bytes per call at large batch.
- Per-op count of today's code grows with njoints (FK chain `_fk_matrix.py:262`, RNEA
  `rnea.py:186,234,241`, CRBA `crba.py:75,87,99`, ABA `aba.py:163,196,238`).

## 1. Strategy table

Speeds: FK forward / RNEA forward+backward, f32, one modern GPU (RTX 6000 Ada class).
"30-DoF" = humanoid tree, depth ~9-10; "200-DoF" = human-with-hands tree (MHR-203 class),
depth ~15-20. All [EST] unless cited.

| | (A) eager torch, depth-independent passes | (B) A + torch.compile / CUDA graphs (optional, same code) | (C) hand CUDA kernels + analytic backward in autograd.Function (cuRobo) | (D) Warp kernels as the only impl | (E) internal IR → emit torch or fused kernel |
|---|---|---|---|---|---|
| ops/launches per call | constant: ~30-40 FK, ~60-100 RNEA | 1 graph replay; inductor fuses to ~5-20 kernels | 1-2 kernels FK (curobo.md:171-174) | 1-3 kernels per pass | 1 kernel (GPU emitter) |
| 30-DoF, b1 | FK 0.6-0.8 ms; RNEA f+b 3-5 ms | FK 40-80 us; RNEA f+b 0.1-0.3 ms (graph replay + guards) | FK+bwd ~20-40 us; RNEA f+b **31.6 us/iter** (cuRobo v2 Table 4, curobo.md:210) | FK 0.54 ms measured here (bridge overhead, `warp_fk_..._b1.json`); ~50 us achievable | like C |
| 30-DoF, b4096 | same as b1 (overhead-bound) | ~same as b1 | ~same as b1 | 0.63 ms measured | like C |
| 30-DoF, 1M | FK 100-200 ms, peak mem several GB (each intermediate (1M,30,4,4) f32 = 1.9 GB) | 10-30 ms (fusion cuts traffic 5-10x) | 1-3 ms (bandwidth: read q 120 MB, write poses 840 MB) | 1-3 ms | 1-3 ms |
| 200-DoF, b1 | FK 0.8-1.0 ms; RNEA f+b 4-6 ms | 60-150 us; 0.2-0.5 ms | ~50-100 us (dual-kernel path for large trees, curobo.md:174) | ~0.1 ms | like C |
| 200-DoF, b4096 | FK 4-6 ms (52 MB/intermediate) | 1-2 ms | ~0.5 ms | ~0.5 ms | like C |
| 200-DoF, 1M | needs chunking; ~1-1.5 s | ~0.2-0.4 s | ~20-40 ms | ~20-40 ms | ~20-40 ms |
| double backward | yes | **no through compiled region [PK]**; fall back to eager | only if backward is itself written as differentiable torch ops or a second kernel pair | no (wp.Tape is not torch autograd; newton.md:300) | only for the torch emitter |
| torch.func (vmap/jacrev/jvp) | yes (pure ops) | compile *inside* functorch transforms works for many cases; jacrev *of* a compiled fn partly [PK] | needs `setup_context` + `vmap` staticmethod; jvp needs a forward-mode kernel | no | torch emitter only |
| grads w.r.t. model params (link lengths, inertias, shape) | free (ModelValues are tensors) | free | extra adjoint outputs per pass, hand-written | hand adjoints; repo says Warp's auto-adjoint is wrong for the dynamic topo loop (`kinematics/_warp_bridge.py:5-8`) | must be generated |
| CPU | yes (eager; ~15-25 us/op → b1 FK ~0.3-0.6 ms) | inductor C++ backend works on CPU [PK]; modest gain | needs separate CPU code → second impl | Warp CPU backend exists, slow compile [PK] | needs CPU emitter |
| maintenance | lowest: one torch file per pass | + compile guards, recompile tests | highest: CUDA + hand adjoints for ~8 passes × joint kinds × params | high: ~2.2k lines today for 2 passes fwd-only; adjoints missing | high: IR + 2 emitters + AD on IR |
| build/packaging | none | none (Triton ships with torch) | nvcc or runtime JIT (cuRobo uses cuda.core JIT, curobo.md:105) | warp-lang dep, kernel cache | JIT pipeline |
| risk | low; f32 world-frame accuracy to check | medium: graph breaks, recompiles, compile time | high correctness risk; buys 10-50x only at b1/1M | high; inverts "torch is truth" | highest |

Reading of the table:
- At batch ≤ 4096 everything is overhead-bound; the gap A→B is ~10x and B→C is ~2-4x.
- At 1M the gap is bandwidth: A→B ~5-10x, B→C ~5-10x. Only C/D/E win there.
- Only A and B keep every autograd feature the four use cases need (double backward for
  NN-layer use and Gauss-Newton-through-gradients, torch.func for optim, shape gradients
  for human fitting). C/D keep them only at multiplied maintenance.

## 2. Recommendation and staged path

**Recommendation: A as the single implementation, with B as an optional accelerator of
exactly the same functions. Delete Warp. Do not start C/D/E; keep C as a gated, per-pass
escape hatch that replaces (never duplicates) the torch pass only if E5 below shows a
need that B cannot meet.**

Why: Warp's backward already *is* torch (`kinematics/_warp_bridge.py:72`), so deleting it
loses ~0.6 ms of FK forward overhead and nothing else; the repo's biggest wins came from
op-count reduction (Jacobian 23-56x, FK 16x); and pyroki/brax show the "precomputed
order + batched ops + compiler" pattern is the JAX-world baseline (pyroki fori_loop over
a topo sort, pyroki.md:116,307; brax unrolls over depths, brax.md:370).

Stages:
1. **S0 measure (1-2 days).** Experiments E1-E3 below on current HEAD; store JSON.
2. **S1 depth-independent eager passes.** Rewrite FK chain, RNEA, CRBA, CCRBA in the
   form of section 3; ABA level-synchronous. Delete `_warp_*` (2206 lines) and the
   `use_warp`/`use_compile` flags. Analytic RNEA derivatives as a separate algorithm.
3. **S2 accelerator.** One `better_robot.accelerate(fn, *, static_batch=True)` style
   wrapper (name open) that applies `torch.compile(mode="reduce-overhead")` with batch
   bucketing (pad to powers of 2) to any raw pass; used by optim's inner loop and by
   benchmarks. No per-function flags. Eager stays the default and the truth.
4. **S3 (gated)** fused kernel for FK+frames+VJP only, if E5 says so.

Decision experiments (same GPU, f32, models: panda 7, G1/H1 humanoid ~30, SMPL 24,
SMPL-X 55, MHR-203 (68 joints, nv 204), synthetic 200-joint tree and 64-link chain;
batches 1/256/4096/65536):
- **E1 FK prototype.** Pointer-jumping FK (section 3) eager, compiled default, compiled
  reduce-overhead; vs current eager matrix FK, current compiled, Warp FK. Fwd and
  fwd+bwd. Decides S1/S2: proceed if eager MHR-203 b256 fwd ≤ 1.5 ms (now 9.5) and
  fwd+bwd ≤ 6 ms (now 44), and reduce-overhead fwd ≤ 0.3 ms at b1-4096 (Warp: 0.54-0.63).
- **E2 compile feature matrix [PK check].** For the E1 function: double backward through
  compiled fn; `torch.func.jacrev`/`vmap` over it; `dynamic=True` vs bucketing recompile
  count across 10 batch sizes; cold and warm (cache) compile time; graph breaks
  (`torch._dynamo.explain`). Gate: cold compile < 60 s for FK+RNEA+Jacobian, warm < 5 s,
  0 graph breaks, ≤ 1 recompile per bucket. If double backward fails, the wrapper must
  route `create_graph=True` calls to eager (document, test).
- **E3 world-frame RNEA accuracy.** Mask-sum RNEA (section 3) f32 vs pinocchio f64 on
  random states, robot at 0 m and 50 m from origin. Gate: rel. torque error ≤ 1e-4 at
  0 m, ≤ 1e-3 at 50 m; else use local-frame level-synchronous RNEA (depth launches).
- **E4 forward dynamics.** CRBA + batched Cholesky vs level-synchronous ABA, nv 7/30/75/
  204, b1-65536. Pick the faster as `forward_dynamics`' one implementation per size
  regime only if crossover is > 3x; otherwise CRBA+Cholesky only (simpler).
- **E5 headroom (only if a use case needs b ≥ 65k).** One Triton or CUDA FK+VJP kernel
  (1-2 days, throwaway) vs best of E1. Gate for S3: kernel ≥ 5x faster fwd+bwd at
  b ≥ 65536 *and* the use case does not need double backward / shape gradients.

What would change the recommendation:
- E1 eager prototype misses its gate by > 2x → op count is not the bottleneck I think; profile before going further.
- E2 shows compile unusable (graph breaks we cannot remove, recompile storms, cold compile > 5 min) → B is dropped; then C for FK+RNEA becomes the speed path and the double-backward/shape-gradient users stay on eager A (accept two impls for 2 passes, or accept the slower speed).
- The owner's main workload becomes ≥ 1M-sample RL/sampling where 10x matters more than autograd features → C for the hot passes.

## 3. Algorithm-level plan

Shared precomputation on `ModelStructure` (built once, plain int tensors, moved with the
model; no hidden caches on a frozen object — store them as declared fields):
- `parent[j]`, `depth[j]`, `levels`: list of index tensors per depth (for level-sync passes).
- `anc_jump[k][j]`: 2^k-th ancestor (sentinel = root/self) for k < ceil(log2(depth)).
- `support pairs`: flat `(ancestor, descendant)` index pairs, size Σ depth(j) (≤ n·d);
  dense bool `support[n, n]` for n ≤ ~256 when a matmul is cheaper than scatter.
- DOF-column maps: `dof_joint[v]`, `dof_slot[v]`, `q_index` per kind group (already in
  `_fk_matrix.get_plan` and `_jacobian_columns.py:56-76`; unify into one plan).

**Joint types without per-joint Python dispatch.**
- Local transforms do not depend on the tree, so compute all of them first, one batched
  op group per *kind family* across all joints (not per level): (1) 1-DoF screw family
  (revolute, prismatic, helical, any axis, unbounded-revolute via cos/sin): one Rodrigues
  formula with per-joint `axis`, `pitch`, `is_prismatic` tables; (2) spherical (quat or
  rotvec exp; SMPL); (3) free-flyer/translation+quat; (4) planar/composite/mimic fallback.
  That is ≤ 4 groups → ~15-20 ops regardless of njoints. `_fk_matrix.py:192` already does this.
- Motion subspace in DOF-column form: `S_world[v] (6,)` per DOF, not padded 6-slot per
  joint. Padding to 6 slots wastes 6x on 1-DoF robots; columns are what the Jacobian
  pass already uses. Joint-space quantities are scatter/gather by `dof_joint`.
- Mimic joints: affine map `q_full = A q_act + b` as a precomputed sparse/gather op before
  FK (cuRobo does the same, curobo.md:156).

**FK.** Pointer jumping (parallel prefix over the parent pointer): `T[j] ← T[anc[j]] ∘ T[j];
anc[j] ← anc[anc[j]]`, ceil(log2 depth) steps, each one gather + one batched compose
over all joints. Launches: ~3 × log2(depth) → 4 steps at depth 10, 5 at depth 20, 6 for
a 64-chain. Represent poses as (R 3x3, p 3), not 4x4 (25% less traffic; cuRobo's compact
R+p, curobo.md:205). Frames: one gather+compose after. Backward: autograd is fine
(reverse of a constant number of ops). Gradients to placements/shape are automatic.

**Jacobians / J-dot.** Keep the current constant-launch column pass
(`02_results.md:32-35`); feed it from new FK. Already analytic; no change of method.

**RNEA (first choice: world-frame, constant launches).** Featherstone's world-frame
form: v_j = Σ_{i∈anc(j)} S_i q̇_i and a_j likewise (+ the S-dot terms) = segment-sum over
support pairs; body forces f_j = I_j a_j + v_j ×* I_j v_j in world frame; joint force
F_i = Σ_{j∈desc(i)} f_j = segment-sum over the same pairs reversed; τ_v = S_v·F. About
60-100 ops independent of n and depth. Risk: f32 cancellation far from the origin (E3);
mitigation: express in a frame at the base / root body, not the world origin. Fallback:
local-frame RNEA level-synchronous (2 × depth level steps of gather-compute-scatter_add).

**RNEA derivatives.** Reverse-mode VJP through the batched RNEA is O(n) and cheap → use
autograd for scalar losses and for torque-residual VJPs. Full ∂τ/∂q, ∂τ/∂q̇ (needed by
LM/GN trajopt with dynamics residuals, DDP-style solvers) → implement Carpentier & Mansard
2018 analytic derivatives as their own algorithm `rnea_derivatives`, in the same
world-frame batched form (they are products of world-frame quantities summed over
support pairs, so they vectorize the same way). Test them against `torch.func.jacrev`
of `rnea`. ∂M/∂q is not needed (∂τ/∂q̈ = M from CRBA).

**CRBA.** Composite inertia I^c_i = Σ_{desc} I_j (world frame, one segment-sum);
M[u,v] = S_u^T I^c_{max(depth)} S_v for DOF pairs in ancestor relation; build as one
batched einsum on the support mask (B·nv²·36 flops: 4096×204² ≈ 1.2e10 → ms-scale).
Do not use Σ_k J_k^T I_k J_k (B·nb·nv²·72 flops, ~100x more at 200 DoF).

**ABA / forward dynamics.** ABA's articulated-inertia pass is genuinely sequential in
depth (a 6x6 or nv_j×nv_j inverse per joint). Two candidates (E4): (i) level-synchronous
ABA: 3 passes × depth level steps; (ii) CRBA + RNEA bias + batched Cholesky solve (MJX
and mujoco_warp factor M rather than run ABA [PK]). Dense Cholesky is nv³/3 per item:
fine to nv ~ 75 at b4096, heavy at nv 204 and 1M. ABA derivatives = −M⁻¹·∂RNEA (reuse
`rnea_derivatives` + Cholesky), never autograd through ABA for full Jacobians.

**Centroidal.** CoM, Ag (centroidal map), CCRBA: one segment-sum of world-frame inertias
to the root + per-DOF column products; constant launches. Derivatives via autograd (VJP)
until a use case needs full Jacobians.

**Where analytic beats autograd** — keep autograd for VJPs (scalar losses, NN layers) and
for model-parameter gradients; use analytic for (a) kinematic Jacobians (done), (b) full
∂RNEA/∂(q, q̇), (c) ∂ABA via (b). Every analytic derivative is a separate algorithm with
a `jacrev` parity test, so there is still one implementation per algorithm.

## 4. Benchmarking and profiling design

Matrix: algorithm {FK, frames, Jacobian, J-dot, RNEA, RNEA-deriv, CRBA, ABA/FD, CCRBA}
× model {panda, ur10, G1/H1 humanoid, SMPL-24, SMPL-X-55, MHR-203, synthetic 64-chain,
synthetic 200-tree} × batch {1, 16, 256, 4096, 65536} (1M only for FK, chunked) × device
{CUDA, CPU-1-thread} × mode {fwd, fwd+bwd (VJP), double-bwd for FK/RNEA} × lane {eager,
accelerated}. Prune: CPU only batches 1/16/256; 65536 only for FK/Jac/RNEA.

Baselines (separate optional env, never a dependency): pinocchio 3 (CPU, single thread;
f64 parity and the batch-1 bar), pyroki (JAX jit FK/IK), cuRobo v2 (FK/IK, RNEA), mujoco_warp
/ MJX (kinematics, rne, factor_m), upstream smplx (LBS forward, the BetterHuman bar).
Each baseline script writes the same JSON schema.

Metrics per cell: median and p10/p90 wall ms; evaluations/s = batch/median; **kernel
launches per call** and **aten ops per call**; peak CUDA memory (`max_memory_allocated`
after `reset_peak_memory_stats`); for accelerated lane: cold compile s, warm compile s,
recompile count; CPU: ms single-thread.

Trustworthy timings:
- `torch.utils.benchmark.Timer` (handles sync and warm-up) or CUDA events around N
  calls with one sync; ≥ 10 warm-up calls; ≥ 50 timed blocks; report median.
- Pin the GPU (`CUDA_VISIBLE_DEVICES`), check it is idle (nvidia-smi) and record clocks;
  optionally lock clocks (`nvidia-smi -lgc`) on the bench box; CPU: `torch.set_num_threads(1)`,
  `taskset`. Record torch/CUDA/driver/GPU/commit/dtype in every JSON.
- Separate cold (first call, compile) from steady state; never mix.
- Verify the result (allclose vs eager) inside the bench once, so a fast wrong lane fails.

Regression tracking:
- `benchmarks/results/<machine>/<commit>.json` (one file per run, schema-versioned);
  a small compare script prints a table vs a chosen baseline commit.
- **CI gate on deterministic counters, not time**: aten op count per call (via a
  `TorchDispatchMode` counter, works on CPU CI) and, on the GPU box, kernel launches per
  call (`torch.profiler` with CUDA activity, count kernel events). Gate: any increase fails
  unless the baseline file is updated in the same commit. This catches the per-joint-loop
  class of regression exactly and without noise.
- Time gate on the bench box only: fail if median > baseline × 1.10 and the difference
  > 3 × MAD, confirmed by one re-run.
- Tools: pytest-benchmark is fine for CPU micro-benches but not GPU-aware; asv gives
  history plots but is heavy; CodSpeed measures CPU instruction counts and does not see GPU
  time [PK]. Recommend: own runner (≈200 lines) + JSON + the op/launch counters; add asv
  later only if history plots are wanted.

Profiling toolkit (one `benchmarks/profile.py`): torch.profiler trace export (Chrome/
Perfetto) with `record_function` ranges per pass; launch and aten-op counters as
context managers (reused by tests); `nsys` recipe for graph-replay timing; py-spy flame
graphs for CPU-side Python overhead; memory snapshot (`torch.cuda.memory._record_memory_history`)
for the 1M cases.

## 5. Performance targets for the plan

| Target | Value | Justification |
|---|---|---|
| Op count of every pass | independent of njoints; ≤ 3·log2(depth)+const (FK), ≤ 100 (RNEA) | cost model §0; current FK is 2/joint |
| Eager GPU FK fwd, MHR-203, b256 | ≤ 1.0 ms (now 9.5) | ~40 ops × 20 us |
| Eager GPU FK fwd+bwd, MHR-203, b256 | ≤ 4 ms (now 44) | backward ≈ 2-3x forward ops |
| Accelerated FK fwd, 30-DoF, b1-4096 | ≤ 0.1 ms | must beat Warp 0.54-0.63 ms (`warp_fk_*_b*.json`) by ≥ 5x to justify deletion without regret |
| Accelerated RNEA fwd+bwd, 30-DoF, b1 | ≤ 0.1 ms | cuRobo v2 31.6 us/iter (curobo.md:210); within 3x of a hand kernel |
| Eager RNEA fwd+bwd, 30-DoF, b4096 | ≤ 3 ms | ~100 fwd ops + ~2x backward at 20 us |
| FK throughput 30-DoF, b65536, accelerated | ≥ 2e7 evals/s | bandwidth bound ~5x off a fused kernel |
| CPU b1 FK 7-DoF eager | ≤ 0.25 ms | 02_results.md:44 has Jacobian 0.131 ms; pinocchio is ~1 us [PK]: we do not chase it at b1, pinocchio stays the parity reference |
| Accuracy | f32 ≤ 1e-4 rel vs pinocchio f64; f64 ≤ 1e-10 | E3 |
| Compile | cold ≤ 60 s, warm ≤ 5 s, 0 graph breaks | E2 |

## 6. Risks and open questions

Risks:
- torch.compile limits [PK]: double backward through compiled regions, jacrev of compiled
  functions, CUDA graphs need static shapes and no host sync (LM's "terminal" check per
  iteration in `optim` is a host sync — must stay outside the captured region).
- World-frame RNEA/CRBA in f32 loses precision far from the origin (E3).
- Deep chains (snake robots, 64+ links) keep level-sync ABA at depth launches; pointer
  jumping helps FK/RNEA only.
- Support-pair tensors grow as n·depth (200 × 20 = 4k pairs: fine); dense masks n² only
  for n ≤ 256.
- 1M-batch memory in eager: every intermediate is GBs; chunking is required in A.
- Benchmark numbers in the repo predate the matrix-FK rewrite (scan_backend §3 caveat).

Open questions for the owner:
1. Is batch ≥ 65k (RL / sampling) a real use case? It is the only regime where C pays.
2. Is double backward through FK/RNEA required in practice (use case 4), or is
   first-order enough? Decides whether a future fused kernel needs a differentiable backward.
3. Which humanoid robot model should be the "30-DoF" benchmark (G1, H1, other)?
4. Should `forward_dynamics` be CRBA+Cholesky only (simpler) unless E4 shows ABA > 3x?
5. Tree depths of SMPL/SMPL-X/MHR are unmeasured (SMPL root→hand is 9 joints by the
   standard kinematic tree [PK]); measure them in S0.
