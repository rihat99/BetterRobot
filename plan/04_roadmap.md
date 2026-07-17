# 04 — Roadmap

Revised after the codex adversarial review (`research/codex_plan_review.md`
§B6/§E): the original M2 packed several risky milestones into one, its
acceptance criteria depended on M3/M4 features, and M1 deleted surfaces
that a named consumer imports. The sequence below is built so that every
milestone leaves the library better than it found it, with the
pinocchio-parity suite green throughout.

Effort labels: **S** ≤ 1 day, **M** ≤ 1 week, **L** = multi-week.
A "gate" is a decision point that requires evidence, not code.

## Standing rules (apply to every milestone)

- **Branch strategy (owner decision 2026-07-17), replacing the earlier
  "deletions land with or after their replacement" rule:** the redesign is
  implemented on a dedicated branch; BHF/BVR keep consuming the
  pre-redesign branch until the M4 migration. Compatibility shims are
  therefore **not required** — deletions may land freely on the redesign
  branch. The M4 symbol-by-symbol migration table remains a deliverable,
  executed once when the consumers switch branches. (For that table: BHF's
  legacy surface is broader than first documented — `CostStack`,
  `Data.oMi` (including *assignments*), `GaussNewton.minimize`, the
  Huber/Cauchy kernels, `Trajectory`, trajectory residuals, and
  `ResidualState`.)
- **A misleading capability is repaired, or made to fail fast with an
  honest error, in the same milestone where the lie is discovered.** It is
  never left advertising itself.
- **Batched-vs-sequential parity uses stated tolerances and per-element
  statuses, not exact equality** — per-element branch decisions
  legitimately differ near thresholds. Performance acceptance uses
  committed benchmark definitions (hardware, dtype, shapes, warmup,
  statistics) — never bare ratios.
- **No new abstraction without a second concrete caller in-tree.** The
  variable-block API itself obeys this, via the M2a vertical slice.
- **Every Warp kernel lands with all five of:**
  1. an adjoint-strategy entry in the design table (03 §2.3);
  2. parity versus the torch lane (fp32/fp64, several batch shapes, both
     base types);
  3. gradcheck versus the torch lane — including singular points,
     branched trees, and chains longer than 16 joints;
  4. a warp-CPU parity run in CI (so it is exercised on machines without
     GPUs);
  5. a committed benchmark against the compiled torch lane.

  Warp-CPU parity alone marks a kernel a **prototype** — it exercises
  none of streams, capture, atomics, or races. A kernel is **production**
  (eligible to become a default) only after CUDA correctness validation
  on a real GPU runner, and it actually becomes the default only on
  benchmark evidence (03 §2.7).
- **GPU caveat:** CUDA is broken on this dev box, so local warp
  validation is warp-CPU (prototype-grade) only. **Securing a remote CUDA
  runner is an M1 action item, not an M6 luxury** — the bridge's most
  load-bearing rules (streams, capture, graph replay) cannot be tested
  without one.

---

## M0 — Truth and correctness (all S, immediate, any order)

The library must stop lying before it grows.

| Item | Effort | Done when |
|---|---|---|
| Fix the NaN gradients at θ=0 in `so3/se3 exp/log`, `tangents.py`, and `JointSpherical.difference` (the safe-`where` idiom) | S | gradcheck passes **at** 0 and at 1e-9, including second order; the SMPL-like rest-residual backward has zero NaNs |
| Make the Taylor cutoff dtype-dependent (fp32: ~1e-5 on θ²; fp64: current value) | S | fp32 gradcheck near the cutoff passes |
| Mimic joints: **reject at build** with a clear error (real enforcement is M3's coordinate-map work — it spans Jacobians/limits/dynamics, not just an FK gather). Scope is an owner decision: the Panda's finger mimic means reject-all refuses the flagship parity robot — decide reject-all (and adjust parity fixtures) vs rejecting only non-identity mimics | S | loading a mimic URDF raises with guidance; the false claim in `data_model/CLAUDE.md` is removed |
| Batched input to `solve_ik`/optimizers: explicit `NotImplementedError` pointing to this roadmap (the repair lands in M2b); docs stop claiming batching | S | a batched call fails with an honest message, not a shape error at `state.py:95` |
| Document the bounded-LM weakness honestly (the repair lands in M2b); report `maxiter` vs `converged` correctly | S | docstring + docs state the active-bounds weakness; no false `converged` |
| Robust-kernel IRLS: accept steps on the robustified cost (ρ), not raw L2 | S | a Huber solve converges on an outlier dataset where L2-accept stalls |
| Fix `solve_ik`'s silent float64→float32 casts | S | a dtype-preservation test |
| Delete the enum lies: the `AUTODIFF`/`FUNCTIONAL` strategy values and the `"cg"`/`"trust_region"` config options (re-add them when implemented); document FD as costing `2·nv+1` evaluations | S | every advertised option either works or cannot be selected |
| Docs truth pass: frame naming (`body_panda_hand`), the `lower_pos_limit` shape, the roadmap page regenerated from a `NotImplementedError` grep, a CLAUDE.md sweep | S | the front-page snippet executes |
| Remove the free-flyer `bool()` validation from the FK hot path (make it an opt-in debug check) | S | free-flyer `forward_kinematics_raw` compiles with `fullgraph=True` |

## M1 — Structure, the two-lane seam, and hygiene

Realistically **4–6 weeks solo** (the earlier "~2 weeks" estimate is
retired). Two items are M–L; the rest are M or smaller. Sequenced roughly
as listed.

1. **Design the two-lane seam, then delete `backends/`** (L).
   Build `ModelStructure` with the **dual representation** — frozen
   Python-side static mirrors for `torch.compile` specialization, plus
   flat device tensors for kernels, consistency-tested (03 §2.1). Build
   `ModelValues` (a tensor pytree; poses alias warp transform arrays
   zero-copy). **Freeze the execution-batch ABI** — flat `E`, per-input
   batch-index maps, in-kernel shared-value gradient reduction (03 §2.4).
   Re-sign the `*_raw` passes as pure functions over these types. Lie
   facades call the implementation directly; the `backend=` kwarg is
   dropped everywhere.
   *Done when:* the layer test is updated and has no `backends` node; the
   parity suite is green; an **eager-CPU non-regression benchmark is
   committed**; a layout test pins `transformf`/`transformd` aliasing by
   pointer and stride; the ABI tests (batched-q × unbatched-values, the
   converse, multi-axis, mismatch errors) pass.
2. **Bridge prototype before bridge promises** (M; 03 §2.3).
   One FK-shaped functional custom-op **pair** (forward + backward, with
   fake registrations) proving: fp32/fp64; gradients to q and to shared
   values with correct reduction; `torch.compile(fullgraph=True)`;
   launch on torch's current stream; and — on a CUDA runner — graph
   replay. The prototype settles custom_op-vs-autograd.Function and the
   allocation policy. The torch floor moves to ≥ 2.4.
   *Done when:* the prototype passes gradcheck + gradgradcheck (via the
   torch-lane recompute path) + compile/fake-tensor tests, and is written
   up as the normative pattern. **A CUDA runner is secured (remote is
   fine) — or this item stays open.**
3. **First Warp kernel: the FK sweep** (M–L), against the new seam.
   Newton-style mapping (one thread per batch element, serial topological
   loop, int8 kind dispatch). The adjoint strategy is chosen from the
   03 §2.3 table (stored intermediates vs hand-VJP vs static
   specialization) and recorded. Differentiable-input matrix: q, joint
   placements, frame placements — the fused pass consumes all three.
   *Done when:* parity + gradcheck versus the torch lane pass, including
   branched trees and >16-joint chains, via warp-CPU in CI. **Prototype
   status until CUDA validation** (in M1 if the runner exists, else M6).
   The GPU benchmark and the default-on decision happen in M6. Ships
   opt-in.
4. **Contract/docs migration for the inversion** (S–M).
   Update `CLAUDE.md` (the backends DAG marked "never violate"),
   `docs/concepts/batching_and_backends.md` (calls Backend objects the
   "architectural core"), `docs/conventions/performance.md` (claims
   capture destroys the autograd tape), and the stale `01_assessment.md`
   lines. Retire the layer-DAG test's `backends` node. State the
   torch-version policy in `pyproject`.
   *Done when:* no doc or contract test contradicts the two-lane design;
   a grep for `backend=` in docs/ comes back clean.
5. **Write the cross-cutting engineering contract** (M; 03 §9): dtype
   policy, quaternion double-cover convention, threading/cache ownership,
   serialization stance, differentiation contract, compile lifecycle,
   **license choice**.
   *Done when:* short written policies are checked in; a LICENSE file
   exists.
6. **Hoist per-call constants** (M): the pose `new_tensor`, `torch.eye`,
   joint axes into ModelStructure; 6×6 inertias precomputed for static
   values.
   *Done when:* zero tensor-constructor/`.to()` calls inside FK/RNEA
   loops, enforced by lint.
7. **Extend the hot-path lint** (S): `bool(`/`float(` on tensors,
   `new_tensor`, `torch.eye`; watch `residuals/` and `lie/` too.
   *Done when:* the lint catches the fixed patterns when they are
   re-introduced.
8. **De-bloat dead surfaces** (S): `utils/`, the residual
   registry, `chain.py`/`indexing.py`, `graph_capture`, the IR schema
   handshake, `rich`; park `dynamics/action/` in a branch; unfreeze the
   API contract. Under the branch strategy (standing rules), `costs/` and
   the `Data` aliases may be deleted outright — record every removed
   consumer-facing symbol for the M4 migration table.
   *Done when:* `wc -l` drops by ≥1k and the removed-symbol list is
   recorded.
9. **Fix the ModelBuilder kind mapping** (S–M): helical,
   `JointModel`-instance kinds; wire or remove `JointComposite`.
   *Done when:* every builder method produces a loadable model.
10. **Extras packaging + CI** (S–M): extras for `viewer`, `mjcf`,
    `meshes`, `demos`; CI running tests + executable docs + the committed
    benchmark definitions.
    *Done when:* a fresh install pulls only torch+numpy+yourdfpy; CI is
    green.
11. **Cache `Data` per iterate in the current solver** (S) — a stopgap
    until the M2a providers land.
    *Done when:* `solve_ik` does at most one FK per iteration plus one
    per accepted step.

## M2 — Optimization core (split in three; the big investment)

**M2a — Variables, evaluation protocol, and the vertical slice (L).**
Manifolds (`Euclidean/SO3/SE3/RobotConfig`) with state-space bounds via
feasible retraction, plus the tangent-space autograd helper (needs the M0
θ=0 fix). Draft `VarSpec`/`Values`/`Problem` with the provider DAG and
evaluation-local context (03 §3, including its seven review constraints).
Then **implement one real consumer vertical slice against the draft API
before freezing it**: two variable blocks, one shared provider (an NN
pass), one custom residual, one scalar term, masks, batching, and a phase
transition — taken from BHF object-align or BVR human_optim. Also decide
now the differentiation-contract questions that constrain state shape
(the implementation stays in M6).
*Done when:* the slice runs end-to-end on the draft API, and only then is
the API declared public; a custom-residual author guide exists and was
followed verbatim by the slice.

**M2b — Batched second-order solvers (M–L).**
LM/GN on the `init_state/update/run` pattern: per-element damping
(`mu[...,None,None]·I`), accept/reject as a 0/1 blend of
already-computed tensors, `cholesky_ex` with info-mask fallback,
per-element statuses (converged / stalled-at-bounds / maxiter / failed),
Madsen–Nielsen numerics on scaled blocks, and a real bounded algorithm
(active-set LM or reflective trust region) with KKT termination.
*Done when:* batched IK (128 targets, one call) matches 128 sequential
solves within stated tolerances, with per-element statuses;
bounds-active IK converges where the M0-documented behavior stalled; the
solver-quality tests from the optim audit's probe set pass; and `update`
satisfies the capture-readiness checklist (03 §2.6: fixed input buffers
updated via `copy_`, warmup-then-record discipline, stable addresses,
branch-free logic including the `cholesky_ex` fallback as fixed tensor
work, host syncs only at `run` boundaries) — certified by the actual
capture/replay parity test in M6, not by lint alone.

**M2c — First-order path + phases + task rebase (M).**
Matrix-free Adam on tangent gradients (batched LBFGS is its own later
item — per-element histories and line search are real design work). The
phase engine (residual/weight/mask/optimizer overrides — BVR's engine
semantics). `solve_ik` re-based as a thin preset; `solve_trajopt` and the
B-spline path fixed (quaternion-safe, bounds honored) or temporarily
dropped; `costs/` (if it survived M1) and the legacy optimizer surface
(`LeastSquaresProblem`, `GaussNewton.minimize`, solver-global kernels)
deleted once no BR-internal caller consumes them — no BHF shim needed
under the branch strategy; replacements recorded in the migration table.
*Done when:* a joint `q + camera-extrinsics` toy problem solves without
touching BR internals; BHF's ICP workarounds (trust-region clip, relative
damping, external stopping) reproduce as configuration, not subclassing.

## M3 — Parametric Model breadth (M–L)

1. Batched/parametric `ModelValues` across
   FK/Jacobians/RNEA/ABA/CRBA/centroidal, with the q-batch × value-batch
   broadcast contract (03 §5). The parity-suite extension over joint
   kinds × batch shapes × base types lands **first**, as the safety net
   (the 2026-07-17 coverage audit confirmed the gaps: FK parity is
   unbatched-only, the suite is fp64-only, no batched-values parity
   exists, and planar/translation/helical/unaligned/mimic kinds are
   uncovered).
   The execution-batch ABI was frozen in M1 (03 §2.4); M3 extends
   *breadth* under that ABI — no kernel-facing ABI change — and the M1 FK
   kernel gains value-batched coverage tests here.
2. `Model.with_values(...)` becomes public, plus the frame table
   (`(*value_batch, nframes, 7)`); an exhaustive, tested `.to()`.
3. The mimic-joint reduced coordinate map, implemented across
   FK/Jacobians/dynamics/limits (acceptance: an explicit reduced-map
   assertion, not pinocchio-default parity).
4. Order-preserving build, plus a public vectorized q-permutation.
5. Swing/twist limits and rotation-prior residuals; a differentiable
   `Inertia.from_mesh`.
6. Vectorize `integrate/difference` by joint-kind grouping.

*Done when:* a betas-parameterized SMPL skeleton (built by a better_human
prototype) runs batched FK + IK with gradients flowing to betas; BVR's
`smplx_robot/` and BHF's remap shims become deletable.

## M4 — Consumer feature packs & full migration (M each, co-developed)

- Vision residuals: projection (+ confidence weights), masked chamfer,
  and the point-SDF trio sharing one provider; per-item robust kernels
  including Geman-McClure.
- The inverse contact-force task (both repos hand-roll the same fext
  scatter + base-wrench solve).
- Small utilities with real demand: euler↔quat, 4×4↔7-vector interop,
  weighted batched Umeyama, trajectory SLERP smoothing.
- Viewer: a public playback API; implement or delete the stub overlays.
  Collision: port capsule self-collision as a residual, or cut the
  package.
- **Full consumer parity now has its prerequisites:** reproduce BVR
  `human_optim` stage 1 (phased, projection+chamfer+SDF, scalar terms)
  and BHF motion optimization on BR primitives, at parity quality and
  runtime per the committed benchmark definitions; the symbol-by-symbol
  migration table is executed; the remaining shims are deleted.

*Done when, per item:* the corresponding consumer code is deleted from
that repo and replaced by a BR call.

## M5 — Sparse trajectory structure (M–L, new milestone)

Dense per-(residual, variable) assembly is the known scalability wall for
long horizons. Design symbolic block sparsity (reviving `ResidualSpec`'s
banded/temporal information from the design notes), JVP/VJP linear
operators, banded/block-tridiagonal solvers, and optional Schur
elimination for camera/nuisance blocks.
*Done when:* trajectory IK at T=500 on the SMPL-scale model runs with
memory and time sublinear in the dense-assembly baseline; the benchmark
is committed.

## M6 — Warp fast path & CUDA graphs (L, requires a real GPU box)

This is no longer a "should we do Warp" gate — that decision is made
(03 §2). What stays evidence-driven is each kernel's *default-on* switch,
per the standing kernel rule. On a real GPU box:

1. **Baseline first.** Profile FK/RNEA batched sweeps and end-to-end
   solver iterations (SMPL-scale and Panda-scale), before and after
   `torch.compile`. Publish committed benchmark results including
   cold-start costs (torch compile ~31 s, measured on the CPU probe; warp
   codegen and graph-record costs measured here). This baseline is the
   bar every kernel must clear.
2. **Kernel build-out**, in boundary-table order (03 §2.2).
   Prerequisite: port BR's Taylor-stitched SO3/SE3
   log/exp/right-Jacobian math as a small `wp.func` library with
   validated adjoints — warp has **no** Lie log/exp builtins. Then: the
   M1 FK kernel gets CUDA validation (if it didn't already), its GPU
   benchmark, and a default-on decision; Jacobians (fused with FK where
   profitable); RNEA, ABA, CRBA/centroidal — each with its own adjoint
   strategy, since they carry more loop state than FK; residual kernels
   with gradients-in-forward (pose first — it erases the FD/jacrev
   Jacobian cost); collision warp-led with its torch reference (capsule
   self-collision first, feeding the M4 residual).
   `integrate`/`difference` only if a kernel beats the §7 vectorized
   torch rewrite on the committed benchmark. Priority inside the
   milestone: FK/Jacobian + one residual + collision **before** any
   dynamics kernel — consumer IK throughput is the demand signal.
3. **CUDA-graph capture of solver inner loops** (cuRobo's GraphExecutor
   pattern: record `inner_iters × update` including the autograd
   backward; warmup-then-record; re-record on resize). Requires the M2b
   capture-readiness acceptance; proven by a capture/replay-vs-eager
   parity test. `wp.capture_while` stays out of scope until mixed
   torch/warp external capture is demonstrated (03 §2.6).
4. Benchmark frax-style ancestor-mask RNEA/CRBA as a torch-lane variant
   against the warp kernel at 25–160 DOF. Benchmark serial-generic
   versus static-specialized (and, if warranted, branch-parallel) warp
   mappings — B=1 latency is a known weak spot of
   one-thread-per-element kernels.
5. Implicit-diff `solve()` per the contract decided in M2a (torch lane).
6. **External reference points:** batched-IK throughput versus cuRobo on
   a comparable Panda-class problem; FK/dynamics sweeps versus a
   JAX-class library (mjx/pyroki) where an apples-to-apples definition
   exists. The benchmark definition is written **now**, not at
   measurement time: hardware, batch sizes (including B=1), robot,
   targets/seeds, fixed iteration budget, success tolerance, collision
   on/off, precision, warm/cold timing, memory. **A gap versus cuRobo is
   expected until measured** — BR deliberately gives up per-robot NVRTC
   specialization and hand-written CUDA. If the generic kernels miss the
   target, the sanctioned escape is warp static specialization (03
   §2.1), never a second kernel language. Committed numbers, stated
   caveats — no bare ratios.

*Done when:* per-kernel committed benchmarks exist; every default-on
kernel beats the compiled torch lane by its stated margin; capture/replay
matches eager within stated tolerances and demonstrates the end-to-end
win on batched IK; and the torch-lane parity suite stays green throughout
(no CPU regression — the eager floor included).

## Success metric

Every deleted line of consumer workaround code, tracked per milestone.
The assessment found waiting to become BR calls: ~24 hand-rolled
optimizer call sites, two ~200-line q-remap shims, a 188-line private
optimization engine, a 564-line loss library, and three hand-rolled
inertia loops.
