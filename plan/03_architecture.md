# 03 — Target Architecture & Redesign Decisions

Each section states a decision, the reasoning behind it (with pointers
into `research/` and `../references/design/`), and an API sketch where the
shape matters. Sketches show direction, not final signatures.

A few terms used throughout this document:

- **Kernel** — a function compiled to run on the GPU across many threads
  at once. In warp-lang you write it in Python-like code and Warp compiles
  it to CUDA (or to CPU code).
- **Whole-pass function** — one function that performs a complete
  algorithm (the full FK sweep, a full RNEA), as opposed to one tiny op
  (a single quaternion multiply).
- **Adjoint** — Warp's name for the automatically generated backward
  (gradient) version of a kernel.
- **VJP** — vector–Jacobian product: the quantity a backward pass
  computes.
- **Oracle** — the reference implementation we trust. Every kernel is
  tested to match it, in values and in gradients.
- **Zero-copy** — two libraries reading the same memory without copying
  anything. Torch tensors and warp arrays can alias each other this way.
- **Host sync** — the CPU stopping to wait for the GPU (usually to read
  one value back). It stalls the GPU pipeline and breaks both
  `torch.compile` and CUDA-graph capture.
- **CUDA graph capture** — recording a whole sequence of GPU operations
  once, then replaying that recording with almost zero CPU overhead per
  step. One third of cuRobo's speed story.
- **gradcheck** — PyTorch's numerical verification that gradients are
  correct; `gradgradcheck` does the same for second-order gradients
  (gradients of gradients).

---

## 1. Layer diagram (target)

```
lie → spatial → data_model → (kinematics, dynamics) → residuals → optim → tasks → viewer
io → data_model            collision ∥ kinematics
```

Changes from today: `backends/` is gone; `costs/` merges into `optim/`
(CostStack is one small file, not a layer); everything else keeps its
place. The layer-DAG contract test stays.

One thing to be clear about: the Warp lane (§2) is **not a layer**.
Kernels live right beside their torch counterparts, inside `kinematics/`,
`dynamics/`, `collision/`, and `residuals/` (for example
`kinematics/_warp_kernels.py`). The layer DAG binds both lanes equally.

## 2. Delete `backends/`; the seam is the whole-pass function — and Warp is its second implementation

**Decision (revised 2026-07-16, owner directive).** The earlier draft
prepared clean seams but postponed Warp to an evidence gate at the end of
the roadmap. That is reversed. Since the architecture is being redesigned
anyway, the compute layer is designed around **Warp now** — one rewrite
instead of two.

What does *not* change: `backends/` (~530 lines) still gets deleted — the
three Protocols, the registry, `set_backend`, `graph_capture`, and the
per-op warp stubs. `lie/se3.py` / `so3.py` call the torch implementation
directly (renamed to `lie/_impl.py`). The `backend=` keyword disappears
from every signature. Going Warp-first does **not** bring back per-op
dispatch: a Protocol full of quaternion multiplies is exactly the
granularity Warp cannot use. The PyposeWarp digest is the cautionary tale:
24 Lie ops in Warp took 13.3k lines of code with hand-written backward
kernels, and published zero benchmark numbers.

**Why the deletion still stands:** the evidence is in
`01_assessment.md §2.1` — one implementation, inverted imports, zero to
two consumers per Protocol, and a facade that already half-bypasses the
whole thing. The reason is structural, not performance (dispatch overhead
is ~1–2 µs per op; the earlier 19 µs figure was refuted).

**Evidence base for the Warp design** — three audits, all file:line-cited:
`research/audit_curobo_warp_integration.md` (how cuRobo actually mixes
CUDA, Warp, and torch), `research/audit_warp_platform.md` (what warp-lang
can and cannot do; the kernel patterns in mujoco_warp and newton), and
`research/audit_compute_pass_inventory.md` (all 53 compute passes in BR
and their loop structure).

**The cuRobo correction, stated up front.** The folklore says "cuRobo is
fast because of Warp". The audit shows that is false. cuRobo's performance
core is ~11.5k lines of hand-written CUDA C++ (fused FK+spheres+Jacobian,
RNEA, an L-BFGS two-loop, line search, B-splines), compiled per robot at
run time through NVRTC — NVIDIA's runtime compiler, which lets cuRobo bake
each specific robot's dimensions and loop bounds directly into the kernel.
Warp-lang is only cuRobo's extensibility/leaf layer (cost kernels, scene
SDF, pose ops, one tile-based LM solve). And a third of the speed story is
CUDA-graph capture of *plain PyTorch* optimizer loops.

Two pillars transfer to BR: **fused whole-pass kernels** and
**graph-captured solver loops**. On kernel language, BR deliberately
diverges: **warp-lang only, no hand-written CUDA**. cuRobo pays 11.5k
lines of `.cuh` plus ~6.3k lines of dual-path launch plumbing — a
maintenance load this project cannot carry. Whatever speed margin
cuRobo's robot-specialized NVRTC kernels keep over our warp kernels gets
measured in committed benchmarks, and closed with fusion and graph
capture — not with a second kernel language.

### 2.1 The two-lane compute model

**In plain words, before the spec.** Every heavy computation (FK,
dynamics, residuals) is one function with up to two implementations
inside: a PyTorch version that always exists and works everywhere, and —
for the passes where it pays off — one Warp kernel that does the whole
algorithm in a single GPU launch. A small `if` at the call site picks the
lane; the caller never sees a difference. Why two: PyTorch is great at
"one big batched op" but bad at serial tree scans (a Python loop
launching many tiny ops — that is why FK costs the same at B=1 as at
B=256 today), while a warp kernel walks the whole tree inside one launch,
one thread per robot. And the torch lane is not just a fallback — it is
the referee every kernel is tested against, in values and gradients.

This pattern is not exotic. PyTorch itself does exactly this per op
(every `matmul` has a CPU and a CUDA implementation behind one name);
PyTorch3D and most custom-op libraries ship CPU + CUDA twins of each op.
cuRobo is the closest relative at pass granularity, but it is CUDA-only
and often has no torch twin to test against; mujoco_warp/newton went pure
warp and gave up torch interop, CPU speed, and (in mujoco_warp's case)
gradients entirely; JAX libraries write one implementation and bet on the
XLA compiler instead. BR takes cuRobo's architecture and fixes its two
gaps: warp-lang instead of hand CUDA, and a mandatory torch lane.

The trade in one sentence: **PyTorch is the truth, Warp is the speed** —
and the price is writing and continuously cross-checking two copies of
the hottest few algorithms (the detailed cost paragraph closes this
section; the drawbacks are itemized honestly in §2.3's gradient-trap
rules and §2.7).

Now the spec. Every performance-critical algorithm becomes a pure whole-pass function
`(ModelStructure, ModelValues, q, …) → tensors` (§5). Each hot pass has up
to two implementations behind **one signature and one memory-layout
contract**:

**The torch lane** (mandatory; always lands first). Batched, vectorized
PyTorch. It plays four roles:

- the semantic **oracle** — every kernel is parity- and gradcheck-tested
  against it, including at singular points;
- the **CPU path** (see §2.5 — eager torch is the supported floor,
  `torch.compile` is the opt-in accelerator);
- the only **second-order** path — warp cannot differentiate its own
  backward passes, so double backward always goes through torch;
- the fallback for everything else.

One caveat from codex round 2, called the **static-topology caveat**: the
measured 5× `torch.compile` win came from the robot's topology being
plain Python tuples, which compile unrolls statically. A version that
walks the tree with tensor lookups (dynamic parent gathers, tensor kind
dispatch) is a materially different algorithm with no measured number
behind it. `ModelStructure` therefore keeps a **dual representation**:
frozen Python-side static mirrors (tuples and ints — what the torch lane
specializes on) plus flat device tensors derived from them at build time
(what kernels index), with a consistency test between the two. We drop
the dual form only if an M1 prototype shows a tensor-only torch recursion
that compiles cleanly *and* does not regress.

**The Warp lane** (the GPU fast path). One warp-lang kernel (or fused
kernel group) per pass. The thread mapping follows newton, not
mujoco_warp: **one thread per batch element**, with a serial topological
joint loop inside the kernel, and joint-kind dispatch as `if kind ==`
chains over int8 codes. That mapping fits BR's workload — few robots,
huge batches.

Two honesty notes from codex round 2:

- (a) A serial loop whose length is only known at run time is **not
  automatically adjoint-safe**. Warp's generated adjoints do not replay
  the local variables of dynamic loops, which produces silently wrong
  gradients. Newton's differentiability evidence covers only a small,
  q-only FK case. So every kernel picks an explicit adjoint strategy —
  see the table in §2.3.
- (b) "Shape-generic, never recompiles" is the default. But per-robot
  **static specialization** — fixing the loop bounds at code generation
  time, so the loop unrolls and the adjoint becomes replay-safe — stays
  available as an evidence-gated variant wherever the generic kernel
  misses its benchmark. Still warp-lang, not a second language.

**How a lane is selected:** a plain `if` at each pass call site, keyed on
device, dtype, layout support, and warp availability. Unsupported layouts
(for example, views that are non-contiguous in the inner stride) **fall
back to the torch lane — they never raise an error**. Turning a warp
kernel on as a default must not change public behavior. No registry, no
Protocol; this is the mujoco_warp/newton pattern.

**The cost, stated openly.** Two lanes means two maintained
implementations of every kernelized pass, kept semantically identical —
in values *and* in gradients — by differential tests, indefinitely.
cuRobo gives us the calibration: ~5.7k lines of core-robotics warp
kernels plus ~1.6k lines of wrappers, at roughly a 1:1 test-to-code
ratio, with about half of its kinematics-kernel code being backward
machinery. That budget is why the boundary table in §2.2 is short, why
kernels land one at a time against the torch oracle, and why several rows
stay torch outright.

### 2.2 Boundary table — where Warp runs, where torch stays

| Computation | Lane | Why (evidence) |
|---|---|---|
| FK sweep + frame placements | **Warp** (one fused kernel) | A serial tree scan, bound by its Python loop today (~4 ms fixed overhead; B=1 costs the same as B=256). This is the first kernel — it proves the seam works (M1). |
| Joint / frame Jacobians | **Warp**, fused with FK where profitable | Same tree structure; cuRobo fuses FK+spheres+Jacobian into one kernel. |
| RNEA / ABA / CRBA / centroidal | **Warp** | Tree scans. ABA's tiny per-joint matrix inverses and CRBA's O(depth²) ancestor walks are hostile to torch's vectorization. |
| `integrate` / `difference` (manifold segment ops) | torch first (vectorized by joint kind, §7 step 4); **Warp only on benchmark evidence** | Once joints are grouped by kind this is no longer a serial tree scan. The vectorized torch rewrite is already planned — a kernel must beat *that*, not today's Python loop (codex round 2). |
| Collision (capsule now; SDF/point-cloud later) | **Warp-led, with a torch reference** | Today it is 100 % stubs; warp has native geometry types; cuRobo does the same. But CPU capability (§2.5) means a torch implementation ships too — "green-field" describes the design freedom, not permission to skip the oracle. |
| Built-in residual eval + analytic Jacobians (pose/position/orientation; later projection, point-cloud) | **Warp**, with gradients computed inside the forward kernel | The cuRobo pattern: backward becomes one torch multiply. Erases today's FD/jacrev Jacobian cost. ABI: the kernel returns the **raw** residual `r` and a canonical contiguous `(E, dim, nv)` Jacobian — robust-kernel weighting and IRLS stay on the torch side, so nothing is double-weighted and robust row-grouping stays visible. (Today's `residuals/pose.py:65,102` pre-weights both `r` and `J` — the M2 residual redesign moves weighting to the torch side to match this ABI.) |
| Dynamics derivatives (`compute_*_derivatives`) | torch lane (autograd) | This is derivative-of-a-pass work, and warp has no double backward. Analytic derivatives (Carpentier–Mansard) stay a roadmap item. |
| The Lie per-op facade (`se3.compose`, …) | torch | Already batched and loop-free; per-op kernels are the PyposeWarp anti-pattern. *Inside* kernels, warp's transform/quaternion builtins cover compose and rotate — but warp has **no SO3/SE3 log/exp**, so BR ports its Taylor-stitched log/exp/right-Jacobian math once, as a small `wp.func` library with validated adjoints. That port is an explicit M6 task, not an assumption. |
| Robust kernels, weights, cost aggregation | torch | Embarrassingly parallel elementwise math; nothing to win. Graph capture erases the launch overhead anyway. |
| JᵀJ assembly, Cholesky/lstsq, damping | torch | One batched op on `(B, nv, nv)` handled by cuSOLVER; and `wp.tile_cholesky` has **no adjoint**. |
| Optimizer control flow (LM accept/reject, Adam, phases) | torch, branch-free and capture-safe | cuRobo runs MPPI/ES *entirely* in torch — it is fast because the loop is graph-captured. |
| Custom user residuals | torch autograd | Extensibility is a promise made on the torch surface. |
| io / builder / viewer | python/torch | Not hot. |

**The boundary principle (evidence over slogan):** Warp kernels go where
the computation is *serial along the tree* or *fusion-heavy* — the places
where torch's per-op dispatch and materialized intermediate tensors are
structurally wasteful. (The inventory audit shows every tree-scan pass in
BR is a Python loop today.) Work that is already one batched tensor op
stays in torch: moving it buys nothing, and CUDA-graph capture (§2.6)
removes the launch-overhead argument. The owner's goal — "the whole
computational tree on warp" — is achieved at **pass** granularity, never
per op.

### 2.3 The bridge — one wrapper *pattern*, prototyped before promised

The bridge is the piece that makes a warp kernel look to PyTorch like a
normal differentiable operation. Warp ships zero-copy interop with torch,
but no ready-made wrapper for this.

The first draft promised "one ~40-line helper with caller-owned
pre-allocated output buffers". Codex round 2 refuted that as written: an
op that writes into caller-provided outputs is not *functional* (it
mutates its arguments instead of returning fresh outputs), and torch
**rejects** `register_autograd` for non-functional custom ops — codex
reproduced the rejection against torch's `_library/custom_ops.py`. Warp's
own documented pattern is a **pair of functional custom ops** — one
forward, one backward, each allocating its own outputs, each with a fake
registration (a shape-only stub that lets `torch.compile` trace the op
without running it). Schemas and shape functions are written per pass,
not once.

So the bridge is a *pattern plus shared utilities*, and M1's first
deliverable is a working prototype: one FK-shaped op pair proving
fp32/fp64 support, gradients to q and to shared model values (with
correct cross-batch reduction), `torch.compile(fullgraph=True)` with fake
tensors, execution on torch's current stream, and — on a CUDA runner —
graph replay. Two design choices are settled by the prototype, not by
prose: (1) functional ops allocating from torch's graph-pool-aware
allocator, versus a functional outer op wrapping private mutating launch
ops; (2) `torch.library.custom_op` versus plain `autograd.Function`
(cuRobo's choice) where compile-safety is not needed.

Rules that survive from the failure-mode list:

- Always `wp.from_torch(..., requires_grad=False)`, with gradient buffers
  managed on the torch side. If warp sees `requires_grad=True`, torch's
  deferred `.grad` allocation inside warp launches triggers device-wide
  syncs — a 4.3× slowdown documented in warp's own docs.
- Always launch on torch's current stream, via `wp.stream_from_torch`.
  This is the rule that makes warp launches capturable inside a CUDA
  graph next to torch ops (cuRobo does this everywhere).
- Layout preconditions are checked up front; unsupported layouts **fall
  back to the torch lane** (§2.1). Inside an active graph capture, the
  fallback becomes a hard error instead — a silent `.contiguous()` copy
  would corrupt the capture (cuRobo documents the identical rule).
- Torch floor: the custom-op route requires **torch ≥ 2.4**; `pyproject`'s
  current `>=2.1` moves up with it (version policy lives in §9).

**Backward policy — an explicit adjoint strategy per kernel, recorded in
a design table. There is no safe default.** The options:

- (a) Warp's generated adjoint, with **named stored intermediates**. Warp
  does not replay the local variables of dynamic loops, so anything the
  backward pass needs must be explicitly written out to arrays during the
  forward pass.
- (b) A hand-written VJP kernel — a reverse sweep over the tree at run
  time. cuRobo hand-writes these, and about half of its
  kinematics-kernel code is backward machinery. That is the honest price
  tag of this option.
- (c) Static specialization (§2.1), where (a) and (b) both lose.

RNEA/ABA/CRBA carry far more loop-carried state than FK, so FK evidence
does not generalize — each pass re-decides.

For built-in residuals: analytic gradients are computed **inside the
forward kernel**, and backward is one torch multiply over the raw
`(E, dim, nv)` Jacobian (the ABI in §2.2). Robust weighting stays on the
torch side.

Never span multiple launches with one `wp.Tape` without an aliasing
review. Warp's silent-wrong-gradient traps are documented: dynamic loops,
in-place `*=`, re-assigning vector components, data-dependent
`atomic_add`. The kernel-author checklist in §9 exists because
mujoco_warp chose `enable_backward=False` globally rather than fight
these traps. BR's differentiability is enforced per kernel by gradcheck
from day one — at singular points, on **branched trees and on chains
longer than warp's unroll threshold (>16 joints)**, and for shared-value
gradient reductions.

**The differentiation contract at this seam.** Each pass gets a
**differentiable-input matrix**: every floating-point input the pass
actually *consumes* is guaranteed a first-order gradient through both
lanes. For FK that means q, joint placements, and frame placements; RNEA
adds inertias; and so on. The earlier blanket slogan "every ModelValues
leaf gets gradients" is retired — an FK pass owes no inertia gradient.

Second order: `fk(q)` cannot know at call time whether
`create_graph=True` is coming later, so routing second-order work is
**not** automatic. Instead, the registered backward detects
grad-enabled backward (`torch.is_grad_enabled()` inside the backward) and
recomputes the VJP differentiably via the torch lane, from saved inputs.
Warp adjoints serve the plain first-order path. `gradgradcheck` runs
through the **public API**, not only against the torch lane directly.

### 2.4 Layout contract (co-designed with §5)

- `ModelStructure` holds int8 joint-kind codes, int32 parent/index
  tables, CSR topology caches (compressed sparse row — a flat-array way
  to store the tree), and packed joint axes — all as flat device tensors
  that **both lanes index**. cuRobo's CudaRobotModel independently
  validates exactly this layout.
- SE3 poses `[tx,ty,tz,qx,qy,qz,qw]` alias warp transform arrays
  **bit-for-bit, zero-copy** — warp's transform layout and scalar-last
  quaternion happen to match BR's convention exactly (verified in warp's
  source). Lucky, and load-bearing: a layout test pins the aliasing by
  **pointer and stride equality**, not just by comparing values.
  Precision dispatch is explicit: `wp.transformf` for fp32,
  `wp.transformd` for fp64 (`wp.transform` is just an fp32 alias). The
  aliasing precondition is warp's actual rule: the trailing value-type
  dimension must be contiguous; outer dimensions may be strided.
  Violations fall back to the torch lane (§2.1).
- Spatial vectors: BR stores `[linear, angular]`; `wp.spatial_vector` is
  **angular-first**. Policy: BR kernels keep BR's ordering using two
  `wp.vec3`s (or plain arrays); warp's spatial builtins are used only
  behind an explicit, tested conversion. Never reinterpret raw memory
  across this difference.
- **The execution-batch ABI (frozen in M1, not M3 — codex round 2).**
  Public APIs accept any batch shape `(B..., ·)`; warp kernels do not.
  Warp arrays are limited to 4 dimensions, and a tensor that was
  broadcast-expanded (stride 0) cannot alias into warp zero-copy. So at
  the seam, every batch is flattened once into a single flat **execution
  batch** of size `E` (the broadcast of the q batch with every value
  batch). Kernels see `E` plus a per-input **batch-index map** for value
  arrays that were *not* expanded — a shared (unbatched) value is passed
  once and *indexed*, never physically repeated E times. The reverse rule
  is part of the ABI too: the gradient of a value shared across `E` is
  **reduced inside the kernel path** (atomically or by segments) — it is
  never returned expanded for torch to re-reduce. This contract is tested
  for batched-q × unbatched-values, the converse, multi-axis batches, and
  mismatch errors, before the FK kernel lands. M3 extends the ABI's
  *breadth* across passes; it never changes the ABI itself.

### 2.5 CPU policy — capability preserved, honestly

- The torch lane **is** the CPU path. Eager torch is the supported
  performance floor, with a committed non-regression benchmark across the
  M1 seam change — the dual static representation in §2.1 exists
  precisely so this holds. `torch.compile` is the opt-in accelerator
  (`model.compile()` or a documented recipe): its ~31 s cold start must
  never happen implicitly. Nothing in the public API ever requires a GPU
  or an installed warp.
- Warp-CPU (warp's embedded Clang code generation — no system toolchain
  needed) compiles everything, including adjoints, but executes each
  launch as a **single-threaded serial loop**. It is a correctness and
  parity vehicle — it gives kernels CI coverage on machines without GPUs,
  including this one — never a performance claim. CI runs the warp extra
  as a dedicated job with a persistent kernel cache and small, bounded
  parity cases, so the serial execution stays affordable.

### 2.6 CUDA-graph capture — planned from day one, landed late

The optimizer redesign (§4) is written **capture-ready from the start**.

A precision from codex round 2: "no allocations during capture" is the
wrong criterion (allocating during capture is legal under graph memory
pools), and a lint cannot *certify* capture safety. What M2 actually owes
is the structural checklist cuRobo follows in practice:

- fixed input buffers, updated with `copy_` rather than replaced;
- warmup iterations before recording;
- tensor addresses stable across replays;
- branch-free logic (`torch.where`, not Python `if`);
- host syncs only at outer boundaries;
- the factorization fallback expressed as fixed tensor work
  (`cholesky_ex` info-mask blending — no dynamic element selection);
- stated eligibility rules for custom residuals.

The capture feature itself follows cuRobo's GraphExecutor pattern: record
`inner_iters × update` — *including the autograd backward* — replay until
shapes change, re-record on resize. It lands with the M6 GPU milestone
and is proven by an actual capture/replay-versus-eager parity test, never
by the checklist alone. `wp.capture_while` stays out of scope until mixed
torch/warp *external* capture is demonstrated — warp only recognizes
captures it started itself. Newton captures forward+backward together the
same way.

### 2.7 Anti-goals

- **No per-op kernels** — PyposeWarp's 13.3k-line lesson stands.
- **No hand-written CUDA C++** — warp-lang is the single kernel language.
- **No Protocol/registry resurrection** — per-pass `if`s only.
- **No CPU regression** — every feature works on CPU via the torch lane.
- **No unverifiable speed claims** — a kernel becomes the default for its
  (device, dtype) combination only when it beats the compiled torch lane
  on a committed benchmark; until then it is opt-in. Building the Warp
  lane now is the committed direction; *defaulting* each kernel stays
  evidence-based, per pass.

**Honesty about the seam (codex correction, still binding):** today's
`*_raw` functions are not yet true pass boundaries. They receive a Python
`Model`, they loop over Python `JointModel` objects, and RNEA mutates
`Data`. So the `backends/` deletion lands **together with** the seam
design: `ModelStructure` + `ModelValues` (§5) as the boundary, the bridge
contract above, and the first real Warp kernel (FK, in M1) as proof that
the boundary is actually consumable by a kernel. Semantic
interchangeability — custom backward covering q *and* model values, the
dtype/stream policy, compile and fake-tensor registration,
unsupported-joint fallback, result parity — is priced into M1, not
hand-waved.

## 3. Variable blocks — the core redesign

**Decision:** replace the single flat `x` with **named variable blocks**.
Each block carries its own manifold, bounds, and optional mask, and
residuals declare which blocks they read. This is the redesign that
unblocks the consumers — their 24 hand-rolled `torch.optim` loops are all
multi-block problems — and better_human (betas become just another
block). Design sources: pyroki/jaxls `Var`, Ceres parameter blocks,
jaxlie's `manifold.rplus` over pytrees, and BVR's `tools/optim.py` (the
de facto requirements spec: lazy shared state, per-phase weights, per-DOF
masks).

```python
@dataclass(frozen=True)
class VarSpec:
    name: str
    shape: tuple[int, ...]          # event shape; batch axes are separate and explicit
    manifold: Manifold = Euclidean()  # Euclidean | SO3 | SE3 | RobotConfig(model)
    bounds: Bounds | None = None    # STATE-space feasibility (nq-shaped for RobotConfig),
                                    # enforced by feasible retraction/projection —
                                    # distinct from any per-step trust-region bound
    scale: Tensor | None = None     # per-coordinate scaling for damping/preconditioning
    mask: Tensor | None = None      # frozen DOFs (per-phase overridable; fixed
                                    # coordinates are ELIMINATED from the system,
                                    # not zero-columned into a singular one)

# A Manifold owns: retract(x, dv), difference(x0, x1), tangent_dim(shape).
# RobotConfig(model) wraps model.integrate/difference — nq != nv handled here.

Values = dict[str, Tensor]          # name -> (B..., *shape)

class Problem:
    vars: tuple[VarSpec, ...]
    residuals: tuple[ResidualItem, ...]   # residual + weight + kernel + name

    def residual(self, values: Values) -> Tensor          # (B..., dim_total)
    def gradient(self, values: Values) -> Values          # tangent-space, per block
    def jacobian_blocks(self, values: Values) -> dict[tuple[str, str], Tensor]
        # (residual_name, var_name) -> (B..., dim_i, tangent_dim_j); absent = zero
```

Key semantics:

- **Residuals declare their dependencies.** Each residual carries
  `reads: tuple[str, ...]`. The Gauss-Newton system is assembled from
  per-(residual, variable) Jacobian blocks; blocks a residual doesn't
  read are structurally zero. For IK-sized problems, assemble densely.
  The block structure is what later enables sparse trajectory solvers
  without another redesign.
- **Derived quantities, with caching.** FK is not special-cased into the
  problem; it is a derived node. A `RobotStateProvider(model, var="q")`
  computes `Data` once per evaluation, and every kinematic residual reads
  from it. This kills the measured waste of 11 FK calls in a 5-iteration
  solve, and replaces today's `state_factory`. better_human plugs in
  here: its provider maps `betas` → model value overrides (§5) → FK.
- **Tangent-space autograd everywhere.** First-order methods
  differentiate `f(values ⊕ δ)` at δ = 0 (jaxlie's `manifold.grad`
  pattern). Adam and LBFGS then get correct manifold gradients with zero
  optimizer changes, and quaternions never see a Euclidean step. This
  requires the θ=0 gradient fix first.
- **Compatibility.** `solve_ik` becomes a thin preset that builds
  `Problem(vars=(q,), residuals=pose+limits+rest)`. The current
  `LeastSquaresProblem` dies. `CostStack` becomes the `residuals` tuple
  with weights/kernels/activity — its snapshot/restore semantics are kept
  for staging.

**Design constraints from the adversarial review.** Codex's verdict on
the first draft was "unsound as specified"; these seven changes are what
make it sound (the full argument is `research/codex_plan_review.md §B2`):

1. **Two kinds of bounds, never conflated.** State feasibility lives in
   configuration space (nq-shaped joint limits) and is enforced by
   feasible retraction. Step bounds (trust regions) live in tangent space
   and belong to the solver. SO3/SE3 blocks have no meaningful global
   tangent-space box at all.
2. **Caching is evaluation-local only.** No persistent provider cache
   keyed on a mutable "values version": tensors mutate invisibly (today's
   `Data` documents that very problem), and cached autograd graphs leak
   memory or break second backward passes. Providers compute into an
   immutable per-evaluation context — one objective/Jacobian evaluation
   shares its FK; nothing outlives the iteration except detached
   accepted-state artifacts.
3. **Providers form a declared DAG.** `reads` covers variables; providers
   additionally declare their inputs and outputs, so shared work — one
   neural-network pass feeding three point-cloud residuals, the BHF
   pattern — is expressed once, not recomputed. Custom-residual authors
   get a written contract: residual shape, semantic grouping for robust
   kernels, optional analytic blocks, provider requests, per-element
   failure signaling.
4. **Scalar objective terms are in scope.** BVR optimizes scalar/reduced
   terms with auxiliary outputs alongside least-squares terms. Either an
   explicit `ObjectiveTerm` protocol participates in first-order solves
   (and is rejected by GN/LM with a clear error), or the docs state the
   least-squares fence and the consumer keeps those terms outside. That
   choice is decided in the vertical slice, not assumed.
5. **Sparsity is a stated non-goal of v1 assembly, with the door held
   open.** Per-(residual, variable) blocks give block *structure*, not
   sparsity — a trajectory is still one big dense q-block. The
   banded/temporal structure information in today's `ResidualSpec` is
   preserved in design notes (not as shipped dead code) for the dedicated
   sparse-trajectory milestone (see 04, M5).
6. **The AD strategy is chosen per block dimension, not assumed.** jacrev
   versus jacfwd versus VJP/JVP versus analytic, decided per
   (residual dim × tangent dim). Graph lifetime and double-backward
   (`create_graph`) support are specified and tested — remember that the
   current code is not even jacfwd-clean (dtype mismatch, assessment
   §1.5).
7. **The API freezes only after a vertical slice.** One real consumer
   problem — two variable blocks, one shared provider, one custom
   residual, one scalar term, masks, batching, a phase transition — is
   implemented against the draft API before that API is declared public.
   This is the plan's own two-caller rule, applied to its own
   centerpiece.

## 4. Optimizer stack

**Decision:** rebuild the optimizers on the jaxopt/optax split, batched
from day one.

- **The state pattern (from jaxopt).** Each solver is a set of frozen
  hyperparameters plus `init_state(values) -> State` plus
  `update(values, state) -> (values, state)` plus a generic `run` loop.
  State is a plain value. That means consumers can embed a single step
  inside their own loops, warm-start across video frames, checkpoint, and
  later compile. This directly answers the top adoption request: "let
  projects keep their loop; BR owns the step".
- **Batched semantics.** Damping is `(B,)`, cost is `(B,)`, accept/reject
  is a per-element 0/1 blend of tensors that are *already computed*, and
  convergence is a per-element mask: `state.converged: (B,) bool`. A
  precision from the codex review: "branchless" does **not** mean
  evaluating two residual branches. The current residual is cached, one
  candidate is evaluated, and `torch.where` blends them; the candidate
  *Jacobian* is deferred to the next iteration, so rejected elements
  never pay for it. Each batch element gets its own normal system with
  `mu[..., None, None] * I` — sharing one factorization across elements
  would simply be wrong. Use `cholesky_ex` and its info mask for
  per-element fallback and status, so one indefinite element cannot fail
  the whole batch. Converged elements keep riding along in the batched
  ops (compacting them away is a later optimization, not a correctness
  issue). Host syncs happen only at user-visible boundaries: `run` may
  sync once per iteration for early exit; `update` never syncs. This same
  discipline is exactly what makes `update` CUDA-graph-capturable (§2.6).
  Capture-safety — no host syncs, no allocations, no shape changes inside
  `update` — is an M2 acceptance criterion, not a retrofit; cuRobo
  captures whole inner loops *including autograd backward* precisely
  because its step logic obeys these rules.
- **LM numerics (Madsen–Nielsen, ported from the jaxopt digest).**
  Damping initialized from `max(diag(JTJ))`; on accept,
  `μ *= max(1/3, 1-(2ρ-1)³)`; geometric escalation on reject. The gain
  ratio is computed on the **robustified** cost — this fixes the verified
  IRLS inconsistency where steps were accepted on raw L2 while the
  problem being solved was the robust one.
- **Bounds.** Pick an actual bounded least-squares algorithm — active-set
  LM or a reflective trust region — with feasible retraction, active-set
  updates, predicted reduction consistent with the projection, and
  KKT-based termination. (Codex's correction to the earlier draft: the
  current code already projects the trial point *before* evaluating it;
  the real failure is the total absence of active-set/KKT treatment, and
  merely adding a projected-gradient stopping criterion would keep the
  same bad steps under a better label.) Terminal statuses are
  per-element: converged / stalled-at-bounds / maxiter / failed.
- **First-order methods: matrix-free only.** The gradient comes from
  tangent-space autograd or `apply_jac_transpose`; J is never
  materialized (today's Adam/LBFGS do materialize it — verified).
  Matrix-free Adam is straightforward. **Batched LBFGS is its own
  milestone**: per-element histories, step lengths, curvature-validity
  checks, and history resets are real design work, not a port. Direction
  and step-size policies can be composed optax-style if and when needed —
  do not port the preset explosion.
- **A prerequisite for the LM numerics:** Madsen–Nielsen damping policies
  act on `diag(JTJ)` magnitudes, so they only make sense after per-block
  `scale` (§3) is defined. Heterogeneous blocks — radians, meters,
  pixels, newtons — must not share one unscaled damping value.
- **Linear solvers.** Keep Cholesky and LSTSQ (both real). Delete the CG
  and SparseCholesky stubs and the `"trust_region"` damping enum value
  until they are implemented. Contract: `solve(matvec_or_matrix, b,
  ridge)`, so damping stays orthogonal to the solver.
- **Implicit differentiation (later; the flagship feature).** A
  `torch.autograd.Function` around `run`, whose backward is a linear
  solve against the optimality system (jaxopt's `root_vjp`). The earlier
  "~100 lines" estimate is retracted (codex): manifolds, active bounds,
  robust/nonsmooth kernels, singular systems, and the split between
  optimized and external parameters are the actual contract. What must
  happen *early* is deciding the **differentiation contract** (§9) so
  that M2's state/residual/manifold APIs don't preclude it; the
  implementation itself stays late.
- **Phases.** Keep MultiStage's snapshot/restore idea, generalized to the
  block world: a `Phase` = active residual set + weight overrides +
  variable mask overrides + optimizer config. This is BVR's engine,
  upstreamed.

## 5. Parametric, batched Model values

**Decision:** split `Model` into an explicit **`ModelStructure`**
(frozen: topology, joint kinds and index tables as device tensors, names)
and **`ModelValues`** (a tensor pytree: `joint_placements`,
`body_inertias`, `frame_placements`, limits, `q_neutral`). Structure is a
compile-time constant; values are explicit inputs to every pass. That one
split is simultaneously the enabler for parametric shape, the layout
`torch.compile` likes (static structure to specialize on), and the
backend seam from §2. `Model` remains as the user-facing pairing of the
two. All algorithms index values as `[..., j, :]`, so values may carry
leading batch dimensions and `requires_grad`.

```python
model2 = model.with_values(joint_placements=jp_B)   # (B, njoints, 7), differentiable
data   = br.forward_kinematics(model2, q)           # q (B?, nq) ⊗ value batches
```

- Autograd through placements and inertias already works today via
  `dataclasses.replace` (verified); this design makes it public, batched,
  and tested. `with_values` validates shapes and devices, and is the API
  better_human calls per sample: betas → joint placements/inertias → BR
  does everything else.
- **The broadcast contract (codex).** `[..., j, :]` indexing alone is not
  a design. The execution batch is *defined* as the broadcast of the q
  batch with every value batch. `Data` allocation derives from that
  broadcast (today it derives from `q.shape[:-1]` alone), and the
  contract is tested for batched-values × unbatched-q, the converse, and
  mismatches (which must raise clear errors).
- **Kernel-consumable layout (§2.4).** The structure tensors (int8 kind
  codes, int32 parents/idx, CSR topology, packed axes) and the value
  tensors are the *same arrays* the Warp lane indexes — poses alias
  `wp.transform` zero-copy, and the broadcast contract defines how value
  batches map onto kernel grids. The layout is pinned by a test before
  the first kernel lands.
- **Inertia caching versus differentiability.** Precomputed 6×6 spatial
  inertias (§7) apply to *static* values only. When `body_inertias` is a
  live parametric tensor, the 6×6 form is derived once per evaluation
  context — so gradients stay attached and caches cannot go stale.
- **`.to()` semantics.** Dtype casts apply to floating-point values only,
  never to index/kind tensors. Device transfer is an exhaustive, tested
  tree operation — the current `Model.to()` forgetting `frames` is the
  cautionary example.
- **Frames become a table:** `frame_placements: (*value_batch, nframes,
  7)` in ModelValues, with parent-joint indices as a device tensor in
  ModelStructure. This fixes `Model.to()` not moving frames, enables
  shape-dependent markers, and vectorizes `update_frame_placements` —
  today a per-frame Python loop worth a third of FK+frames time. `Frame`
  objects remain as metadata views.
- **Marker/site frames** are then just rows in that table — better_human's
  landmark sets ride the existing residual/Jacobian stack for free.
- **Order-preserving build:** `build_model(...,
  preserve_joint_order=True)` (or make it the default when the input is
  already topological), plus a public vectorized `q_permutation`. This
  deletes the ~180-line remap shims both consumers wrote.
- **Batched-limit semantics:** limits stay per-model (unbatched) in v1;
  per-batch limits are explicitly out of scope until a use case exists.
- **Human-joint support:** a swing/twist limit residual and a per-joint
  rotation prior for spherical joints (the current `[-1, 1]` quaternion
  boxes are inert); `Inertia.from_mesh` (differentiable, batched),
  replacing the three hand-rolled trimesh loops found across the
  workspace.
- **Mimic joints:** reject at build time *now* (never silent); implement
  properly as part of this milestone's coordinate-map design, not as an
  FK gather. Jacobians, limits, torque accumulation, and all of
  RNEA/ABA/CRBA need the same reduced nq/nv semantics, or FK and dynamics
  will disagree (codex). The acceptance test asserts the reduced
  coordinate map explicitly — pinocchio's *default* loader is equally
  permissive, so parity with it proves nothing. Rejection scope is an
  owner decision (2026-07-17): the Panda itself has a finger mimic, so
  reject-all refuses the flagship parity robot — either adjust the parity
  fixtures or reject only non-identity mimics until this milestone lands.

## 6. Residual library

Keep the callable-object protocol — it earned its keep (analytic
Jacobians verified ~39× faster than FD) — with three changes:

1. **Blocks.** Residuals declare `reads`; `jacobian()` returns per-block
   pieces. The kinematic residuals keep their analytic LWA-Jacobian math
   unchanged — it is correct and fast.
2. **An autodiff fallback that is actually autodiff:**
   `torch.func.jacrev` over the tangent perturbation, batched. FD remains
   only as an explicit `strategy="fd"` debug tool. Delete the enum values
   that lie.
3. **Per-item robust kernels.** The robust kernel moves from
   solver-global to per-residual-item — consumers use Geman-McClure on
   reprojection and plain L2 on priors, simultaneously. Add
   Geman-McClure.

New residual families (from the consumer-gap audit, in demand order):

- `ProjectionResidual` — pinhole reprojection of frame origins/markers
  with per-point confidence weights (BVR's `losses.py` GM-reprojection).
- Point-cloud terms: masked chamfer, and SDF-style
  penetration/attraction/clearance sharing one NN pass (BHF
  object-align).
- `PosePriorResidual` / mean-pose priors (RestResidual generalized
  per-block).
- Keep and finish the velocity/acceleration banded terms; delete or
  implement the stub residuals (Jerk, Yoshikawa, Nullspace) — no shipped
  stubs.
- Collision: port the old capsule mode as `SelfCollisionResidual` for
  humanoid work, or cut `collision/` from the tree until it is real. No
  third option.

## 7. Performance strategy

Two lanes (§2), ordered by leverage:

1. **Fix the syncs and graph breaks.** The free-flyer `_validate_q`
   `bool()` becomes an opt-in debug check (this restores fullgraph
   compile; it is the verified breaker). The `float()`/`bool()`
   per-iteration syncs in the optimizers become tensor ops. Extend the
   hot-path lint to `bool(` / `float(` / `new_tensor` / `torch.eye`, and
   make it watch `residuals/` and `lie/` too. This step is a prerequisite
   for **both** `torch.compile` and CUDA-graph capture.
2. **Hoist the constants.** The pose-residual `new_tensor` per call;
   joint axes and kind codes moved into flat device tensors on
   `ModelStructure`; constant 6×6 spatial inertias precomputed at build
   (RNEA/ABA/CRBA rebuild them per call today). This work doubles as the
   kernel layout work of §2.4.
3. **CPU fast path: `torch.compile` the torch-lane passes** (a 5× CPU
   speedup is already verified for fixed base). Ship a `model.compile()`
   or a documented recipe, tested in CI for both base types, once step 1
   lands.
4. **Vectorize the remaining torch-lane Python loops.**
   `update_frame_placements` becomes one batched compose against the
   frame table. `Model.integrate/difference` group joints by kind and run
   one vectorized op per kind (pyroki-style; BVR measured 2.4–10× and
   already wrote the vectorized version we can steal back). The torch
   lane stays the oracle and the CPU path — it must be *good*, not
   abandoned to the kernels.
5. **GPU fast path: Warp whole-pass kernels** per the §2.2 boundary
   table, landed one at a time (FK first, in M1, as the seam proof; the
   rest in M6), each with parity tests, gradchecks, and a committed
   benchmark against the compiled torch lane. Default-on per kernel only
   on benchmark evidence (§2.7).
6. **CUDA-graph capture of solver inner loops** (§2.6) — the multiplier
   that makes the torch-lane optimizer logic effectively free; a third of
   cuRobo's speed story. Requires the M2 capture-safe `update`
   discipline; lands in M6.
7. **Evaluate per formulation — don't assume.** frax's ancestor-mask
   RNEA/CRBA (einsum-based, no recursion) versus compiled Featherstone
   versus the warp kernel, at 25–160 DOF, on a real GPU box. Benchmarks
   live in-repo with committed numbers, including cuRobo batched-IK
   throughput and a JAX-class (mjx/pyroki) reference point where an
   apples-to-apples comparison exists. (PyposeWarp's unverifiable speedup
   claims are the cautionary tale.)

## 8. Testing & docs strategy

- **Keep** pinocchio parity as the backbone. Extend it over joint kinds ×
  batch shapes × base types *before* the §5 indexing sweep — it is the
  safety net for that refactor.
- **Add:** gradchecks *at* singular points (θ=0, identical quaternions,
  θ→π); jaxlie-style property tests (hypothesis: group axioms × batch
  shapes × near-identity sampling, with NaN-asserting comparators);
  batched-IK and bounds-active-IK regression tests; a CI assertion that
  analytic Jacobians beat autodiff (jaxlie's trick — performance claims
  as tests); solver convergence tests on non-trivial targets with limits
  enabled.
- **Docs:** every snippet executes in CI (myst doctest or scripts); the
  stub inventory (`roadmap.md`) is generated by grepping
  `NotImplementedError`; per-package CLAUDE.md claims get the same
  treatment as docs — they drifted identically.
- **Un-ossify:** unfreeze the 26-symbol API contract until 1.0; delete
  the deprecation shims, the `no_legacy_strings` test, and the IR
  schema-version handshake.
- **Set up CI.** There is none. Tests on push + snippet execution + the
  performance assertions above.

## 9. Cross-cutting engineering contract (added after the codex review)

These are decisions that constrain the architecture — not post-M5
cleanup. Each gets a short written policy before the API it constrains
freezes:

- **Dtype & numerics policy.** Supported dtypes (fp32 primary, fp64
  supported, fp16 explicitly rejected until designed — an exception class
  already claims this without enforcing it); accumulation and
  factorization dtypes; dtype-dependent tolerances and Taylor cutoffs; a
  TF32 stance for convergence-critical dot products; "preserve input
  dtype" as a tested invariant.
- **Quaternion double cover & continuity.** How q ~ −q is handled in
  priors, temporal residuals, interpolation, and better_human pose
  parameters — hemisphere alignment or sign-invariant metrics, plus a
  stated convention near the log-map discontinuity at θ = π.
- **Threading & reentrancy.** Concurrent solves must not share mutable
  state — one more reason caching is evaluation-local (§3) and `Model`
  must be deeply immutable (today it carries mutable dicts and `meta`).
  CUDA stream behavior and multiprocessing-spawn behavior are stated.
- **Serialization.** A versioned structure/values `state_dict` (plus
  optimizer/warm-start state), `map_location` behavior, and an explicit
  stance on pickle. Today `Model.meta` retains builder IR and resolver
  objects — exactly what must not leak into a checkpoint format.
- **Differentiation contract.** Which gradients are guaranteed (with
  respect to q, model values, residual parameters, solver
  hyperparameters), to what order, through which paths (unrolled versus
  implicit), and what happens on non-converged solves. "PyTorch-native"
  is not a contract; this is.
- **Compile lifecycle.** Which dimensions are dynamic; graph-cache keying
  and limits; cold-start expectations (codex measured ~31 s for the first
  compile of FK — amortized only after ~12k calls); and whether a phase
  change means zero weights on fixed shapes or separate compiled
  programs. The same questions for the warp side: kernel-cache location
  and versioning in CI, first-call codegen latency, CUDA-graph re-record
  triggers.
- **Warp kernel policy (the normative form of §2.3–§2.5).** Stream
  bridging via `wp.stream_from_torch`, always. `requires_grad=False` at
  `wp.from_torch`, with gradients managed torch-side. Layout
  preconditions checked up front, with torch-lane fallback (a hard error
  only inside capture — never a silent `.contiguous()`). fp32 via
  `transformf`, fp64 via `transformd`, bf16 rejected (warp has none).
  Second order served by torch-lane VJP recomputation inside the
  registered backward (§2.3), gradgradcheck'd through the public API.
  **Torch ≥ 2.4 floor** for the custom-op bridge (the version policy is
  owned here). Kernels live in real `.py` files — warp parses source
  files, so no generated strings. Every kernel ships: an adjoint-strategy
  entry in the design table; parity + gradcheck versus the torch lane,
  including singular points, branched trees, and >16-joint chains; a
  warp-CPU CI run; and a committed benchmark. The kernel-author
  gradient-trap checklist (dynamic loops, in-place ops, vector-component
  writes, data-dependent atomics, cross-launch buffer aliasing) is
  applied in review.
- **Per-element failure semantics.** Batched APIs report status per
  element (NaN residual, factorization failure, infeasible bounds, stall,
  maxiter). A single batch-wide enum would recreate today's misleading
  statuses.
- **Licensing & provenance.** The owner selected Apache-2.0 on 2026-07-17.
  Keep a source ledger distinguishing algorithm reimplementation from copied
  code (pyroki is MIT; jaxopt/mujoco_warp/newton are Apache-2.0), and retain
  every applicable upstream license, attribution, and notice.
- **Migration as a deliverable.** A symbol-by-symbol table for the named
  consumers (BHF's legacy surface: `CostStack`, `Data.oMi` including
  *assignments*, `GaussNewton.minimize`, the Huber/Cauchy kernels,
  `Trajectory`, trajectory residuals, `ResidualState`), executed at M4
  when the consumers switch to the redesign branch. Under the branch
  strategy (owner decision 2026-07-17), shims and deletion ordering on
  the redesign branch are no longer constraints.

## 10. Packaging & repo hygiene

- Extras: `viewer` (viser), `io-mjcf` (mujoco), `meshes` (trimesh),
  `demos` (robot_descriptions), and `warp` (warp-lang — the GPU fast
  lane). The core stays importable and fully functional without the warp
  extra, per §2.5; the extra is promoted to a hard dependency only if and
  when kernels become default-on for CUDA. Core depends on torch + numpy
  + yourdfpy only. Drop `rich` (unused).
- Delete: `utils/` (unimported), the residual registry, `ResidualSpec`
  (its banded/temporal structure information moves into the
  sparse-milestone design notes first), the `chain.py` / `indexing.py`
  stubs, `dynamics/action/` (park it in a branch until a DDP solver is
  actually planned — it is ~290 lines of unusably-slow speculative
  scaffolding), and the `retarget` stub from the public API (design doc
  first, code when real). **Update (owner decision 2026-07-17,
  superseding the codex shim exception):** the redesign is implemented on
  a dedicated branch and BHF/BVR stay on the pre-redesign branch until
  the M4 migration, so no shims are required — `costs/` and the
  deprecated `Data` aliases may be deleted outright. Every removed
  consumer-facing symbol goes into the M4 migration table; BHF's legacy
  surface also includes `GaussNewton.minimize`, the Huber/Cauchy kernels,
  `Trajectory`, trajectory residuals, and `ResidualState`.
- `Model.meta["ir"]` retention: slim it down to the geometry table the
  viewer needs, or make IR retention opt-in at load time.
