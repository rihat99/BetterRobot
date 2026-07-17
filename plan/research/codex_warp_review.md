# Codex adversarial review — Warp-first revision (2026-07-16, round 2)

Reviewer: codex/gpt-5.6-sol, xhigh reasoning, read-only. Scope: the warp-first revision of plan/ (03 §2/§7/§9, 04 roadmap, 02 vision) and the three warp audits. Corrections folded back into plan/ are marked in the docs themselves.

1. [BLOCKING] The bridge contract is internally impossible as written — `03_architecture.md` §2.3; `04_roadmap.md` M1.

   Claim: one ~40-LOC `torch.library.custom_op` helper will use caller-owned, preallocated output/gradient buffers while supporting registered autograd, `torch.compile`, and graph capture ([03_architecture.md:117](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:117), [03_architecture.md:121](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:121), [03_architecture.md:132](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:132)).

   Refutation: passing an output buffer and writing into it makes the custom op non-functional. PyTorch rejects `register_autograd` for such schemas ([custom_ops.py:597](/data3/rikhat.akizhanov/better/BetterRobot/.venv/lib/python3.11/site-packages/torch/_library/custom_ops.py:597)). I reproduced the exact failure locally: `Cannot register autograd formula for non-functional operator ... Please create a functional operator`. Warp’s own recommended implementation uses two functional custom ops—forward and backward—with separate fake registrations, and allocates outputs in both ([interoperability.rst:388](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/docs/user_guide/interoperability.rst:388), [interoperability.rst:403](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/docs/user_guide/interoperability.rst:403)). Shape functions, saved context, schemas, and backward returns are pass-specific; this is not “written once.”

   Minimal fix: before preserving any bridge claim, prototype one real multi-output FK-shaped functional forward op plus a functional backward op, including fake registration, `torch.compile(fullgraph=True)`, AOTAutograd, first/second derivatives, and CUDA capture. Decide explicitly whether outputs allocate from Torch’s graph pool or whether a functional outer op wraps private mutating launch ops. Remove the caller-owned-output and “~40 LOC” promises until that prototype exists.

2. [BLOCKING] The proposed common layout removes the static metadata on which the claimed Torch CPU path depends — `03_architecture.md` §§2.1, 2.4, 5; M1 seam.

   Claim: both lanes index the same device-tensor parents/kinds/indices, pass signatures contain no Python-object hot-loop state, while `torch.compile` remains the CPU fast path ([03_architecture.md:156](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:156), [04_roadmap.md:57](/data3/rikhat.akizhanov/better/BetterRobot/plan/04_roadmap.md:57)).

   Refutation: the existing compile result was obtained because topology, slice bounds, and joint dispatch are Python tuples/objects and therefore statically unrolled ([audit_compute_pass_inventory.md:122](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_compute_pass_inventory.md:122)). A device tensor’s contents do not become compile-time constants merely because its dataclass is frozen. A tensor-only Torch recursion must instead implement dynamic parent gathers, sentinel handling, variable q slices, and tensor kind dispatch. That may compile, but it is a materially different algorithm from the one that produced the 5×/10× evidence. The plan currently spends that old benchmark as evidence for an unbuilt design.

   Minimal fix: choose one of two honest designs in M1: retain immutable Python/static mirrors for Torch specialization alongside device tables for Warp, with consistency tests; or first build and benchmark a tensor-only Torch FK/RNEA prototype. Do not delete the existing static representation until the latter is shown compile-clean and competitive.

3. [BLOCKING] The value-batch/kernel-grid ABI is explicitly both “decided before M1” and deferred to M3 — `03_architecture.md` §§2.4, 5; `04_roadmap.md` M3.

   Claim: q-batch × value-batch mapping is fixed before the first kernel ([03_architecture.md:168](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:168), [03_architecture.md:411](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:411)).

   Refutation: M3 says that milestone will “fix” the mapping and update the M1 FK kernel ([04_roadmap.md:111](/data3/rikhat.akizhanov/better/BetterRobot/plan/04_roadmap.md:111)). This is an ABI change, not breadth work. Warp kernel array rank is part of the signature, and arrays support at most four dimensions ([runtime.rst:181](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/docs/user_guide/runtime.rst:181)). Arbitrary public `B...` therefore needs a canonical flattening rule. Broadcast-expanded Torch values have zero strides; flattening those generally cannot remain both zero-copy and shape-generic. Gradients for an unbatched placement shared across B must reduce across threads, not return an expanded gradient that Torch subsequently reduces again.

   Minimal fix: define in M1 a fixed-rank kernel ABI, for example flat execution batch `E`, unexpanded value arrays, and explicit per-input batch-index maps/strides. Specify the reverse reduction for shared values and test unbatched-values × batched-q, batched-values × unbatched-q, multi-axis batches, and mismatches before FK lands. Remove the M3 remapping.

4. [BLOCKING] “Runtime serial loop” is not a generally adjoint-safe kernel policy — `03_architecture.md` §2.1/§2.3; Warp audit §7.3.

   Claim: a runtime `njoints` loop, one thread per batch element, is the “differentiable-safe” Newton form and remains robot-shape-generic ([03_architecture.md:76](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:76)).

   Refutation: Warp documents that dynamic loops do not replay required local intermediates and can silently return `[32,8,2]` instead of `[4,4,4]` ([differentiability.rst:1411](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/docs/user_guide/differentiability.rst:1411), [differentiability.rst:1473](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/docs/user_guide/differentiability.rst:1473)). Static loops only unroll below `max_unroll` unless configured otherwise ([codegen.py:2572](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/warp/_src/codegen.py:2572)).

   The audit itself overstates the evidence: it calls Newton’s topology loop “static ... unrolled” ([audit_warp_platform.md:213](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_warp_platform.md:213)), but Newton actually executes `range(joint_start, joint_end)` with runtime bounds ([articulation.py:213](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/newton/newton/_src/sim/articulation.py:213)). Newton manually unrolls only its D6 local accumulation ([articulation.py:281](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/newton/newton/_src/sim/articulation.py:281)). Its IK tests show q-Jacobian parity for a small FK case ([test_ik.py:454](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/newton/newton/tests/test_ik.py:454)); they do not prove placement gradients or RNEA/ABA/CRBA recursions safe.

   Thus per-robot codegen is not logically forced: explicit intermediate arrays or a hand-written runtime reverse sweep can preserve shape-generic kernels. But the plan must choose. RNEA/ABA/CRBA have much more loop-carried local state than FK.

   Minimal fix: add a per-kernel adjoint design table: generated adjoint with named stored intermediates, hand VJP, or static specialization. M1 FK must test branched trees and chains longer than 16 joints, gradients to q and placements, and shared-value reductions. Stop generalizing FK evidence to all tree passes.

5. [BLOCKING] Automatic second-order routing cannot be decided at the pass call site — `03_architecture.md` §2.1/§2.3.

   Claim: runtime selection is keyed on “differentiation order,” and `create_graph=True` automatically routes to Torch ([03_architecture.md:84](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:84), [03_architecture.md:149](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:149)).

   Refutation: when `fk(q)` executes, it cannot know whether the caller will later invoke `backward(create_graph=True)`. Warp has no adjoint-of-adjoint support ([audit_warp_platform.md:101](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_warp_platform.md:101)). A separate Warp backward custom op without its own autograd formula simply terminates the second derivative. This also affects users differentiating a returned analytic Jacobian and the `compute_*_derivatives` APIs.

   Minimal fix: specify one implementable mechanism: an explicit Torch-lane context/argument before forward; or a registered backward that detects grad-enabled backward and recomputes the Torch VJP with saved inputs. Add gradgradcheck through the public API, not merely direct Torch-lane tests.

6. [BLOCKING] The “every ModelValues leaf” gradient guarantee is broader than the scheduled adjoints — `03_architecture.md` §§2.2–2.4, 5; M1/M3/M6.

   Claim: both lanes guarantee gradients to q and every ModelValues leaf ([03_architecture.md:149](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:149)).

   Refutation:

   - M1 accepts only q and joint-placement gradients ([04_roadmap.md:59](/data3/rikhat.akizhanov/better/BetterRobot/plan/04_roadmap.md:59)).
   - The boundary table calls the first kernel “FK + frame placements,” which also depends on `frame_placements` ([03_architecture.md:92](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:92)).
   - Dynamics needs placement and inertia VJPs; residuals may need frame placements, targets, limits, and robust parameters. cuRobo’s hot implementations do not establish this—the audited FK backward is q-only, and roughly half its kinematics kernel code is backward machinery ([audit_curobo_warp_integration.md:403](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_curobo_warp_integration.md:403)).
   - A literal “every leaf” promise is also ill-formed: an FK pass should not invent gradients for inertias or limits it does not consume.

   Minimal fix: replace the slogan with a pass-by-pass differentiable-input matrix and require gradients for every floating input that the pass actually consumes. Resolve whether M1 FK includes frames. Add unbatched and value-batched gradient-reduction tests. Price inertia, frame, and residual-parameter VJPs explicitly in M6.

7. [MAJOR] The two-lane maintenance budget is not credible for a solo maintainer — `03_architecture.md` §§2.1, 7; `04_roadmap.md` M1/M6.

   Claim: Warp-first is “one rewrite instead of two” ([03_architecture.md:25](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:25)).

   Refutation: the plan immediately mandates a rewritten, vectorized, compiled Torch oracle and a separate Warp forward/adjoint implementation ([03_architecture.md:70](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:70), [03_architecture.md:491](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:491)). That is two implementations plus differential tests.

   The current eight central BR algorithm files are about 1,699 LOC; all BR Python is about 16,058 LOC and tests 8,679 LOC. The cuRobo audit finds about 5.7k LOC of core-robotics Warp code, 1.6k wrapper LOC, and a near-1:1 test/code ratio overall ([audit_curobo_warp_integration.md:380](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_curobo_warp_integration.md:380)). PyposeWarp’s 13.3k figure is explicitly unverified in the fresh audit ([audit_warp_platform.md:267](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_warp_platform.md:267)), so it cannot serve as a calibrated estimate.

   M1 contains two M–L items, multiple M items, packaging, CI, docs, deletion, and migration, yet is labeled 3–4 weeks ([04_roadmap.md:53](/data3/rikhat.akizhanov/better/BetterRobot/plan/04_roadmap.md:53)). By the roadmap’s own effort definitions, those serial solo-maintainer estimates do not add up.

   Minimal fix: split M1 into seam/layout, bridge prototype, and GPU-validated FK milestones. Add per-family implementation/test estimates and a maintenance ceiling. Treat the M6 build-out as multi-month unless evidence shows otherwise; prioritize FK/Jacobian and one residual/collision path before committing to all dynamics kernels.

8. [MAJOR] M1 cannot validate the CUDA-specific reasons for landing its production bridge or kernel — `04_roadmap.md` standing caveat/M1.

   Claim: Warp-CPU parity is sufficient for M1; GPU validation waits until M6 ([04_roadmap.md:30](/data3/rikhat.akizhanov/better/BetterRobot/plan/04_roadmap.md:30)).

   Refutation: Warp-CPU cannot test `wp.stream_from_torch`, side-stream ordering, CUDA graph capture, GPU atomics used for shared-value gradients, block/grid races, device alignment, NVRTC codegen, or useful thread mapping. It executes the entire launch grid serially in one host thread ([codegen.py:4312](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/warp/_src/codegen.py:4312)). Therefore the M1 bridge’s most load-bearing rules are not exercised at all.

   Minimal fix: land the layout and Torch seam without a production Warp default, but require at least a remote CUDA correctness runner before accepting the bridge/FK milestone. If no GPU is available, keep the Warp CPU kernel as a research prototype and do not call it proof of the production seam.

9. [MAJOR] Several boundary-table decisions violate the table’s own boundary principle — `03_architecture.md` §2.2.

   Refutation:

   - `integrate/difference` is committed to Warp ([03_architecture.md:95](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:95)) while §7 also commits to a grouped vectorized Torch rewrite ([03_architecture.md:491](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:491)). Once grouped by kind, this is no longer a serial tree traversal and should be benchmark-gated, not preassigned to Warp.
   - Warp has no SO3/SE3 log/exp builtins ([audit_warp_platform.md:150](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_warp_platform.md:150)), contradicting the implication that native transform/quat builtins cover the Lie math used inside kernels ([03_architecture.md:99](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:99)). Integrate, difference, and pose residual kernels therefore require new numerically stitched Warp Lie functions and adjoints.
   - “Collision, no torch legacy” conflicts with mandatory CPU capability: M4 must still implement a Torch CPU version before the M6 Warp version.
   - “Gradients in the forward residual kernel” is underspecified. If the kernel returns raw residual `r` and Jacobian `J`, Torch robust weighting is compatible. If it returns already-weighted gradients, Torch IRLS can double-weight or use a different robust grouping.
   - No canonical J layout/stride contract exists for Warp-produced J feeding Torch `JᵀJ`.

   Minimal fix: define the residual ABI as raw residual plus canonical contiguous `(E,R,nv)` Jacobian and explicit robust row-group semantics. Move integrate/difference to “Torch; Warp only if profiling wins.” Add the missing Warp Lie-math task and a Torch collision reference.

10. [MAJOR] The cuRobo/JAX performance target is not an acceptance criterion and the proposed kernel mapping is structurally disadvantaged — `02_vision.md` horizon; M6.

   Claim: cuRobo-class batched IK and JAX-class sweeps ([02_vision.md:92](/data3/rikhat.akizhanov/better/BetterRobot/plan/02_vision.md:92)).

   Refutation: cuRobo’s relevant speed core is robot-specialized hand CUDA, including fused FK+spheres+Jacobian, RNEA, optimizer primitives, and line search ([audit_curobo_warp_integration.md:48](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_curobo_warp_integration.md:48)). BR explicitly gives up both per-robot specialization and that kernel language. At B=1, its stated mapping launches one CUDA thread to serially traverse the whole robot. Graph capture removes launch overhead; it does not create intra-kernel parallelism or make Torch optimizer tensor work “free.”

   Worse, M6 succeeds if each enabled kernel merely beats compiled Torch ([04_roadmap.md:190](/data3/rikhat.akizhanov/better/BetterRobot/plan/04_roadmap.md:190)); it may remain far behind cuRobo and still pass. “JAX-library-class” is undefined across different formulations, precision, XLA fusion, compilation, and outputs.

   Minimal fix: specify canonical external benchmarks now: hardware, B, robot, targets, seeds, fixed iterations, success tolerance, collision inclusion, precision, warm/cold timing, and memory. State that a gap is expected until measured. Benchmark serial-generic versus cached per-robot/static and branch/level-parallel Warp mappings; permit specialization if generic kernels miss the target.

11. [MAJOR] “Capture-safety lint” cannot certify the promised solver capture — `03_architecture.md` §§2.6, 4; `04_roadmap.md` M2b/M6.

   Claim: `update` will be allocation-free and a lint in M2b makes M6 capture a feature rather than a rewrite ([04_roadmap.md:94](/data3/rikhat.akizhanov/better/BetterRobot/plan/04_roadmap.md:94)).

   Refutation: a pure `update(values,state) -> new values,state` using `torch.where`, matmul, `cholesky_ex`, and residual evaluation allocates output tensors by normal Torch semantics. Warp itself documents that allocation during capture can be legal under graph pools ([audit_warp_platform.md:158](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_warp_platform.md:158)); “no allocations” is therefore both unrealistic and the wrong criterion. An AST lint cannot establish capture safety of cuSOLVER, `lstsq`, autograd/Jacobian code, or user residuals.

   cuRobo’s real requirements are stronger and different: three warmups, static input buffers, `copy_` before replay, stable addresses, explicit resize/reset, and outer/inner loop separation ([audit_curobo_warp_integration.md:175](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_curobo_warp_integration.md:175), [audit_curobo_warp_integration.md:210](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_curobo_warp_integration.md:210)). `cholesky_ex` masks are capture-safe only if the fallback is also expressed as fixed tensor work; selecting failed elements dynamically is not.

   Also, `wp.capture_while` does not automatically see a Torch-started capture: it looks up a Warp-registered active graph ([context.py:8666](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/warp/_src/context.py:8666)). External Torch capture requires explicit Warp external-capture integration.

   Minimal fix: make M2’s criterion a structural checklist, not proof. Define the fixed-buffer executor, warmup, input-copy, output-lifetime, invalidation, custom-residual eligibility, and branch-free factorization fallback. Require an actual CUDA capture/replay parity test in M6. Drop `wp.capture_while` until mixed Torch/Warp external capture is demonstrated.

12. [MAJOR] The CPU non-regression story has no performance acceptance and relies on evidence from the old representation — `03_architecture.md` §§2.5, 7; M1/M6.

   Refutation: the measured compile probe took about 31.5 seconds and amortized only after roughly 12,000 FK calls ([codex_plan_review.md:186](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/codex_plan_review.md:186)). M1 changes the metadata and algorithm shape, so even its warm performance is unproven. Users without or unwilling to invoke Inductor fall back to eager Torch, yet no milestone requires eager CPU latency not to regress.

   The suggested Clang-availability concern is somewhat misplaced: Warp wheels use embedded LLVM/Clang for CPU kernels ([audit_warp_platform.md:110](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_warp_platform.md:110)). The real CI risks are cold compilation per dtype/module, cache permissions/restoration, wheel availability over the supported Python matrix, and running fp32/fp64 forward+adjoint kernels serially.

   Minimal fix: add eager and compiled CPU baselines after the new seam, with cold time, warm time, peak memory, and break-even call count. Keep `model.compile()` opt-in and document eager as a supported performance floor. Add a dedicated Warp-extra CI job with a persistent kernel cache and bounded small parity cases.

13. [MAJOR] The zero-copy/layout guarantee is only conditionally true and is misstated for fp64 and non-contiguous inputs — `03_architecture.md` §§2.3–2.4, 9.

   Refutation:

   - `[t,qxyzw]` is genuinely layout-compatible, but `wp.transform` is an fp32 alias; fp64 requires `wp.transformd` ([types.py:1619](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/warp/_src/types.py:1619), [types.py:1831](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/warp/_src/types.py:1831)).
   - `wp.from_torch(..., dtype=transform*)` requires the trailing value-type dimension and its inner stride to be contiguous ([torch.py:223](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/warp/_src/torch.py:223)); outer non-contiguous and zero strides are supported. The blanket hard-fail-contiguity policy is stricter than Warp itself.
   - Once Warp is default, rejecting a non-contiguous Torch slice that currently works is a public semantic regression. It should fall back to Torch unless contiguous input is an explicit public contract.
   - Explicit preallocated gradient descriptors require gradient strides to match the input exactly ([torch.py:258](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/warp/_src/torch.py:258)), which interacts badly with broadcast-expanded zero-stride values.
   - The spatial `[lin,ang]` warning is correct; Warp is angular-first ([spatial.h:18](/data3/rikhat.akizhanov/better/BetterRobot/references/sim/warp/warp/native/spatial.h:18)). This means BR cannot casually use Warp’s spatial cross/adjoint/mass functions after only swapping input/output once; convention handling must surround every such builtin.

   Minimal fix: state the exact alias preconditions, dispatch `transformf`/`transformd`, define non-contiguous fallback behavior, and test pointer equality plus strides—not only values. Add canonical layouts for q, poses, inertias, J, residuals, and gradients.

14. [MAJOR] The consistency sweep is incomplete and will actively misdirect implementation — plan assessment, `CLAUDE.md`, docs, packaging.

   Examples:

   - `01_assessment.md` says the existing `*_raw` functions are already the needed tensor-only seam ([01_assessment.md:157](/data3/rikhat.akizhanov/better/BetterRobot/plan/01_assessment.md:157)); the revised architecture correctly says they still consume `Model`, Python JointModels, and mutable Data ([03_architecture.md:204](/data3/rikhat.akizhanov/better/BetterRobot/plan/03_architecture.md:204)).
   - It says mujoco_warp and Newton wrap their kernels in `torch.autograd.Function` ([01_assessment.md:158](/data3/rikhat.akizhanov/better/BetterRobot/plan/01_assessment.md:158)); the fresh audit says Newton has no Torch bridge in core ([audit_warp_platform.md:204](/data3/rikhat.akizhanov/better/BetterRobot/plan/research/audit_warp_platform.md:204)).
   - It still sends GPU profiling to M5 ([01_assessment.md:230](/data3/rikhat.akizhanov/better/BetterRobot/plan/01_assessment.md:230)); GPU profiling is now M6.
   - `CLAUDE.md` says the `backends → lie` DAG must “never” be violated ([CLAUDE.md:41](/data3/rikhat.akizhanov/better/BetterRobot/CLAUDE.md:41)), while M1 deletes it.
   - `docs/concepts/batching_and_backends.md` declares explicit Backend objects and `backend=` kwargs the architectural core ([batching_and_backends.md:22](/data3/rikhat.akizhanov/better/BetterRobot/docs/concepts/batching_and_backends.md:22)).
   - `docs/conventions/performance.md` says graph replay destroys the grad tape and capture is opt-in ([performance.md:140](/data3/rikhat.akizhanov/better/BetterRobot/docs/conventions/performance.md:140)); the revision depends on capturing backward.
   - `pyproject.toml` permits Torch 2.1 ([pyproject.toml:12](/data3/rikhat.akizhanov/better/BetterRobot/pyproject.toml:12)), while the compile-safe custom-op route requires Torch ≥2.4.

   Minimal fix: correct the internal plan contradictions immediately. Add an explicit M1 documentation/contract-test migration item covering `CLAUDE.md`, architecture/batching/performance docs, generated API pages, dependency tests, mixed-precision claims, and Torch-version policy. “Docs truth pass” is too vague for this architectural inversion.

## Verdict

The Warp-first direction is plausible, but this revision is not sound enough to execute as written. It has three unresolved ABI-level contradictions: the custom-op/preallocated-buffer design, the shared tensor layout versus the Torch compile path, and the value-batch/gradient mapping. The dynamic-loop policy and automatic second-order routing are also not yet implementable contracts.

Before M1 begins, the plan must:

- Replace the bridge prose with a working functional custom-op design.
- Freeze the execution-batch, layout, and per-pass gradient ABI.
- Decide how Torch retains static topology performance.
- Require GPU correctness access before the production bridge/FK kernel can be accepted.
- Re-estimate M1/M6 as two maintained implementations, not “one rewrite.”

## Three assumptions to validate before writing M1 code

1. A complete Torch↔Warp custom-op prototype works for fp32/fp64, first and second derivatives, shared model-value gradients, `torch.compile`/fake tensors, current-stream execution, and CUDA graph replay.

2. A robot-size-generic Warp FK with a runtime loop produces correct q and placement adjoints on branched and >16-joint models—and is competitive on a real GPU at B=1, medium B, and large B. Compare generated adjoint, stored-intermediate, hand-VJP, and static-specialized versions.

3. The proposed ModelStructure/ModelValues representation can simultaneously provide zero-copy fixed-rank Warp inputs and a compile-clean, non-regressing Torch CPU lane across q-batch × value-batch broadcasting. If not, accept a tested dual static/device metadata representation.
