# Adversarial review of the BetterRobot redesign plan

This review treats the plan as a proposal to falsify, not a direction to endorse. I inspected the synthesized documents, the implementation, selected raw audits, and the three named consumer repositories. I also ran CPU-only probes with `uv run python` (PyTorch 2.11, generally one Torch thread for timings). CUDA-specific performance claims could not be reproduced because CUDA is unavailable on this machine.

## A. Refuted/corrected claims

### A1. NaN-at-identity gradients: verified, but the stated blast radius is too broad

The underlying defect is real. `_so3_exp_impl` computes `theta = sqrt(theta_sq)` before selecting its small-angle polynomial, and `_so3_log_impl` similarly computes a norm before masking (`src/better_robot/lie/_torch_native_backend.py:145-160`, `src/better_robot/lie/_torch_native_backend.py:163-184`). The SE(3) implementations reuse these paths (`src/better_robot/lie/_torch_native_backend.py:259-311`). CPU probes produced finite forward values but NaN first derivatives for `so3.exp(0)`, the rotational part of `se3.exp(0)`, `so3.log(identity)`, and the rotational part of `se3.log(identity)`. Second derivatives were also non-finite. The `[verified]` finding in `plan/01_assessment.md:18-41` therefore survives.

The consequence is overstated, however. The current BetterRobot Adam and LBFGS implementations form a residual Jacobian explicitly (`src/better_robot/optim/optimizers/adam.py:79-80`, `src/better_robot/optim/optimizers/lbfgs.py:81-82`, `src/better_robot/optim/optimizers/lbfgs.py:148-149`) and apply the retraction outside that differentiation. They do not necessarily encounter this NaN on every ordinary iteration. The defect does block a consumer's `torch.optim` closure over tangent increments, the proposed `f(x ⊕ δ)` Jacobian construction, and higher-order differentiation. The plan should say that precisely instead of implying that every current first-order step is already unusable.

### A2. Batched `solve_ik`: verified, and the failure is more fundamental than one bad matrix product

The immediate crash is real. With a Panda model and `q` of shape `(4, nq)`, `solve_ik` fails at `r0 @ r0` in `src/better_robot/optim/state.py:90-95` with incompatible 2-D batch semantics. The solver also allocates a single unbatched vector and dense Jacobian in `src/better_robot/kinematics/jacobian.py:246-284`; its finite-difference loop is scalar-problem code. `solve_ik` additionally hard-codes float32 initialization (`src/better_robot/tasks/ik.py:184-188`, `src/better_robot/tasks/ik.py:221-222`). Thus `plan/01_assessment.md:43-51` is supported, but fixing the cost reduction alone will not make the path batched.

### A3. Mimic metadata is computationally unused: verified, but the proposed acceptance test can certify the same mistake

The model stores mimic source, multiplier, and offset (`src/better_robot/data_model/model.py:75-78`), and the builder populates them (`src/better_robot/io/build_model.py:479-504`), but FK, Jacobians, and dynamics do not consume them. On the bundled Panda, both finger joints still own independent scalar configuration coordinates (`nq = nv = 9`), despite the second finger naming the first as its mimic source. That part of `plan/01_assessment.md:53-61` survives.

The roadmap's “matches Pinocchio” acceptance criterion is insufficient (`plan/04_roadmap.md:17`). A normal Pinocchio `buildModelFromUrdf` probe on the same Panda also reported `nq = nv = 9`; ordinary loading does not by itself establish a reduced mimic-coordinate reference. The test must build an explicitly constrained reference or assert the intended reduced coordinate map. Moreover, enforcing mimic joints is not merely a gather in FK: Jacobians, limits, torques, CRBA/RNEA/ABA, and configuration/tangent indexing all need the same reduced-coordinate semantics. “Enforce or reject” is sensible; treating the enforcement branch as a small M0 patch is not.

### A4. Bounded LM does stall, but the plan gives the wrong mechanism

`plan/01_assessment.md:63-73` says the solver clamps “after an accepted step” and thereby wrecks the acceptance ratio. The implementation does the opposite: it projects the trial configuration before evaluating the trial residual (`src/better_robot/optim/optimizers/levenberg_marquardt.py:90-110`). Acceptance is then the simple comparison `cost_new < cost` (`src/better_robot/optim/optimizers/levenberg_marquardt.py:114`); the gain ratio is computed later (`src/better_robot/optim/optimizers/levenberg_marquardt.py:116-117`) and the default adaptive strategy ignores it, merely halving or doubling damping (`src/better_robot/optim/strategies/adaptive.py:16-25`). The claimed causal chain is therefore false.

The adverse result is reproducible. For an unreachable Panda pose, an unconstrained solve converged in 25 iterations to roughly `9.6e-15` cost but violated limits. The bounded solve remained feasible but reached 300 iterations at approximately `0.0341` cost, `0.259 m` position error, and `0.0345 rad` orientation error, with three joints pinned at bounds. The status was `maxiter`, not a false `converged`; the real status defect is that there is no distinct active-set/stalled/KKT outcome. The plan is right to replace the bounded method, but it must retract both the clamp ordering and gain-ratio explanation.

### A5. “Autodiff” silently means central finite differences: verified, with an off-by-one and an important benchmark correction

The strategy switch has only an analytic branch and a default fallback (`src/better_robot/kinematics/jacobian.py:246-284`). `AUTODIFF`, `FUNCTIONAL`, and `FINITE_DIFF` all enter the same central-difference loop despite their public descriptions (`src/better_robot/optim/jacobian_strategy.py:11-25`). A custom residual probe gave bit-identical Jacobians for all three settings and counted 19 residual calls at `nv = 9`: one base evaluation plus `2*nv`, not merely `2*nv`. Batched input also fails in this fallback. The substance of `plan/01_assessment.md:75-83` survives; the call-count wording should be `2nv + 1` for the complete routine.

The reported 42x figure is not “analytic versus autodiff.” On a controlled Panda pose-residual probe, analytic Jacobian evaluation took about `1.59 ms`, an actual `torch.func.jacrev` closure about `13.99 ms`, and central finite differences about `62.0 ms`. That is approximately 8.8x for analytic versus reverse-mode autodiff and 39x for analytic versus FD. A `torch.func.jacfwd` attempt failed with a float/double mismatch in the current closure. The plan must stop using the mislabeled current enum as a proxy for real autodiff performance.

### A6. Adam/LBFGS build dense Jacobians: verified

This claim is exactly supported by `src/better_robot/optim/optimizers/adam.py:79-80` and `src/better_robot/optim/optimizers/lbfgs.py:81-82`. Neither is matrix-free today, so `plan/01_assessment.md:129-145` lands a valid architectural criticism. What does not follow is that every future first-order objective should be forced into least-squares residual form; the named consumers also have scalar/reduced terms and auxiliary outputs.

### A7. Collision is operationally absent, but “100% stubs/every path raises” is literally false

All useful query paths are stubs: closest points (`src/better_robot/collision/closest_pts.py:14-36`), SDF distance (`src/better_robot/collision/geometry.py:57-66`), pair distance (`src/better_robot/collision/pairs.py:32-40`), robot collision methods (`src/better_robot/collision/robot_collision.py:36-73`), and collision residual evaluation/Jacobian (`src/better_robot/residuals/collision.py:49-57`, `src/better_robot/residuals/collision.py:93-97`). So the product conclusion in `plan/01_assessment.md:85-90` is fair.

However, geometry dataclasses and the pair registry are implemented, and construction/registration does not raise (`src/better_robot/collision/pairs.py:16-29`). The accurate statement is “every collision query/optimization path is unimplemented,” not “every code path is a stub.” This distinction matters when deciding whether to delete a misleading API or retain usable schema pieces.

### A8. The 19 µs / 27% dispatch-overhead claim is refuted

Repeated blocked-autorange CPU measurements of the public FK facade versus its direct implementation found:

| Workload | Facade | Direct | Difference |
|---|---:|---:|---:|
| Panda, unbatched | 59.41 µs | 57.66 µs | 1.75 µs (3.0%) |
| Panda, batch 256 | 88.89 µs | 86.88 µs | 2.01 µs (2.3%) |

This agrees with the separate raw quality audit, which says approximately 1 µs / 3% (`plan/research/audit_quality_perf.md:122-124`), and contradicts the 18.8-19 µs claim in the architecture audit (`plan/research/audit_core_architecture.md:37-43`) and synthesis (`plan/01_assessment.md:96-127`). The plan cites the favorable-to-deletion measurement without resolving its own raw reports' order-of-magnitude conflict. Deleting the current abstraction may still be structurally justified; the measured performance case is not.

This also directly undercuts the README's claim that “every load-bearing finding” was established first-hand (`plan/README.md:3-6`). The finding may have been measured twice, but the incompatible results were neither reconciled nor bounded; presenting one as settled is not verification.

### A9. Per-call host-to-device “constants” mix real defects with no-ops and unmeasured speculation

There are real device-lifetime problems. Pose residuals call `.to()` and construct tensors per call (`src/better_robot/residuals/pose.py:59-65`, `src/better_robot/residuals/pose.py:76-102`); joint axes live in Python joint-model objects and are repeatedly normalized/moved; and `Model.to()` omits frame placements (`src/better_robot/data_model/model.py:92-119`). Those can produce allocation or transfer overhead on GPU.

But `model.joint_placements[j].to(device=q.device, dtype=q.dtype)` is a no-op when the model and `q` already match, not necessarily an H2D copy. `new_tensor(Python-list)` is an allocation/copy, but calling it an “implicit sync” requires a trace that the audit did not provide. CUDA is broken on the audit box, as the plan admits (`plan/03_architecture.md:229-233`), so the synthesis should classify H2D/synchronization impact as source-inspected, not measured. Hoisting constants is right; the causal/performance wording is overconfident.

### A10. Differentiable placements work unbatched; the claimed one-line batching fix does not

Replacing joint placements with grad-requiring tensors produced finite, nonzero gradients through unbatched FK, supporting `plan/01_assessment.md:147-167`. Replacing them with shape `(4, njoints, 7)` and running FK failed with a size-14-versus-4 mismatch. Thus the narrow empirical finding survives.

The architectural inference “all algorithms just index `[..., j, :]`” (`plan/03_architecture.md:171-179`) is incomplete. Current `Data` allocation derives batch shape from `q`, not model values; dynamics allocate/expand from that same shape; frame placements are Python records; and several algorithms rebuild spatial inertias inside loops. Indexing is one necessary edit, not the broadcasting, allocation, cache, compile, or serialization design.

### A11. Silent float64-to-float32 initialization: verified

`solve_ik` constructs initial values with `dtype=torch.float32` (`src/better_robot/tasks/ik.py:184-188`, `src/better_robot/tasks/ik.py:221-222`). This directly supports the assessment. The fix must be broader than preserving one input dtype: the library needs a declared supported-dtype and accumulation policy, covered in section C.

### A12. Robust LM acceptance uses the wrong objective: verified

The LM path weights residual/Jacobian for the normal equations but evaluates acceptance from the raw residual cost (`src/better_robot/optim/optimizers/levenberg_marquardt.py:90-117`). That is not the same robust objective or a consistently frozen IRLS surrogate. The M0 correction is justified, but the plan still has to define whether robust kernels apply per scalar, per 2-D observation, or per semantic residual group.

### A13. “Dead” compatibility code is not dead to the named consumers

The plan schedules deletion of `costs/` and deprecated aliases in M1 (`plan/03_architecture.md:261-272`, `plan/04_roadmap.md:25-35`). BetterHumanForce directly imports `better_robot.costs.stack.CostStack` (`../BetterHumanForce/scripts/motion/optimize_motion.py:210`, used at line 257) and uses the deprecated `Data.oMi` surface (`../BetterHumanForce/tools/robot_motion/playback.py:115`, `../BetterHumanForce/tools/robot_motion/playback.py:180-190`, `../BetterHumanForce/tools/robot_motion/motion.py:201`). Breaking changes are allowed, but calling these paths dead and deleting them before a replacement exists misstates the migration cost.

## B. Decisions attacked

### B1. Delete `backends/`: **sound-with-changes**

The present registry/protocol is weak: it routes at fine granularity, has only one substantive implementation, and complicates compilation. Whole-pass dispatch is the correct future granularity and is compatible with the vision's preference for functions and explicit values (`plan/02_vision.md:45-74`). The decision survives even though its headline performance number does not.

The proposed replacement seam is not ready. `forward_kinematics_raw(model, q)` still receives a Python `Model`, validates shapes, loops over Python tuples and `JointModel` objects, and invokes Lie operations through global dispatch (`src/better_robot/kinematics/forward.py:73-135`). RNEA accepts and mutates `Data` (`src/better_robot/dynamics/rnea.py:71-111`, `src/better_robot/dynamics/rnea.py:204-209`); ABA/CRBA have comparable object/data coupling. These are not yet flat tensor-in/tensor-out pass boundaries.

Reintroducing Warp would cost substantially more than “an `autograd.Function` and at most six `if`s” (`plan/03_architecture.md:44-51`). A credible whole-pass ABI must cover gradients with respect to `q`, batched/differentiable placements and inertias from M3, frame values, stream/device/dtype policy, higher-order-gradient capability, `torch.compile`/fake-tensor registration, unsupported-joint fallback, result/Data parity, and caching. Public dispatch can indeed be six conditionals; making the two implementations semantically interchangeable cannot.

Required change: delete the registry, but first name and test a backend-neutral `ModelStructure` plus tensor `PassInputs`/`PassOutputs` boundary. Keep one PyTorch implementation. Document a capability/fallback policy and require a second implementation to justify runtime selection. This prevents the M3 model design from making M5 prohibitively expensive without preserving today's speculative machinery.

### B2. Variable-block `Problem`: **unsound as specified**

The need for multiple parameter blocks is real; the proposed semantics are not sufficiently defined to implement safely.

1. **Bounds are attached to the wrong space.** `VarSpec.lower/upper` are described as tangent-space bounds (`plan/03_architecture.md:65-76`). Robot joint limits live in configuration coordinates with shape `nq`, while increments have shape `nv`; SO(3)/SE(3) have no origin-independent global tangent-space box. The current `JointPositionLimit` already exposes this mismatch by special-casing or zeroing tangent behavior for `nq != nv` (`src/better_robot/residuals/joint_limits.py:19-54`). The design needs separate state feasibility constraints/projection from trust-region or step bounds, plus feasible retraction and KKT termination semantics.

2. **Shape and ownership are ambiguous.** `shape` does not distinguish batch axes from event/manifold axes, shared parameters from per-frame parameters, or fixed/masked coordinates. A Boolean mask that leaves zero columns in a normal equation can make it singular; fixed variables should usually be eliminated or constrained with explicitly defined rows. Block scaling/preconditioning is absent, yet a camera translation, joint angle, pixel correction, log-scale, and contact force cannot responsibly share one global damping value without scaling.

3. **The residual/provider protocol is incomplete.** A `reads` list covers variables, not derived dependencies; providers need their own declared inputs/outputs and an acyclic dependency graph. A custom residual needs a concrete contract for residual shape, semantic grouping, auxiliary diagnostics, analytic Jacobian blocks, provider requests, batch reduction, and failure/status reporting. The sketch in `plan/03_architecture.md:78-102` is not enough for a consumer author to implement one without reading internals.

4. **It does not express all named consumer objectives.** BetterVideoReconstruction uses scalar/reduced objectives with auxiliary values and phase-specific terms (`../BetterVideoReconstruction/tools/human_optim/optimizer.py:503-563`, phases at `../BetterVideoReconstruction/tools/human_optim/optimizer.py:586-593`). Forcing every such term into an arbitrary residual vector changes weighting and robust semantics. Add a first-order `ObjectiveTerm` protocol or explicitly constrain the first-order solvers to least squares and explain how scalar objectives participate.

5. **Block Jacobians do not create sparsity.** A whole trajectory can remain one enormous `q` block with a dense Jacobian. Splitting every knot into a `VarSpec` makes the public API and provider graph unwieldy. The plan removes `ResidualSpec` while providing no replacement symbolic sparsity, banded structure, linear operator, Schur complement, or sparse solver milestone. Dense assembly remains the bottleneck for trajectory optimization.

   For dense problems, the proposed dictionary of Jacobian blocks may be slower than today's single dense Jacobian: repeated `jacrev` transforms can redo common provider work, while concatenation/scatter into the LM matrix adds Python and allocation overhead before producing the same dense `J` or `JᵀJ`. The plan promises that dense assembly “still works” (`plan/03_architecture.md:92-96`) but supplies neither an assembly algorithm nor a benchmark against the current path. Block structure is an API property, not a performance result.

6. **The proposed persistent cache is unsafe.** “Values version” is undefined, tensors remain mutable, and current `Data` explicitly warns that in-place mutation is not detected (`src/better_robot/data_model/data.py:17-19`). Caching provider outputs with attached graphs can retain old autograd graphs, leak memory, or make a second backward invalid. `torch.func` transforms are easiest over pure functions, not hidden mutable caches. Use an evaluation-local immutable `EvaluationContext`/pytree; cache only within one objective/Jacobian evaluation, and retain only detached accepted-state artifacts between iterations.

7. **Tangent autograd has hidden costs.** Evaluating `f(values ⊕ δ)` builds retraction and provider graphs every time. Separate `jacrev` calls per block can recompute the same providers; differentiating through an optimizer requires `create_graph`/double backward; and the current code is not even `jacfwd`-clean. The plan must choose `jacrev`, `jacfwd`, VJP/JVP, or analytic blocks according to dimensions, define graph lifetime, and test higher-order support rather than assuming tangent construction solves it.

Before freezing this API, implement one vertical slice from BetterVideoReconstruction or BetterHumanForce: at least two variable blocks, one shared derived provider, one custom residual, one scalar first-order term, masks, batching, and a phase transition. The “second caller” rule in `plan/04_roadmap.md:110-116` should apply here before—not after—the public abstraction is committed.

### B3. Optimizer redesign: **sound-with-changes**

Per-element damping and convergence are necessary for batched solves. The branchless story is muddled, however. Accept/reject does **not** inherently require evaluating both residual branches: the current residual is cached and one candidate residual is needed to decide. `torch.where` blends already computed tensors; it does not lazily evaluate Python branches. Evaluating a candidate Jacobian before acceptance would waste Jacobian/provider work for rejected elements, but that can be deferred to the next iteration. Conversely, converged elements still consume batched provider/linear-algebra work unless the solver compacts the batch or accepts that overhead.

Per-element damping is mathematically compatible with a batched solve only if every element has its own Hessian and the diagonal update is `mu[..., None, None] * I`. A single shared factorization would be wrong. `torch.linalg.cholesky` can also fail the whole batch when one element is indefinite; the plan needs `cholesky_ex` information masks, per-element fallback/status, and a policy for singular/invalid residuals.

The criticism of current bounds is fair in outcome but inaccurate in description, as A4 shows. Merely switching convergence to projected-gradient norm does not repair bad trial steps. Choose an actual bounded algorithm—active-set LM, reflective/trust-region least squares, or a documented projected method—and specify feasible retraction, active-set updates, predicted reduction, and KKT conditions. A clipped candidate with a projected-gradient stop is still the current crude scheme with a better status check.

Robust acceptance must use either the true robust objective or a consistently frozen IRLS surrogate and matching predicted reduction. Kernel grouping must be explicit. Madsen-style damping is reasonable only after variable scaling is defined; otherwise diagonal magnitudes across heterogeneous blocks dominate policy.

Matrix-free Adam is straightforward. Batched LBFGS with per-element histories and line searches is not: elements need different accepted step lengths, curvature validity, and history resets. Treat those as separate milestones. Finally, “implicit differentiation in about 100 lines” (`plan/03_architecture.md:142-146`) is unjustified: manifolds, robust/nonsmooth losses, active bounds, singular Hessians, optimized versus external parameters, and backend higher-order support are the difficult contract, not the linear solve boilerplate.

### B4. Parametric `Model`: **sound-with-changes**

Making placements and inertias differentiable values is essential for `better_human`. The `[..., j, :]` recipe is necessary but nowhere near sufficient.

- The execution batch must be the broadcast of `q` batch axes and every dynamic model-value batch. Today public FK allocates `Data` solely from `q.shape[:-1]` (`src/better_robot/kinematics/forward.py:168-189`). A batched model with unbatched `q`, or differently broadcastable placement/inertia batches, therefore has no defined behavior.
- The proposed frame table is written as `(nframes, 7)` (`plan/03_architecture.md:181-184`), omitting the very model batch axes M3 introduces. It should be `(*model_batch, nframes, 7)` or an equivalent broadcastable representation, with parent indices as device tensors and names/metadata kept separately.
- Topological Python loops are not inherently broken; they are static structure and can compile. What breaks is hiding changing tensors inside a guarded Python `Model` object. Separate immutable `ModelStructure` from a tensor pytree `ModelValues`, and pass values explicitly to compiled passes. Repeated `with_values` calls on a whole dataclass invite graph guards and recompilation.
- `.to(dtype=...)` must cast floating values but never joint/frame index or kind tensors. Current `Model.to()` already misses frames (`src/better_robot/data_model/model.py:92-119`), so device transfer needs an exhaustive, tested tree operation.
- M1's “precompute all 6x6 inertias once” conflicts with M3's differentiable, replaceable inertias. Precompute canonical 6x6 values at model construction only for static models; for optimized inertias, derive them once per evaluation context so gradients remain live and caches do not go stale. Current dynamics index/rebuild inertias inside loops (`src/better_robot/dynamics/rnea.py:167`, `src/better_robot/dynamics/aba.py:121`, `src/better_robot/dynamics/crba.py:60`).
- Mimic reduction must be part of the same structure/value coordinate map, not a late FK gather. Otherwise FK and dynamics disagree about `nq`, `nv`, torque accumulation, and limits.

### B5. Merge/delete `costs/`: **unsound in the proposed sequence**

One residual-composition layer is preferable to two near-duplicates, so merging the concepts may be fine. But M1 deletes a live public import before M2 supplies its replacement, and it deletes deprecated `Data` aliases used by a named consumer. That makes the intermediate milestone worse and prevents consumer parity testing. Keep a thin re-export/deprecation shim until the new `Problem` has a migration adapter, or coordinate the consumer changes in the same milestone. Also do not discard whatever structural information `ResidualSpec` contains until the sparse/structured replacement is designed.

### B6. Roadmap sequencing: **unsound**

M2 is not a realistic acceptance boundary. Its item (c) asks for BetterVideoReconstruction stage-1 parity/runtime (`plan/04_roadmap.md:54-62`), but the real stage is phased (`../BetterVideoReconstruction/tools/human_optim/stages.py:363-383`, invocation at `../BetterVideoReconstruction/tools/human_optim/stages.py:479-485`) and uses projection, chamfer, shared nearest-neighbor/SDF work, scalar objectives, and auxiliaries (`../BetterVideoReconstruction/tools/human_optim/losses.py:90-106`, `../BetterVideoReconstruction/tools/human_optim/losses.py:135-244`). The plan schedules projection/SDF/vision residuals in M4 and batched parametric models in M3. M2 cannot prove consumer parity without either secretly implementing later milestones or weakening “parity” until it is meaningless.

Other ordering failures:

- M0 promises “make the current library truthful” while leaving the headline batched task API and broken bounded solver until M2. M0 should fix them, or fail fast/document them as unsupported and remove the misleading switches.
- M1 breaks named consumers before a replacement exists.
- The consumer vertical slice happens after the variable/provider API is designed, contrary to the plan's own two-caller rule.
- There is no sparse/banded trajectory-solver milestone despite trajectory optimization being a core goal and dense block assembly being the predictable scalability wall.
- M2 combines a variable system, provider graph, batching, bounds, robust LM, four optimizer paths, phases, task rebases, and consumer parity. That is several independently risky milestones, not one checkable step.
- Implicit-differentiation semantics must be decided before M2 freezes residual, manifold, bound, and solver state contracts, even if implementation remains M5.
- Batched parity should use stated tolerances and statuses, not exact equality to sequential runs; per-element branch decisions will differ near thresholds. Runtime acceptance also needs a fixed hardware/dtype/problem benchmark.

A defensible order is: M0 correctness plus explicit unsupported guards; M1 model structure/value and pure pass boundary; M2a minimal variable/evaluation protocol proven by one real consumer slice; M2b one robust batched LM with correct bounds; M2c matrix-free first-order methods; then sparse trajectory structure, parametric breadth, and full consumer migrations. Module deletion can occur when its replacement lands.

## C. Missing considerations

1. **Threading, reentrancy, and multiprocessing.** A mutable global/persistent provider cache is unsafe under concurrent solves. `Model` is frozen only shallowly and contains mutable dictionaries/metadata (`src/better_robot/data_model/model.py:44-46`, `src/better_robot/data_model/model.py:84-88`). Specify per-call `Data`/evaluation ownership, cache scope, thread safety, multiprocessing-spawn behavior, and CUDA stream semantics. Compiled-graph caches also need bounded, thread-safe ownership.

2. **Serialization and checkpoint compatibility.** A current Panda `Model` happens to pickle and run FK after unpickling, but that is not a stable contract; its metadata retains builder IR/resolver objects (`src/better_robot/io/build_model.py:591`). Deleting the only IR/schema without a replacement leaves no versioned interchange. Define a versioned structure/value `state_dict`, reconstruction metadata, optimizer/warm-start state, map-location behavior, and an explicit stance on pickle.

3. **Numerical dtype and units policy.** The code has an exception claiming half precision is unsupported (`src/better_robot/exceptions.py:55-60`) but does not consistently enforce it. State supported dtypes, mixed-precision rules, accumulation/factorization dtype, default tolerances by dtype, small-angle cutoffs, robust-kernel scale units, variable scaling, and deterministic behavior. “Preserve input dtype” alone is not a policy.

4. **Sparse trajectory solvers.** Dense Jacobian-block assembly does not scale to long horizons. The architecture needs structural sparsity, JVP/VJP linear operators, banded/block-tridiagonal solvers, and perhaps Schur elimination for camera/body nuisance blocks. Without that, the redesign still cannot credibly claim trajectory optimization as a first-class use case.

5. **Quaternion double cover and continuity.** Rotation values satisfy `q ~ -q`. Priors, temporal residuals, interpolation, and learned/body-model parameters need hemisphere alignment or a sign-invariant metric, plus a stated convention around the log-map discontinuity at pi. Canonicalizing by scalar-part sign avoids some duplicates but introduces its own discontinuity; it is not a complete temporal policy.

6. **Migration as a deliverable.** The plan names consumers but has no symbol-by-symbol migration table, compatibility window, conversion adapter, or coordinated commit order. BetterHumanForce already uses `CostStack` and `Data.oMi`; BetterVideoReconstruction has its own phased scalar-objective engine and vectorized motion code (`../BetterVideoReconstruction/tools/human_optim/motion.py:19-54`). “Delete their engines” is an outcome, not a migration plan.

7. **Licensing and provenance.** The root repository has no visible top-level license, while code the roadmap proposes to “port” is under licenses that impose notice/provenance obligations—for example PyRoki's MIT license (`references/kin_dyn/pyroki/LICENSE`) and JAXopt, MuJoCo Warp, and Newton's Apache-2.0 licenses. Maintain a source ledger, preserve notices, distinguish algorithm reimplementation from copied code, and decide BetterRobot's license before wholesale ports.

8. **Differentiation contract.** Decide whether public guarantees cover gradients with respect to state, placements, inertias, residual parameters, and solver hyperparameters; first or second order; unrolled or implicit solves; active bounds and nonsmooth robust kernels; and failed/nonconverged solves. “PyTorch-native” is not a precise AD contract.

9. **Compilation lifecycle.** Dynamic batch/time shapes, phase-dependent residual sets, Python model replacement, and changing provider graphs can all recompile. Define supported dynamic dimensions, graph-cache keys/limits, cold-start expectations, and whether phase changes use fixed-shape zero weights or separate compiled programs.

10. **Failure semantics.** Batched APIs need per-element status for NaN residuals, factorization failure, infeasible bounds, stagnation, active-set stationarity, maximum iterations, and callback/user abort. A single enum for the whole batch would recreate today's misleading status problem.

## D. Number checks

| Plan number | Check | Verdict |
|---|---|---|
| 19 µs / 27% backend dispatch | Direct facade-versus-implementation timing was 1.75-2.01 µs, 2.3-3.0%; a second raw audit independently says ~1 µs / 3%. | **Refuted.** The raw reports conflict and the synthesis selected the larger result without reconciliation. |
| 5x `torch.compile` FK speedup | Fixed Panda, batch 256: eager ~2.91 ms, steady compiled ~0.269 ms, about 10.8x. First compile took ~31.5 s. | **Plausible but underspecified.** The ratio is profile-sensitive; cold compile amortized only after roughly 12,000 calls in this probe, and object/shape changes may recompile. A bare “5x” is not a product guarantee. |
| 42x analytic versus FD | Analytic ~1.59 ms, actual `jacrev` ~13.99 ms, FD ~62.0 ms: ~39x analytic/FD and ~8.8x analytic/autodiff. | **Plausible for this exact analytic-vs-central-FD case, mislabeled if generalized to autodiff.** Specify residual, `nv`, dtype, mode, warmup, and transform. |
| 2.4x `Model.integrate/difference` loop slowdown | For an SMPL-like 24-joint, 200-frame difference, current code was ~3.05 ms versus ~0.832 ms for the consumer's vectorized twin, ~3.66x. | **Plausible and sensitive.** The plan incorrectly phrases `2.4-10x` as one measured BetterVideoReconstruction range: the raw audit measured 2.4x in one setup, while 4-10x was a consumer claim for another setup and was not independently verified. |
| “At most six `if`s” to restore Warp | Counts only public call sites, not semantic parity, custom backward, dynamic model gradients, compilation registration, fallbacks, or result conversion. | **Not a useful cost estimate.** Six dispatch sites can hide months of backend work. |
| “Implicit differentiation in ~100 lines” | No implementation/probe supports it; bounds, manifolds, nonsmooth losses, singularity, parameter partitioning, and failure semantics are omitted. | **Overreach.** Estimate only after the differentiation contract and one end-to-end prototype exist. |

Performance assertions in CI should compare stable benchmark distributions on named hardware and catch regressions, not encode attractive ratios from one workstation. Correctness CI should not depend on speedup ratios at all.

## E. Overall verdict and five required changes

**Overall verdict: revise before implementation.** The plan identifies several real defects: zero-gradient NaNs, the non-batched task stack, fake autodiff strategies, unused mimic metadata, dense first-order solvers, inconsistent robust acceptance, and the need for parametric model values all survive attack. The broad direction—PyTorch-first passes, whole-pass backend boundaries, multi-variable optimization, and differentiable model values—is defensible.

It is not yet an execution-grade architecture. Two load-bearing factual stories are wrong or overstated (bounded-LM mechanics and 19 µs dispatch cost), the `Problem` sketch lacks enough constraint/shape/evaluation semantics to serve the named consumers, the future backend seam is asserted rather than designed, and M2's acceptance depends on capabilities scheduled for M3/M4. The roadmap also deletes live consumer surfaces before replacing them and omits the sparse solver needed for trajectory-scale problems.

The five changes I would insist on are:

1. **Publish factual errata and reproducible benchmarks.** Correct the LM clamp/acceptance explanation, `2nv + 1` FD count, dispatch measurement conflict, autodiff labeling, H2D certainty, and compile cold-start context. Check in benchmark/probe definitions with hardware, dtype, shapes, warmup, and statistical method before using ratios as acceptance gates.

2. **Replace the `Problem` sketch with a formal executable contract.** Separate state constraints from tangent step bounds; define batch/event/shared axes, fixed variables, scaling, residual groups, scalar objective terms, provider DAGs, evaluation-local caching, AD transform/graph lifetime, auxiliary outputs, and per-element failure status. Validate it with a real consumer vertical slice before freezing the API.

3. **Design `ModelStructure`/`ModelValues` and a pure pass ABI together.** Specify broadcasting across `q` and model-value batches, batched frames, exhaustive device/dtype transfer, versioned serialization, dynamic-inertia cache rules, and reduced mimic-coordinate semantics across FK, Jacobians, and all dynamics. Make these explicit tensor inputs/outputs the tested seam a future Warp implementation would target.

4. **Reorder and split the roadmap.** M0 must either repair or explicitly reject misleading batch/bounds/mimic modes. Land compatibility adapters with replacements, not a milestone earlier. Build a two-block real-consumer slice before the generic API, split batched LM from first-order/LBFGS work, move minimal parametric-model support ahead of consumer parity, and add a separate sparse trajectory milestone.

5. **Add a cross-cutting engineering contract.** It must cover dtype/units/numerical tolerances, quaternion sign/continuity, threading/reentrancy and cache ownership, serialization/checkpoints, compile lifecycle, differentiation guarantees, migration sequencing, and third-party license/provenance. These are architecture constraints, not cleanup to discover after M5.
