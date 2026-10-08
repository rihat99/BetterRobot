# P1 — optim: adversarial architecture review

Scope: BR `src/better_robot/optim` (1 773 LOC), `src/better_robot/terms` (325 LOC), BH `src/better_human/terms.py`
(242 LOC). Worktree heads BR `23070f4`, BH `bad0de3`. All paths below are relative to BR unless prefixed `BH`.
Runtime probes: CPU, `uv run python -c ...`, 2026-10-08 (numbers quoted in F1, F2, F5, F4).

## Current state (facts, with paths)

- **Problem** (`optim/problem.py:115`): named variables `(B, *event)`, optional per-variable `manifolds`, `bounds`,
  `data`, `time=(...)` for trajectory variables, `dense=` override. Terms are plain functions whose parameter names are
  their reads (`_resolve_reads`, `problem.py:404`); `TermFn` adds analytic `jac` or a time `window`
  (`problem.py:38`). Two term kinds only: `residual` and `scalar` (`problem.py:192,222`). `cached(...)` shares an
  expensive intermediate (FK state) between terms (`problem.py:181`). `with_(weights=, free=)` makes views
  (`problem.py:237`).
- **Linearization** (`problem.py:282-341`): one traced evaluation at `delta=0` in the tangent space. Analytic `jac`
  terms bypass autograd; windowed terms get per-window Jacobians; all other residuals are stacked and differentiated
  with one `autograd.grad(..., is_grads_batched=True)` over an identity of size M (one reverse pass per residual row).
  Scalar terms add their gradient to `g` but nothing to `H`. Robust kernels enter by IRLS (`optim/kernels.py`).
- **Normal-equation backends** (`optim/linear.py`): `Normal` (dense), `BandedNormal` (block-tridiagonal over
  knots, own banded Cholesky `_factor/_substitute`, `linear.py:144-180`), `ArrowNormal` (band bordered by shared
  variables, Schur on the border, `linear.py:206-294`). Built through a sink seam: `DenseSink/BandSink/ArrowSink` with
  `add(columns_a, columns_b, values)` + `finish(g)` (`linear.py:296-388`). The layout is inferred
  (`_band_layout`, `problem.py:466`).
- **Solvers** (`optim/solvers.py`): `Solver` protocol `init(problem, x) -> State`, `step(problem, state) -> State`
  (`solvers.py:30`); `State(x, loss, done, it, extra)` (`solvers.py:20`). `LevenbergMarquardt` (Nielsen damping,
  per-element accept/reject, `solvers.py:80-106`), `GaussNewton`, `LBFGS`, `TorchOptim` (any `torch.optim` class).
  Box bounds become tangent step bounds via `Normal.restrict` (projected/clamped, not an active-set QP).
- **Drivers** (`optim/phases.py`): `solve`, `run(problem, [Phase...])` (weight overrides + frozen coordinates per
  phase), `solve_layer` = implicit differentiation by one detached-H Newton step, rejects scalar terms
  (`phases.py:85-118`). `Result(x, loss, done, iters, history)` where `history` holds only per-phase losses
  `(steps, B)` (`phases.py:38-46,73`).
- **Constraints** (`optim/constrained.py:28`): `solve_constrained` = outer augmented-Lagrangian loop around any
  solver, over residual terms named as equalities/inequalities; multipliers and penalty are local variables and are
  not returned; `history["violation"]` is `(rounds, B)`.
- **Manifold** (`optim/manifold.py:15`): protocol `tangent_dim/retract/project/tangent_bounds`; `Euclidean` in optim,
  `RobotManifold` in `terms/robot.py:21` (wraps `Model.integrate`).
- **Robot terms** (`terms/robot.py`, `terms/ik.py`, `terms/collision.py`): factories returning `TermFn`
  (`frame_pose`, `frame_position`, `joint_limits`, `rest`, `finite_difference(order, dt)`, `sphere_clearance`,
  `self_clearance`), `ik_problem`, `solve_ik(..., restarts=)`. Variable/cache names are string kwargs
  (`q="q"`, `state="state"`).
- **BH terms** (`BH src/better_human/terms.py`): same factory pattern (`human_state`, `joints`, `points`,
  `landmarks`, `reprojection`, `ground`, `contact`, `pose_prior`, `shape_prior`, `contact_wrenches`, `root_wrench`).
  BH reuses BR optim without forking it — good coherence.
- `optim` imports only torch (`optim/__init__.py` docstring; enforced by `tests/contract/test_layer_dependencies.py`).

## How references evolved (with sources)

- **Theseus** (`references/design/theseus.md:14-18`, `references/optim/theseus/theseus/optimizer/linear/`):
  kept the cost-weight/whitened-residual core; the parts the digest calls regrettable are global registries,
  attribute-held damping state, "everything is 2D" batching, a heavy C++/CUDA sparse extlib
  (`baspacho_sparse_solver.py`, `lu_cuda_sparse_solver.py`, CHOLMOD) and vmap-jacrev as default Jacobian path.
  They added `Vectorize` (`core/vectorizer.py`) later to batch structurally identical cost functions because
  per-cost autograd was the bottleneck, and four backward modes (unroll / implicit / truncated / DLM,
  `theseus_layer.py`, digest l.370) because one implicit mode was not enough for learning pipelines.
- **jaxopt** (`references/design/jaxopt.md:7-12,95-138`; `references/optim/jaxopt/jaxopt/_src/implicit_diff.py:33,256`):
  solvers are frozen dataclasses with `init_state/update/run`, warm start by passing a previous `OptStep`;
  implicit differentiation is generic (`custom_root` over any optimality function incl. KKT), so constrained
  solvers (OSQP, projected gradient) are differentiable with the same machinery. Now in maintenance mode; its ideas
  moved into optax/optimistix.
- **optax** (`references/design/optax.md:7,56-114`): one protocol `(init, update)` + `chain`; extension by an
  extra-args channel rather than new protocols. Our `TorchOptim` adapter already covers this ground in torch.
- **Crocoddyl** (`references/design/crocoddyl.md:15-32,55,230`; `references/optim/crocoddyl/CHANGELOG.md`):
  `ActionModel`/`ActionData` split, `core/` never includes Pinocchio. Solver family grew DDP → FDDP (feasibility,
  multiple shooting) → Box-FDDP (control bounds) → Intro/KKT/Ipopt adapter/ODYN-SQP; constraints, residual models and
  terminal dimensions were retrofitted (PR #1269 "notion of terminal dimension, residuals and constraints"), and the
  Ipopt adapter exists as the "escape hatch for hard problems" (digest l.230). Lesson: constraints arrived late and
  forced API growth across every action model.
- **pyroki / jaxls** (`references/design/pyroki.md:128-153`): PyRoki has no solver; it delegates to jaxls
  (sparse least squares over typed `Var`s, trust region, CG or Cholesky) and later added augmented-Lagrangian
  constraints (limits, poses, velocity, swept-volume collision) as first-class cost kinds.
- **cuRobo v2** (`references/design/curobo.md:3,29,65,107`): rewrote v1 around two protocols (Optimizer, Rollout)
  that compose (`MultiStageOptimizer`), flattened import paths, CUDA graphs in `runtime.py`; particle seeding +
  L-BFGS on GPU.
- **Drake MathematicalProgram** (`references/design/drake.md:46-48`): decision variables + `Binding<Cost|Constraint>`
  objects; the program is solver-agnostic and a solver is chosen per program class (LP/QP/SOCP/NLP). Constraint
  type is declared, never inferred from usage.
- **mink**: IK as a per-step QP over velocities with limits as inequality constraints (BH `references/retarget`);
  shows that constraint-as-declaration is what retargeting users expect.

## Findings (ranked, F1..Fn: severity, evidence, consequence)

**F1 — high. Sparsity is inferred and only three fixed shapes exist; a plain trajectory term silently goes dense.**
Evidence: `Problem({'q': (4,T,7)}, time=('q',))` with an un-windowed residual returns `Normal` (dense):
T=32 linearize 2.1 ms, T=256 76 ms, and the solve is a dense Cholesky of 1792×1792 per batch element. Banded/arrow
layouts appear only when every time-coupled term declares `window` (`_band_layout`, `problem.py:466`). There is no
general block-sparse backend (multi-variable factor graphs: multi-person or multi-camera fitting, several robots,
contact-rich trajectories with per-knot contact variables). Consequence: the first BVR-scale problem
(many bodies × frames × cameras) needs a new backend; today that means a 4th hand-written `*Normal` + `*Sink`.
Good news: the sink seam (`add(columns_a, columns_b, values)` / `finish(g)`) is exactly the right brick boundary
for a `BlockSparseSink`.

**F2 — high. Default autograd Jacobian costs one reverse pass per residual row.** Evidence: M rows on an
elementwise residual, B=8, CPU: M=64 1.4 ms, M=512 5.8 ms, M=2048 166 ms (superlinear). `linearize` always uses
reverse mode (`is_grads_batched` over `eye(M)`) even when `tangent_dim ≪ M` (BH vertex/point fitting: thousands of
rows, ~70–200 tangent columns), where forward mode (`tangent_dim` JVPs) is far cheaper. Theseus hit the same wall
and added `Vectorize`. Consequence: BH fitting and dense point terms are slow by construction unless every term
author writes `jac`.

**F3 — high. Constraints are not part of the Problem.** Evidence: term kinds are `residual|scalar` only;
`solve_constrained` (`constrained.py:28`) re-labels existing residuals per call, keeps λ and ρ local and does not
return them; `Result` has no dual or KKT fields; `solve_layer` cannot differentiate a constrained solution; bounds are
clamped steps, not an active set. Consequence: SQP, interior point, Box-DDP, QP-IK (mink style) cannot be added as
new solvers behind the same `Solver` protocol because the problem cannot say what is a constraint; MPC cannot
warm-start multipliers between ticks. Crocoddyl retrofitted this late and paid across all action models.

**F4 — medium. Scalar terms are first-order only inside GN/LM.** Evidence: `linearize` adds the scalar gradient to
`g`, never to `H`; `predicted_decrease` therefore ignores scalar curvature, so the LM gain ratio is biased (probe:
residual + `10·Σx⁴` scalar converged but needed 22 LM iterations on a 2-D problem). `solve_layer` correctly rejects
scalars. Consequence: users mixing a smooth cost (energy, effort) with residuals get a silently weaker solver; there
is no way for a term to supply curvature (`hess` or GN-factorizable form).

**F5 — medium. Batch semantics: one flat leading axis, no compaction.** Evidence: variables `(2,3,4)` give
`loss` shape `(2,)` — the `(3,4)` became event, while the BR tensor contract says "arbitrary leading batch axes"
(BR `CLAUDE.md`, `docs/contracts/02-tensors.md`). `done` is per element but `LevenbergMarquardt.step` still linearizes
and solves the full batch (`solvers.py:80-106`). Consequence: multi-start IK (`solve_ik(restarts=)`) and large
seed batches pay for converged seeds; users with `(cameras, people, ...)` batches must flatten by hand. This is an
undocumented exception to a contract — fix the doc or the behavior before merge.

**F6 — medium. No warm-start or solver-state handoff in the result.** Evidence: `Result(x, loss, done, iters,
history)` drops `State.extra` (LM damping, L-BFGS memory) and constraint multipliers; `history` holds losses only,
no gradient norm, step norm, damping, accept ratio or termination reason (`phases.py:38-73`). Consequence: MPC and
streaming fitting (BVR frame-by-frame) restart cold each tick; convergence debugging needs custom loops.

**F7 — medium. No trajectory-optimization brick.** Evidence: no shooting/DDP/iLQR; the banded GN with
`finite_difference` terms is direct transcription with dynamics as soft penalties. `BandedNormal`'s block-tridiagonal
Cholesky (`linear.py:144-180`) is already the Riccati-equivalent factorization, so multiple-shooting GN with
dynamics as equality constraints is one brick away once F3 exists. Consequence: legged/humanoid users go to
crocoddyl today.

**F8 — low-medium. One differentiable-solve mode.** `solve_layer` = implicit, GN `H`, unconstrained, converged
solutions only (`phases.py:85-118`). Theseus needed unroll/truncated/DLM; jaxopt generalizes via `custom_root`
over KKT. Consequence: learning pipelines with partial convergence or constraints have no option.

**F9 — low. String-name namespace for composition.** Every factory takes `q="q"`, `state="state"` kwargs (BR
`terms/robot.py`, BH `terms.py`). Composing BR robot terms and BH body terms in one Problem (human + robot,
two people) relies on users renaming caches by hand; collisions surface as errors only if `_require_new_name` fires
on terms, not on reads. Consequence: friction, not a wall. A naming convention (prefixing) is enough.

**F10 — low. Bounds are clamps, not constraints.** `Normal.restrict` clamps the step per coordinate; correct for
LM with boxes, but no active-set handling for general linear inequalities (`J_c Δ ≤ b`), which QP-IK needs.

Not findings (verified good): `optim` imports only torch; the `Solver` protocol is the jaxopt/optax shape
(value-typed `State`, `init/step`); `TorchOptim` gives torch.optim interop; `check_jacobian` exists; BH reuses BR
optim without duplication.

## Roadmap (table: brick | why | prerequisite | when)

| Brick | Why (user) | Prerequisite in architecture | When |
|---|---|---|---|
| Doc fix: optim batch = one leading axis, list exception in tensor contract | contract honesty (all) | none | now |
| Return solver state + duals in `Result` (`state: State`, `multipliers`), accept them back (`x=`, `state=`) | MPC, streaming fitting (BVR) | `Solver.init` accepting an optional prior state | now |
| Per-step diagnostics in `history` (`grad_norm`, `step_norm`, `damping`, `accepted`) and a `reason` per element | convergence debugging (all) | solvers write into `State.extra` under fixed keys | now |
| Warn (once) when a `time=` problem falls back to dense | trajectory users | `_band_layout` already knows | now |
| Forward-mode Jacobian when `tangent_dim < M` (`torch.func.jvp`/`jacfwd` over tangent columns) | BH fitting, point terms | traced evaluation is already a pure function of `delta` | next |
| Batch compaction: step only `~done` elements, scatter back | multi-start IK, large seed batches | `select` helper (`solvers.py:44`) generalised to index-select | next |
| Constraint term kinds on `Problem` (`problem.equality(...)`, `problem.inequality(...)`) with `solve_constrained` reading them | QP-IK, SQP, Box-DDP, retargeting with limits | F3; AL loop already works over residuals | next |
| `BlockSparseSink` + block-sparse Cholesky / PCG (pure torch, `torch.sparse` BSR or own blocks) | multi-body/multi-camera BVR fitting, factor graphs | sink seam exists; needs per-term column groups instead of time windows | next |
| Schur complement for "landmark-like" variables (point/shape blocks) | bundle-adjustment style fitting (shape shared across frames) | `ArrowNormal` generalised from one border to N block borders | later |
| QP subproblem solver (active set / ADMM, batched) behind `Normal.restrict` | QP-IK (mink), SQP inner loop, WBC | constraint kinds; linear inequality rows in the normal sink | later |
| SQP / interior point solver implementing `Solver` | hard-constrained NLP | constraint kinds, duals in `State` | later |
| Shooting problem + DDP/FDDP/iLQR (`optim.shooting`: `ActionFn(x,u)->x'`, residual per knot) | legged/humanoid trajopt, MPC | constraints (dynamics), warm start, `BandedNormal` Riccati reuse | later |
| Term curvature hook (`TermFn(hess=...)` or scalar → GN split) | effort/energy costs with LM | F4 | later |
| More differentiable modes for `solve_layer` (unroll, truncated, KKT-implicit for constraints) | learning pipelines | duals in `Result`; constraint kinds | later |
| `torch.compile`/CUDA-graph friendly fixed-shape step | real-time MPC, GPU batches | no Python branching on tensor values inside `step` (now: `bool(...all())` in drivers) | maybe |
| Optional external backends (CHOLMOD/cuDSS, Ipopt adapter) as lazy bricks | very large sparse NLP | lazy-import boundary rule (BR CLAUDE.md) | maybe |

## Prep changes recommended now (each ≤ 1 day, non-breaking)

1. **Result carries the final solver state.** Add `state: State | None = None` (or `extra: dict`) to `Result` and
   let `solve`/`run` accept `state=`; `solve_constrained` stores `multipliers` and `rho` in `Result.history` or a new
   field and accepts them back. Default behavior unchanged. Opens MPC warm start and SQP later.
2. **Fixed diagnostic keys.** Each solver writes `grad_norm`, `step_norm`, `accepted` (and `damping` for LM) to
   `State.extra`; `run` stacks them into `history[f"{phase}.{key}"]`. Pure addition.
3. **Dense-fallback warning.** In `linearize`, if `time` is set but `_band_layout` returns `None`, emit a
   `warnings.warn` naming the un-windowed terms (once per Problem). Turns F1's silent cliff into a message.
4. **Document the batch rule.** State in `optim/__init__.py`, `optim/CLAUDE.md` and `docs/contracts/02-tensors.md`
   that optim variables have exactly one leading batch axis; give the reshape recipe. (Or decide F5 differently — see
   questions.)
5. **Reserve the constraint vocabulary.** Document (not implement) that future term kinds are
   `residual | scalar | equality | inequality` and that `solve_constrained(equalities=, inequalities=)` will become
   the fallback for terms declared with those kinds. Prevents a second spelling from appearing in BH first.
6. **Name the sink seam.** Mark `DenseSink/BandSink/ArrowSink` (`linear.py:296-388`) as the extension point for new
   normal-equation backends in `docs/contracts/08-extension-points.md`, with its two-method contract.

## Open questions for the owner

1. Batch semantics (F5): keep one leading axis in optim and document it, or support arbitrary leading axes by
   flattening at the `Problem` boundary? (Recommend: document now, flatten later — it is cheap and non-breaking.)
2. Constraints (F3): should constraints become term kinds on `Problem` (Drake-style declaration), or stay a
   driver-level choice (`solve_constrained`)? This decides whether SQP/QP/DDP are `Solver`s or separate drivers.
3. Sparse backend: pure torch (own block Cholesky / PCG, slower, no dependency) vs lazy optional CHOLMOD/cuDSS
   bricks? Theseus regretted the heavy extlib.
4. Trajectory optimization: is DDP/MPC in scope for BR (crocoddyl parity) or is direct transcription with banded GN
   enough for the near-term BH/BVR users?
5. Should `solve_ik` and BH fitting default to forward-mode Jacobians once available, or keep reverse mode for
   predictability?
