# Scan: BetterRobot optim / residuals / tasks — usability and design audit

Scope read: src/better_robot/optim (3,519 lines .py: lm 945, problem 751, variables 594,
implicit 458, temporal 451, solvers 297, optimizers 293, kernels 118, utils 44, __init__ 68),
residuals (3,113 lines), tasks (1,379 lines), guides custom_residual (169),
own_your_optimization_loop (145), staged_fit (136). All paths below are relative to
`src/better_robot/` unless they start with `docs/`. Read selectively (grep + key spans);
claims marked (unverified) were not read in full.

## 1. Concept inventory

`optim.__all__` exports about 43 names (optim/__init__.py:4-20). Grouped:

Modelling (what a user writes against)
- `Variable` optim/variables.py:56 — named tensor that owns its current value; `trainable`, `batch_ndim`, `time_axis`, `bounds`; `retract`/`difference` (250, 275).
- `SO3Variable` :307, `SE3Variable` :333 — manifold variables (nq != tangent_dim).
- `RobotVariable` :400 — q of a `Model`; joint groups, freezing, joint bounds (`frozen_groups` 462, `tangent_groups` 466, `tangent_weight` 515, `joint_bounds` 543).
- `Bounds` :24 — box bounds on state coordinates.
- `Residual` residuals/base.py:126 — ABC; ctor takes `*variables, dim, weight, row_weight, reduce, kernel, group_size, name, enabled` (128-139); user implements `error()` with no arguments (218), optional `jacobian()` (221), `active_groups()` (225).
- `residual` decorator base.py:292 — wraps `fn(*tensors)` into `_FunctionResidual` (234); requires `dim=`.
- `ScalarCost` base.py:248 — scalar f>=0 turned into row `sqrt(2f)`; makes problem implicit-ineligible (base.py docstring ~258).
- `Difference` base.py:329 — manifold difference residual.
- `Weight` base.py:17, `ScaleWeight` :48, `DiagonalWeight` :77 — row weights with `apply`/`apply_jacobian`.
- `Node` residuals/nodes.py:12 — memoized shared sub-computation (`compute`, `value`, `merge_key` 46); `RobotState` :93 and `robot_state()` :111 for shared FK.
- `TimeIndexedResidual` residuals/temporal.py:16, `TemporalPattern` residuals/structure.py:9 — banded temporal Jacobian blocks.
- Robust kernels: `RobustKernel` protocol optim/kernels.py:17 (`rho`, `weight` on squared norm), `L2` 29, `Huber` 39, `Cauchy` 70, `Tukey` 85, `GemanMcClure` 100.

Assembly / evaluation
- `Problem` optim/problem.py:64 — list of residuals; harvests variables + nodes at first use (`_freeze` 100-187, 88 lines); `update(values)` 291, `error` 485, `objective` 498, `term_costs` 503, `gradient` 510, `jacobian_blocks` 607, `dense_jacobian` 684, `retract` 710, `difference` 718, `structured_normal` 727.
- `JacobianStrategy` problem.py:22 = auto|analytic|jacrev|jacfwd|finite_difference; `AutodiffFallbackWarning` :27.
- Temporal: `TemporalAnalysis` temporal.py:33, `LinearizationReason` :21, `BlockBandedMatrix` :62, `StructuredNormal` :302.

Optimizers
- `Optimizer` ABC optimizers.py:41 (`step` 67 abstract, `optimize` 70, `reset` 90, `resume` 94); `OptimizerInfo` :29, `OptimizerStatus` :18.
- `LevenbergMarquardt` lm.py:409 (16 ctor params, 412-428), `GaussNewton` lm.py:939 (LM subclass); `LinearizationMode` lm.py:28 (auto|dense|structured), `LinearizationDecision` :34.
- `TorchOptimizer` optimizers.py:105 — wraps any `torch.optim` class over tangent `nn.Parameter` buffers.
- Linear solvers: `LinearSolver` / `InformativeLinearSolver` protocols solvers.py:35/45, `Cholesky` :131, `BandedCholesky` :181, `LinearSolveResult` :23, `LinearSolveStatus` :15, `LinearSystem`.
- Implicit diff: `ImplicitDiffConfig` implicit.py:56, `ImplicitDifferentiationError` :40.

Tasks: `solve_ik` tasks/ik.py:172 (+ `IKCostConfig` 57, `OptimizerConfig` 74, `IKResult` 94), `solve_trajopt` tasks/trajopt.py:138.

Concepts a user must learn:
- (a) plain least squares on a tensor: 5 — `Variable` (twice: one trainable, one `trainable=False` if the data will change), `residual` + `dim`, `Problem`, `LevenbergMarquardt`, read result from `variable.tensor` (not the return value).
- (b) IK with a custom cost: 9-10 — `RobotVariable` (+`batch_ndim`), `PoseResidual` (its base `_KinematicResidual` ctor has all-required kw-only args `frame, frame_id, target, knot, dim, weight, row_weight, kernel, name`, pose.py:69-81; public subclass defaults unverified), `RestResidual`, `JointPositionLimit`, custom `Residual`/`residual`, `Node`/`robot_state` if the custom cost needs FK and should share it, `Problem`, `LevenbergMarquardt`, `OptimizerInfo`. `solve_ik` has no hook for an extra residual (signature ik.py:172-180: `model, targets, initial_q, cost_cfg, optimizer_cfg, differentiable`), so a custom cost means rebuilding the recipe by hand.
- (c) torch.optim.Adam: 5-6 — `Variable`, `residual` (+`dim`) or `ScalarCost` for a non-squared loss, `Problem`, `TorchOptimizer`, `OptimizerStatus`/`resume`/`reset` for phases.

## 2. Minimal user code (from the real API)

(a) Least squares (verbatim shape of docs/guides/own_your_optimization_loop.md:13-56):
```python
theta = Variable(torch.zeros(2, dtype=torch.float64), name="theta")
obs = Variable(y, name="observations", trainable=False)
@residual(theta, obs, dim=4, name="line_error")
def line_error(p, measured):
    return p[..., 0:1] * x + p[..., 1:2] - measured
problem = Problem([line_error])
LevenbergMarquardt(problem, max_iterations=20).optimize()
theta.tensor   # result lives in the variable
```
8 lines. Boilerplate: `dim=4` (must equal output width; changes if len(x) changes), wrapping data in a `Variable(trainable=False)` purely so `problem.update({"observations": ...})` can name it (guide lines ~95-100: a bare tensor is a frozen construction-time constant), result read by side effect. Surprising: with default `jacobian_strategy="auto"` and no analytic `jacobian()`, Problem emits `AutodiffFallbackWarning` (problem.py:635-640) although the custom-residual guide says autodiff "is the default" (docs/guides/custom_residual.md:151).

(b) IK + custom cost (kwargs of PoseResidual abbreviated; unverified defaults):
```python
q = RobotVariable(model, q0, name="q", batch_ndim=1, bounds=True)
terms = [PoseResidual(q, frame="hand", target=T, ...),
         RestResidual(q, ...), JointPositionLimit(q, ...)]
state = RobotState(q)  # or robot_state(q); only needed to share FK with Pose terms
class MyCost(Residual):
    def __init__(self, state): self.state = state; super().__init__(state.q, dim=3, name="my")
    def error(self): return f(self.state.value())
terms.append(MyCost(state))
problem = Problem(terms)
info = LevenbergMarquardt(problem, max_iterations=100).optimize()
```
~12-15 lines vs 1 for `solve_ik`. Heavy: the recipe is all-or-nothing; the user re-derives the weights that `IKCostConfig` hides; sharing FK requires learning the Node lifetime rules (custom_residual.md:134-151: retain children, pass them to `Node.__init__`, call `child.value()`, never keep memos across epochs). Using `self.state.value()` vs calling FK inside a plain `@residual` silently changes cost (duplicate FK) but not results.

(c) Adam:
```python
theta = Variable(init, name="theta")
@residual(theta, dim=N)
def loss_rows(t): return model_out(t) - target
problem = Problem([loss_rows])
TorchOptimizer(problem, torch.optim.Adam, lr=1e-2, max_iterations=500).optimize()
```
5-6 lines. Heavy: a scalar loss must become `ScalarCost(fn, theta)` -> rows `sqrt(2f)` (base.py:248-290), with GN columns `grad f / sqrt(2f)` blowing up near 0 and implicit diff disabled. The user cannot write the idiomatic `loss = problem.objective(); loss.backward(); opt.step()` over `theta.tensor` in a documented way: `TorchOptimizer` owns private tangent buffers `_buffers: dict[str, nn.Parameter]` (optimizers.py:131) and rebases them each step (`base = {...variable.tensor.detach()}` 247). `torch.optim.LBFGS` uses one closure over the summed batch objective, coupling batch elements (docstring optimizers.py:113-114).

## 3. Jacobians

- Entry: `Problem.jacobian_blocks(strategy, create_graph, fd_eps)` problem.py:607-683 returns `dict[(residual_name, variable_name)] -> block`. Per (residual, variable):
  analytic = `Residual.jacobian()` tuple in tangent coordinates (base.py:221); if None and strategy auto -> `_ad_block` (549) using `torch.func.jacrev` or `jacfwd` (567), chosen by `item.dim <= variable.free_dim` (638, 673); `_fd_block` (581) for finite differences; FD forbidden with `create_graph` (problem.py:46).
- Row weights are applied to blocks by each `Weight.apply_jacobian` (base.py:25, 70, 105); robust kernel weights are applied inside LM (`_robustify` lm.py:239), not in Problem.
- Evaluation at a non-current point works by temporarily assigning tensors into the Variable objects (`Problem._assigned` problem.py:269, `_evaluation` 282, `_evaluate_at` 469). So torch.func differentiates through object mutation plus a node-memo epoch (`_node_evaluation` 243, `_invalidate_nodes` 236).
- Temporal/banded: residual opts in with `temporal_structure()` + `temporal_jacobian_blocks()` (residuals/temporal.py:70-84) returning `TemporalPattern` offsets (structure.py:41). `analyze_temporal_problem` (temporal.py:186, ~85 lines, PLR0911) decides dense vs structured; `assemble_structured_normal` (temporal.py:341-451, PLR0912/0915) builds `BlockBandedMatrix` (62) solved by `BandedCholesky` (solvers.py:181-297, `solve_with_info` PLR0915). `_warn_missing_temporal_blocks` (270) warns when a residual lacks blocks. `_FunctionResidual.__getattr__` forwards `temporal_structure`/`temporal_jacobian_blocks` from the wrapped function object (base.py:243-246): a decorator-made residual opts in by attributes stuck on the function.
- Complexity concentration: problem.py (751; `_freeze` 88 lines, `jacobian_blocks` 77 lines with PLR0912), temporal.py (451), lm.py `_static_layout` (142-238) and `_update` (613-756, 144 lines). Analytic `jacobian()`/`temporal_jacobian_blocks()` live in 7 residual files (14 methods: contact 2, limits 2, pose 2, smoothness 2, temporal 2, regularization 3, projection 1).
- Duplication residuals/ vs optim/: the core contract `Residual`, `Weight`, `residual`, `TemporalPattern` live in residuals/ but are re-exported as optim API (optim/__init__.py:4-5), while `Problem` must know `Node` and temporal hooks; the kernel type is `kernel: object | None` on `Residual` (base.py:137) because `RobustKernel` sits above in optim/kernels.py. Dense temporal Jacobian helpers exist in residuals/_temporal_jacobian.py:14,41 alongside `BlockBandedMatrix.densify` (optim/temporal.py:115) (likely overlap; not diffed).

## 4. Adding an optimizer

- Contract: subclass `Optimizer` (optimizers.py:41) and implement `step()`; `optimize()` is inherited. Linear solvers are pluggable via the 2-method `LinearSolver` protocol (solvers.py:35) -> "GN with a different linear solver" is the one easy case: `GaussNewton(problem, solver=MySolver())` (lm.py:428-434 accepts any `LinearSolver`).
- Anything else needs private Problem API. Private members used by in-tree optimizers:
  lm.py: `_dense_jacobian_at _evaluate_at _freeze _kernel _set_trainable _trainable_values _update_serial _validate_trainable_values`;
  optimizers.py: `_batch_and_exemplar _freeze _set_trainable`;
  implicit.py: `_evaluate_at _kernel _ordered_variables _validate_trainable_values`;
  temporal.py: `_error_current _item_variable_reads _residuals_by_name _vars_by_name _warned_fallbacks`.
  There is no public `linearize(x) -> (r, J, weights)` or `objective(x)` that is pure in `x`; public `jacobian_blocks`/`gradient` act on the variables' current tensors.
- An L-BFGS native optimizer: needs flat tangent vector <-> per-variable retract (`Problem.retract` 710 is public, good), objective at a point (private `_evaluate_at`), batch layout (private `_batch_and_exemplar`), bounds (LM's private `_static_layout`). An Adam+LM hybrid: today done by two optimizer objects over one Problem + `resume()` (staged_fit.md:83-104); no shared state between them.
- LM is monolithic: class spans lm.py:409-938; it fuses damping schedule, robust IRLS (`_robustify` 239, `_robust_decrease` 263), box-bound projection and active sets (`_static_layout` 142, `_active_mask` 288, `_projected_gradient` 306), dense/structured dispatch (`resolve_linearization` 479, `_linearize_model` 321), warm-start/resume (804, 830) and implicit-diff hookup. `GaussNewton` is LM with fixed damping (939-945).
- Hard-coded: robot topology inside the generic optimizer — `_unsafe_translation_v_indices` checks `joint.kind == "free_flyer"` and `JointComposite` (lm.py:99-110); solver "auto" -> Cholesky or BandedCholesky only; linearization modes fixed to dense|structured (lm.py:28); 16 ctor scalars incl. `mu_max`, `increase_factor_max` (lm.py:412-428). Task recipes pick optimizers by string (`OptimizerConfig.optimizer`, `linear_solver: Literal["cholesky"]`, `kernel: Literal[...]`, ik.py:74-93; `_make_linear_solver` 122, `_make_robust_kernel` 129).

## 5. Rigidities (buys / costs)

- Fixed-width residuals (`dim` required and validated, base.py:140; output checked `_validate_output` problem.py:334). Buys: static row offsets (`row_offsets` problem.py), compile-friendly fixed shapes, no host sync. Costs: user computes widths by hand; variable-count things (contacts, visible keypoints) need masks + `active_groups()` + `reduce="mean_active"`; any data size change needs a new residual.
- Object-owned variables, `error()` with no args (base.py:218, 239-240). Buys: residuals read like plain code; sharing a variable between terms is free; `Problem.update` keeps optimizer state across phases (staged fits). Costs: Problem is stateful; evaluation at other points mutates objects (`_assigned` problem.py:269); hostile to `torch.func.vmap`/`torch.compile`/multi-threading; results come back by side effect; data you want to change must be wrapped as `Variable(trainable=False)`.
- Freeze at first use (problem.py:92-95, `_freeze` 100). Buys: one validated layout. Costs: no adding/removing terms; phases = toggling `enabled`/`weight` + `optimizer.resume()` (own_your_optimization_loop.md ~70-80).
- Node memo system (nodes.py:12-118; epoch handling problem.py:236-268; merge by `merge_key` nodes.py:46). Buys: one FK per evaluation shared across pose/position/orientation terms. Costs: a second graph concept with lifetime rules users must obey; `RobotState`/`robot_state` coupling in residual ctors (pose.py:83).
- Batch semantics: `batch_ndim` declared on each variable (variables.py:71, 88); per-element status in `OptimizerInfo`. Buys: true per-element LM at large batch. Costs: extra argument to get right; `TorchOptimizer`+LBFGS couples batch; eager `bool(...)` sync points left to the user loop.
- Manifolds (nq != nv): `Variable.retract/difference`, `tangent_dim`, `SO3/SE3/RobotVariable`. Buys: correct Lie updates, quaternion handling in implicit diff (`_smooth_quaternion_representative` implicit.py:162). Costs: three parallel hierarchies of tangent bookkeeping (variables.py `free_indices`, `gather_tangent`, `expand_tangent`, `temporal_free_indices` 155-222).
- Robust kernels: tiny protocol (kernels.py, 118 lines) with `group_size` grouping. Buys: easy to add a kernel. Costs: applied inside LM only (`_robustify`), so kernel semantics depend on the optimizer; typed `object` on Residual.
- Weights: `weight`, `row_weight` (`Weight` object or tensor), `kernel`, `reduce`, `group_size`, `enabled` on every residual (base.py:128-139). Costs: six knobs, three of them weight-like.
- Constraints: only box `Bounds` on state via projected LM (variables.py:24, lm.py:142-320); no equality/inequality constraints other than penalty residuals (`JointPositionLimit` limits.py:24); TorchOptimizer bound handling (unverified).
- Sparsity: only temporal block-banded (`BlockBandedMatrix`); no general sparse or Schur; dense `dense_jacobian` otherwise.

## 6. Worth keeping (evidence)

- Manifold-aware `Variable` with `retract`/`difference` and `RobotVariable` joint groups/freezing (variables.py:56-594).
- Optional analytic Jacobian blocks per (residual, variable) with autodiff fallback, keyed dict output (problem.py:607-683).
- `LinearSolver` protocol, 2 methods (solvers.py:35-55) and a working `BandedCholesky` for long trajectories (solvers.py:181).
- Robust kernel protocol with `rho`/`weight` (kernels.py:17-27).
- Fixed-shape, host-sync-free LM inner update with per-batch-element status (`_update` lm.py:613, `OptimizerStatus` optimizers.py:18) — matches use case 2 (large GPU batch).
- `Problem.update` + `resume()` preserving Adam moments/scheduler and LM damping across phases (staged_fit.md:83-104, optimizers.py:105-113) — matches BVR staged fits.
- Implicit differentiation through the LM optimum (implicit.py, `solve_ik(differentiable=True)` ik.py:180) — use case 4.
- `term_costs()` per-term logging (problem.py:503).
- Guides are executable `testcode` blocks (own_your_optimization_loop.md:13).

## 7. Top 8 design problems (ranked)

1. No pure functional core. Residuals read `variable.tensor` (base.py:239-240), evaluation at a point mutates objects (`_assigned` problem.py:269), results are side effects. Blocks clean torch.func/vmap/compile use, NN-layer use, and simple user loops.
2. No public interface for writing an optimizer. In-tree optimizers use 18 distinct private `Problem` members (list in §4). A user L-BFGS/Adam-LM hybrid must copy LM internals.
3. LM is a 530-line class fusing damping, robust IRLS, box projection, dense/structured dispatch, warm start and robot topology (lm.py:99-110 `free_flyer`/`JointComposite`; 142-320; 613-756). 16 ctor parameters (412-428). Nothing reusable for GN-with-line-search, dogleg, or a hybrid.
4. API surface: ~43 exports in optim plus `Node`, `RobotState`, `TimeIndexedResidual` from residuals; internal-looking types public (`LinearizationDecision`, `LinearizationReason`, `TemporalAnalysis`, `StructuredNormal`, `InformativeLinearSolver`, `LinearSolveResult`) (optim/__init__.py). Six weighting knobs per residual (base.py:128-139).
5. Mandatory fixed `dim` + scalar-loss hack: every residual declares its width (base.py:140); scalar losses become `sqrt(2f)` rows via `ScalarCost` (base.py:248) with singular GN columns at f->0 and implicit diff disabled. Bad fit for Adam-style weighted-loss fitting (use case 1).
6. Closed task recipes: `solve_ik` accepts no extra residuals (ik.py:172-180); string-enum configs (ik.py:74-93) re-encode optimizer choices; a custom cost forces a full manual rebuild.
7. torch.optim only via wrapper: `TorchOptimizer` owns hidden tangent `nn.Parameter` buffers (optimizers.py:131, 247); no documented way to hand `problem` parameters to a user-owned optimizer loop or scheduler; LBFGS couples the batch.
8. Temporal structure is a parallel subsystem: `TimeIndexedResidual`, `TemporalPattern`, `analyze_temporal_problem` (temporal.py:186), `assemble_structured_normal` (341), missing-block warnings (270), attribute forwarding through `_FunctionResidual.__getattr__` (base.py:243-246), plus `LinearizationMode` choice in LM. Plus inconsistent defaults: `auto` warns on autodiff (problem.py:635-640) while the guide calls autodiff the default (custom_residual.md:151).

## Open questions

- Does `TorchOptimizer` respect `Bounds` and robust kernels (not read)? Does `Problem.objective()` apply `rho` (problem.py:388-468 not read)?
- Public defaults of `PoseResidual`/`RestResidual` ctor args (pose.py:159+, regularization.py:24) not read; the base class has none.
- Exact overlap between residuals/_temporal_jacobian.py and optim/temporal.py densification not diffed.
