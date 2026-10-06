# Design study — a new optimization layer for BetterRobot

Author: subagent (opus), 2026-10-06. Inputs: `context.md`, `scan_bvr_optim.md`,
`src/better_robot/optim/CLAUDE.md` + `__init__.py` (no `scan_br_optim.md` existed),
`references/design/{theseus,pyroki,pypose,jaxopt,curobo}.md` (grepped, not read whole).

**Verification status.** Budget allowed one web search; it returned nothing on jaxls. jaxls is
not in `references/` (grep for `create_factory`/`jac_custom_fn` found nothing). So all claims
on jaxls, optimistix/lineax, torchmin, torchopt, GTSAM/Ceres and torch.func speed below are
**from my prior knowledge, unverified this session** (marked [U]). Claims on theseus, pyroki,
pypose, jaxopt, curobo cite the local summaries.

## 0. Why the present optim fails (facts)

- 4019 lines, 40+ exported names (`optim/__init__.py:5-22`): Variable, SO3Variable,
  SE3Variable, RobotVariable, Bounds, Residual, Weight, ScaleWeight, DiagonalWeight,
  Difference, ScalarCost, TemporalPattern, TemporalAnalysis, StructuredNormal,
  BlockBandedMatrix, LinearizationMode/Reason/Decision, JacobianStrategy, LinearSystem,
  LinearSolver, InformativeLinearSolver, ... A user must learn the graph model (object-owned
  Variables, Residuals referencing them, Nodes, harvest) before writing one cost.
- Residuals are fixed-width objects (`optim/CLAUDE.md:38-44`); a robust scalar sum (chamfer
  top-k, Geman-McClure) has no natural home. BVR's ~38 terms are mostly such scalars
  (`scan_bvr_optim.md:17-18`).
- Result: BVR never imports `better_robot.optim` and wrote a 193-line engine on torch.optim
  (`scan_bvr_optim.md:10-11`). That 193-line engine is the real specification.

## 1. Comparison of designs

Table A — how a user states a problem.

| Lib | Variables | Costs / residuals | Weights, robust | Manifolds |
|---|---|---|---|---|
| theseus | `th.Vector`, `th.SE3` objects, named, mutable, updated by name (`theseus.md:12`) | `CostFunction` subclass or `AutoDiffCostFunction(optim_vars, err_fn, dim)` | `CostWeight` scales error AND Jacobian (sqrt-info) (`theseus.md:13-14`); robust as re-weighting | Manifold classes with retract/local |
| pypose | `nn.Module` parameters (LieTensor) | model `forward(input)` returns residual; `optimizer.step(input)` | `weight=` info matrix, `kernel=` Huber/Cauchy | LieTensor, retract in step |
| jaxls / pyroki [U] | `Var` subclasses with default + tangent_dim + retract; ids index a batch of vars | `@Cost.create_factory` on a plain function `(vals, var, ...) -> residual`; constraints by `kind=` (AL) | scale inside the residual function | `Var` carries retract; SE3/SO3 via jaxlie |
| optimistix [U] | any pytree `y0` | `fn(y, args) -> residual pytree` (least_squares) or scalar (minimise) | user's job | none built-in (Euclidean pytrees) |
| optax | any pytree params | user computes grads; optax only transforms them | none | none |
| jaxopt | pytree | `fun(params, *args)` scalar or residual (`GaussNewton`, `LM`) | user's job | none |
| crocoddyl | state/control per knot | `ActionModel` per knot; `CostModelSum` of named, weighted residual costs | named weights, activations (robust) | `StateMultibody` diff/integrate (nq≠nv) |
| curobo | joint trajectory tensor | cost `nn.Module`s summed, custom CUDA kernels | per-cost weights, fixed | joints only (Euclidean) |
| torchmin [U] | one flat tensor | scalar `fn(x)` or residual `fn(x)` for `least_squares` | none | none |

Table B — Jacobians, solver composition, scale features, cost to use.

| Lib | Jacobians | Solver composition | Batching | Sparsity | Implicit diff | LOC simple LSQ | User complaints |
|---|---|---|---|---|---|---|---|
| theseus | analytic per cost or vmap+jacrev default (`theseus.md:16`) | Optimizer + LinearSolver (dense Chol/LU, CHOLMOD, Baspacho) | everything 2-D, batch dim | sparse via C++/CUDA extlib (`theseus.md:42-48`) | `TheseusLayer` unroll/implicit/truncated/DLM (`theseus.md:49`) | ~12 | heavy install, global registry, mutable state, slow small problems |
| pypose | autograd default, "slow for large problems" (`pypose.md:211`); sparse via `bae` (`pypose.md:212`) | `LM(model, solver=, strategy=, kernel=)` + `Scheduler` (StopOnPlateau) (`pypose.md:187-198`) | via model batch | optional bae backend only (`pypose.md:284`) | no (unroll) | ~8 | Jacobian of the whole model output is dense; slow at scale |
| jaxls [U] | autodiff per cost group, `jac_custom_fn` analytic override | `.analyze().solve(linear_solver=, trust_region=, termination=)` (`pyroki.md:139-147`) | costs of same structure stacked and vmapped | sparse block structure from var ids; dense Chol / CG / sparse QR | no built-in | ~8 | JIT warm-up (`pyroki.md:285`) |
| optimistix [U] | jacfwd/jacrev via lineax `FunctionLinearOperator`/`JacobianLinearOperator` | solver = descent (Newton, damped Newton, dogleg) x search (trust region, backtracking) x lineax linear solver; `rtol/atol` termination | vmap | via lineax operator tags | `ImplicitAdjoint` default | ~5 | abstract, many classes to compose a custom solver |
| optax | n/a (grads only) | `GradientTransformation(init, update)`, `chain`, `masked`, schedules | vmap | n/a | n/a | ~10 | none for its scope; no second order |
| jaxopt | autodiff | solver objects with `init_state/update/run` (`jaxopt.md:10`) | vmap of run, per-instance adaptive state (`jaxopt.md:190`) | via jax sparse CG (`jaxopt.md:299`) | `custom_root`, reference impl (`jaxopt.md:34`) | ~5 | archived/merged into optax; slow compile |
| crocoddyl | analytic everywhere (calcDiff) | FDDP/BoxDDP over shooting problem | none on GPU | Riccati exploits time structure | no | ~40 | verbose, C++ |
| curobo | custom kernels for FK/collision | MultiStage: MPPI → L-BFGS (fused CUDA) → CG (`curobo.md:566-583`) | massive particles, fixed iters | link/joint masks (`curobo.md:128`) | no | n/a (config-driven) | config-heavy, hard to customize |
| torchmin [U] | autograd + HVP | `minimize(fn, x0, method=...)` | none (single problem) | no | no | 3 | no batching, no manifolds |

## 2. Ideas worth stealing (7)

1. **Costs are plain functions; structure comes from what they read** (jaxls [U], optimistix).
   A term is `fn(**vars) -> Tensor`. No Residual class hierarchy. Fits torch: any existing
   loss code drops in unchanged.
2. **Same-structure terms are stacked and vmapped** (jaxls [U], theseus `Vectorize`
   `theseus.md:38`). In BR the natural stack axis already exists: time `T` and batch `B`. A
   term evaluated on all knots at once gives a block-banded Jacobian for free.
3. **sqrt-information weight scales residual and Jacobian together; robust kernel = IRLS
   reweighting of both** (theseus `theseus.md:12-15`, Ceres/GTSAM [U]). One rule makes the
   same term valid for Adam (scalar ½w‖ρ(r)‖) and for GN/LM.
4. **Solver = small `init/step` over a value state, not a class tree** (optax, jaxopt
   `jaxopt.md:10`). Lets users write an optimizer in 20 lines and keeps per-problem state as
   `(B,)` tensors (jaxopt per-instance state `jaxopt.md:190`).
5. **Damping/line search and stopping as pluggable small objects** (pypose Strategy +
   Scheduler `pypose.md:187-188`; optimistix descent x search [U]). Take only two slots:
   `damping` and `stop`. Do not take the full optimistix lattice.
6. **Implicit diff via the optimality condition** (jaxopt `custom_root` `jaxopt.md:34`,
   theseus backward modes `theseus.md:49`). For least squares, F = Jᵀr = 0 and the GN
   Hessian JᵀJ reuses the forward factorization.
7. **Staged pipelines with fresh optimizer state per stage** (curobo MultiStage
   `curobo.md:583`, BVR `Phase` `scan_bvr_optim.md:63-69`). A phase = weights + free mask +
   solver + iters + hook.
8. **Lazy shared intermediates computed once per evaluation** (BVR `put_lazy`,
   `scan_bvr_optim.md:49-55`). 25 lines with string keys beat the current `Node` graph.

## 3. Proposal

### 3.1 Concepts a user learns: five

`Problem`, `term` (residual or scalar), `Manifold` (only if a variable is not Euclidean),
a solver (`Adam(...)`/`LM(...)`/any `torch.optim` class), `Phase`. Kernels, `cached`
intermediates, bounds are keyword arguments, not concepts.

### 3.2 Variables and terms

- Variables are a `dict[str, Tensor]`, each with leading batch axes `(B, ...)` (or `(B, T, ...)`
  for trajectories). Optional per-variable `Manifold` (retract, difference, tangent_dim,
  project) and optional `bounds=(lo, hi)`. `RobotManifold(model)` wraps
  `Model.integrate/difference`, so nq≠nv works.
- A term is a function whose **parameter names are the variables or cached intermediates it
  reads** (dependency injection by name; `reads=` overrides). Two kinds:
  - `kind="residual"`: returns `(B, ..., m)`. Its scalar value is ½·w·Σρ(‖√W r‖²).
  - `kind="scalar"`: returns `(B,)` or `()` already reduced. Only first-order solvers use it;
    in GN/LM it is skipped with a warning or folded as a gradient-only term (see 3.6).
- `P.cached` registers `fn(**reads) -> Any` computed lazily once per evaluation.

### 3.3 Jacobian representation and assembly

Per residual term, `J_term` is a `dict[var_name, Tensor]` with blocks in **tangent
coordinates**:
- dense block: `(B, m, nv_var)` for terms on non-temporal variables;
- temporal block: if the term declares `stencil=(-1, 0, 1)` (offsets it reads along `T`) the
  term is written per knot window and the block is `(B, T, m, len(stencil), nv_knot)`;
- global block in a temporal problem (betas shared over all T): `(B, T, m, nv_global)`.

How it is obtained, in order: (1) `term.jac(**reads)` if provided (analytic, same layout);
(2) otherwise `torch.func.jacfwd` if `nv_reads <= m` else `jacrev`, of
`δ -> fn(retract(x, δ))` at δ=0, vmapped over `B` (and over `T` for stencil terms). The
retract-at-zero trick gives tangent Jacobians for any manifold without per-manifold
Jacobian code.

Assembly never forms the full `J`. The solver accumulates `H = Σ Jᵢᵀ Wᵢ Jᵢ` and
`g = Σ Jᵢᵀ Wᵢ rᵢ` block by block:
- all-dense problems (IK, N ≤ ~200): `H (B, N, N)`, batched Cholesky;
- temporal problems: block-banded `H (B, T, s, nv, nv)` with half-bandwidth s = stencil
  span, banded Cholesky (the current `BandedCholesky` can be kept); global vars form an
  arrowhead border handled by a Schur complement on the `(n_global, n_global)` block.
Anything else (arbitrary sparse graphs, loop closures) is **not supported**: dense
fallback only. Generality traded for simplicity.

### 3.4 Manifolds in both solver families

- Second order: solve for δ in tangent space, `x ← retract(x, δ)`. Exact.
- torch.optim: two modes. Default `"project"`: optimize the ambient tensor, apply
  `manifold.project` (renormalize quaternions) after each step — what BVR already does
  (`scan_bvr_optim.md:47-48`). Opt-in `"tangent"`: optimize δ around an anchor, re-anchor
  `x ← retract(x, δ); δ ← 0` every step, keeping Adam moments (no parallel transport; fine
  for small steps, unverified for large ones). Trade: no Riemannian Adam.

### 3.5 Solver protocol and a custom optimizer

```python
class Solver(Protocol):
    def init(self, prob: Problem, x: dict) -> State: ...
    def step(self, prob: Problem, st: State) -> State: ...   # fixed-shape, no host sync

@dataclass
class State:                       # all tensors; batch-first
    x: dict[str, Tensor]
    loss: Tensor                   # (B,)
    done: Tensor                   # (B,) bool
    it: int
    extra: dict[str, Tensor]       # solver-owned: lambda, moments, history
```

`prob` gives the solver four calls: `prob.loss(x) -> (B,)`, `prob.grad(x) -> dict` (tangent),
`prob.linearize(x) -> (H, g, loss)` (structured), `prob.retract(x, dx_dict) -> dict`.
`torch.optim` classes are wrapped by one adapter `TorchOpt(cls, **kw)`. Full toy optimizer
(damped GN with backtracking on λ, per-problem):

```python
class SimpleLM(bo.Solver):
    def __init__(self, lam=1e-3, up=10.0, down=0.3, tol=1e-8):
        self.lam0, self.up, self.down, self.tol = lam, up, down, tol

    def init(self, prob, x):
        loss = prob.loss(x)
        lam = torch.full_like(loss, self.lam0)
        return bo.State(x, loss, torch.zeros_like(loss, dtype=torch.bool), 0, {"lam": lam})

    def step(self, prob, st):
        H, g, _ = prob.linearize(st.x)                 # structured H, g: (B, N)
        lam = st.extra["lam"]
        dx = -H.solve(g, damping=lam)                   # (B, N); Cholesky of H + lam*diag(H)
        x_new = prob.retract(st.x, dx)
        loss_new = prob.loss(x_new)
        ok = (loss_new < st.loss) & ~st.done            # (B,) per-problem accept
        x = bo.where(ok, x_new, st.x)
        loss = torch.where(ok, loss_new, st.loss)
        lam = torch.where(ok, lam * self.down, lam * self.up)
        done = st.done | ((st.loss - loss).abs() < self.tol * (1 + st.loss))
        return bo.State(x, loss, done, st.it + 1, {"lam": lam})
```

The library LM is this plus Nielsen damping update, gradient/step tolerances and Jacobian
column scaling (BVR's gradient-scale complaint, `scan_bvr_optim.md:128-131`).

### 3.6 Mixed scalar + residual terms

One problem may hold both. First-order solvers use the total scalar. GN/LM use residual terms
for `H` and add the gradient of scalar terms to `g` with `H` unchanged (a "gradient-only"
term; LM damping keeps the step safe). That is cheap and honest; a phase that is mostly
scalar should use L-BFGS/Adam instead. Users choose per phase.

### 3.7 Phases, masks, hooks

```python
bo.Phase(name, weights: dict[str, float], solver, iters, free: dict[str, Tensor|str] = "all",
         on_start: Callable[[Ctx], None] | None = None, stop_every: int = 0)
```

- weight 0 ⇒ term inactive (not evaluated, not linearized). Weights are a dict per phase.
- `free`: bool mask over each variable's tangent coordinates (or named group from the
  manifold, e.g. `RobotManifold.groups["root"]`). torch.optim: grad masked after backward
  (works for Adam and L-BFGS, as BVR). GN/LM: masked columns get zero `g` and unit diagonal
  in `H` — fixed shape, no gather, no host sync.
- `on_start(ctx)`: `ctx` holds `x`, `problem`, `phase`, a free `ctx.store` dict (replaces
  BVR closure cells `scan_bvr_optim.md:139-141`), and may replace data tensors.
- Fresh solver state per phase. `bo.run(P, x0, phases, log=...)` returns `Result(x, trace)`;
  per-term values are kept as tensors and synced only at log points.

### 3.8 Batching without host sync

Every tensor in `State` is `(B, ...)`. Steps compute for all `B` and write back with
`torch.where(done, old, new)`. The driver syncs `done.all()` only every `stop_every` iterations
(0 = never; fixed iterations, CUDA-graph and `torch.compile` friendly). LM λ, line-search
step and L-BFGS history are per problem. Note: `torch.optim.LBFGS` is single-problem and syncs
internally; a batched L-BFGS (curobo-style) must be written for IK/trajopt at B=4096.

### 3.9 Implicit differentiation

`bo.solve_layer(P, x0, phases | solver, params: dict, mode="implicit"|"unroll"|"detach")`
returns `x*` attached to `params` (the non-optimized tensors terms read, e.g. targets, NN
outputs). `implicit` (residual problems only): custom `autograd.Function`; backward solves
`(JᵀWJ + εI) v = ∂L/∂x*` with the forward's last `H`, then returns `-vjp_params(Jᵀ W r)(v)`
(the jaxopt `custom_root` formula with the GN Hessian). `unroll`: plain autograd through a
fixed number of steps (default for torch.optim solvers). Scalar-term implicit diff (needs
HVP + CG) is deferred.

### 3.10 Bounds and constraints — minimum

- In scope now: box bounds per variable (`bounds=(lo, hi)`), applied by projection after every
  step for both families; for LM also by clamping the step (projected LM; not exact KKT, good
  enough for joint limits and λ ≥ 0 forces, which BVR does with `clamp_(min=0)`
  `scan_bvr_optim.md:72-73`). Joint limits as soft residuals remain the IK default.
- Augmented Lagrangian: not a solver. A 30-line recipe using a phase loop: outer phases
  update multipliers in `on_start`. Promote to the library only when two users need it.
- Deferred: general inequality constraints, QP subproblems, DDP/FDDP, SQP.

### 3.11 The five use cases

Plain least squares (5 lines):

```python
P = bo.Problem({"p": torch.zeros(3)})
P.residual("fit", lambda p: model_fn(t, p) - y)
x = bo.solve(P, bo.LM(), iters=50).x       # or bo.solve(P, torch.optim.Adam, lr=1e-2, iters=500)
```

BVR staged human fit:

```python
P = bo.Problem({"q": q0, "betas": b0, "f": f0},
               manifolds={"q": RobotManifold(body.model)},  # nq=211, nv=158
               bounds={"f": (0.0, None)})
@P.cached
def posed(q, betas): return body.forward(betas, q)
@P.cached
def tau(q, f): return contact.rnea_frame_forces(body.model, q, f, frames, dt)

P.scalar("reproj",  lambda posed: robust.gm(cam.project(posed.joints) - kp2d, kp_conf, s=100))
P.scalar("chamfer", lambda posed: scene.chamfer(posed.vertices[:, sub], depth))
P.residual("pose_prior", lambda q: prior.pose(q, q0))
for k in (1, 2, 3):
    P.residual(f"smooth{k}", temporal.diff("q", order=k))      # stencil term, banded J
P.residual("base_wrench", lambda tau: tau[..., :6])
P.residual("force_smooth", temporal.diff("f", order=1))

g = P.manifolds["q"].groups
plan = [bo.Phase("root", W[0], bo.Adam(lr=1e-2, milestones=(.5, .75)), 150, free={"q": g.root}),
        bo.Phase("full", W[1], bo.Adam(lr=1e-2), 300, free={"q": ~g.fingers}, on_start=build_scene),
        bo.Phase("polish", W[1], bo.LBFGS(history=10), 30, free={"q": ~g.fingers}),
        bo.Phase("kindyn", W[3], bo.LM(scale_columns=True), 150, free={"q": ~g.fingers, "f": True})]
res = bo.run(P, plan, log=bo.log_terms(every=0.25))
```

Batched robot IK with a custom cost (B=4096):

```python
P = bo.Problem({"q": q0.expand(4096, -1)}, manifolds={"q": RobotManifold(model)},
               bounds={"q": model.joint_limits})
@P.cached
def fk(q): return kinematics.frame_poses(model, q, frames=[ee])
P.residual("pose", lambda fk: se3.log(se3.inv(target) @ fk[..., 0, :]),
           jac=kinematics.pose_residual_jacobian)              # analytic block (B, 6, nv)
P.residual("elbow_up", lambda q: torch.relu(-q[..., 3:4]), weight=0.1)   # custom cost
res = bo.solve(P, bo.LM(), iters=30, stop_every=10)            # per-problem done mask
```

Human → robot retargeting (trajectory, T knots):

```python
P = bo.Problem({"q": q_init}, manifolds={"q": RobotManifold(robot)})   # (T, nq)
@P.cached
def fk(q): return kinematics.frame_poses(robot, q, frames=robot_keypoints)
P.residual("keypoints", lambda fk: (fk[..., :3] - human_kp * scale).flatten(-2), weight=w_kp)
P.residual("orient", lambda fk: so3.log(so3.inv(human_rot) @ fk_rot(fk)).flatten(-2))
P.residual("vel", temporal.diff("q", order=1), weight=0.1)
P.residual("feet_contact", lambda fk: foot_slide(fk, contact_mask), weight=10.0)
res = bo.solve(P, bo.LM(), iters=50)                           # banded Cholesky over T
```

Differentiable IK layer inside training:

```python
P = bo.Problem({"q": q0}, manifolds={"q": RobotManifold(model)})
P.residual("pose", lambda q, target: pose_err(model, q, target))   # target is a parameter
for batch in loader:
    target = net(batch.obs)                                         # (B, 7), requires grad
    q_star = bo.solve_layer(P, bo.LM(), iters=20, params={"target": target}, mode="implicit")
    loss = task_loss(q_star, batch); loss.backward(); opt.step()
```

## 4. Internal architecture

Modules (target ≤ ~1500 lines total, vs 4019 now):

| Module | Content | Est. lines |
|---|---|---|
| `optim/problem.py` | `Problem`, `Term`, cached-intermediate evaluator, weights, masks | 300 |
| `optim/manifold.py` | `Manifold` protocol; `Euclidean`, `SO3`, `SE3`, `RobotManifold` (delegates to lie/data_model) | 150 |
| `optim/jacobian.py` | tangent autodiff (jacfwd/jacrev+vmap), stencil vmap, block accumulation into `H, g` | 250 |
| `optim/linear.py` | `DenseNormal`, `BandedNormal` (+ arrow Schur) with `.solve(g, damping)`; CG | 250 |
| `optim/solvers/` | `TorchOpt` adapter, `Adam`/`LBFGS` sugar, batched `LBFGS`, `GaussNewton`, `LM` | 350 |
| `optim/phases.py` | `Phase`, `Ctx`, `run`, logging trace | 120 |
| `optim/implicit.py` | `solve_layer` modes | 120 |
| `optim/kernels.py` | Huber, Cauchy, GM, Tukey as `ρ(s)` + IRLS weights (keep current) | 60 |

Core types and fields:
- `Term(name, fn, kind: "residual"|"scalar", reads: tuple[str,...], weight: float|Tensor,
  sqrt_info: Tensor|None, kernel: Kernel|None, jac: Callable|None, stencil: tuple[int,...]|None)`.
- `Problem(vars: dict[str, Tensor], manifolds: dict[str, Manifold], bounds, terms: dict[str, Term],
  cached: dict[str, Callable], data: dict[str, Tensor])` — `data` holds non-optimized tensors
  (targets, NN outputs) addressable by name like variables; this is what `solve_layer` differentiates.
- `State` as in 3.5. `Phase` as in 3.7. `Normal` (H with `.solve`, `.matvec`, `.diag`).

Deliberately NOT included (each is a generality-for-simplicity trade):
- No variable objects owning tensors, no graph harvest, no `Node` class: plain dicts + names.
- No general sparse linear algebra (CHOLMOD, Baspacho, sparse QR); only dense, block-banded,
  arrowhead. Pose-graph SLAM-style problems are out of scope.
- No `LinearizationMode/Decision`, `TemporalAnalysis`, `InformativeLinearSolver`: structure
  comes from the `stencil` declaration, not from analysis.
- No optimistix-style descent x search lattice; LM, GN, batched L-BFGS, torch.optim adapter.
- No Riemannian first-order methods (project / re-anchor only).
- No finite-difference strategy in the public API (keep as a test utility for checking `jac`).
- No exact constrained solvers (QP, SQP, DDP); bounds by projection; AL as a recipe.
- No implicit diff for scalar terms in v1.
- Dependency injection by parameter name trades explicitness for brevity; `reads=` exists.

## 5. Risks, open questions, what to prototype first

Prototype first (each ≤ 1 day, run on the Blackwell box):
1. **torch.func cost at B=4096.** Pose residual of a 7-DoF Panda and a 30-DoF humanoid:
   `vmap(jacfwd)` through FK vs the analytic frame Jacobian vs `torch.autograd` row-by-row.
   Measure ms/iter, peak memory, and under `torch.compile`. Expectation [U]: jacfwd+vmap is
   2-5x slower than analytic for FK, acceptable for prototyping, not for the IK hot path —
   hence the `jac=` slot. If vmap breaks on data-dependent indexing in FK, that kills the
   default path and must be known now.
2. **Stencil Jacobian for T=300, nv=158 (BVR q)**: banded `H` build + banded Cholesky time
   vs dense `(47k)²` (infeasible) vs L-BFGS. Decides whether LM is viable for the BVR kindyn phase.
3. **Batched L-BFGS without host sync** (two-loop recursion on `(B, m, N)` history, fixed
   Armijo backtracking count) vs `torch.optim.LBFGS` on BVR polish: quality and speed.
4. **Tangent re-anchoring for Adam** vs ambient+project on BVR root/full phases: same final
   loss? (Unverified that moments stay meaningful.)
5. **Implicit vs unrolled gradients** for the IK layer: gradient agreement when LM has not
   fully converged (20 iters); the known weakness of implicit diff.

Risks:
- Name-based injection: a typo in a parameter name becomes a "missing variable" error at
  `Problem` freeze — must fail loudly at registration, not at solve.
- Scalar terms inside LM as gradient-only can stall: H ignores their curvature. Document;
  warn when scalar terms dominate the loss in an LM phase.
- `torch.compile` over user Python callables: graph breaks on `.item()`, data-dependent
  shapes (chamfer top-k chunking). Keep the compiled region to `step`, allow eager fallback.
- Masked columns via unit diagonal waste FLOPs when most DOFs are frozen (BVR root phase:
  7 of 158 free); acceptable for dense IK, maybe not for banded T=300. Gather may be needed.
- Arrowhead (global betas) + banded Schur: more code than it looks; first version may treat
  globals by alternating (freeze betas in LM phases).
- Per-problem convergence with fixed iterations wastes compute on converged problems; a
  compaction step (re-pack unfinished problems) is a possible later optimization, not v1.

Open questions for the owner:
- Is dependency injection by argument name acceptable, or must terms list `reads` explicitly?
- Keep `RobotVariable.frozen_groups` semantics, or masks only per phase (proposal: per phase)?
- Is AL as a recipe enough, or is a first-class constraint kind needed in v1 (pyroki has one [U])?
- Should BVR's multi-person `(P, N, ...)` become the batch axis `B` of this API?
