# 02 — Target Architecture

What the library looks like when the polish phase is done. The layer DAG is
unchanged; this document describes what changes *inside* the layers: one
optimizer stack instead of two, an open core, and a single validation policy.

```
lie → spatial → data_model → {kinematics, dynamics} → residuals → optim → tasks → viewer
io reads from data_model; collision sits beside kinematics
```

## 1. One optimizer stack

Everything under `optim/` today (8,055 lines, two coexisting stacks, sixteen
public/prevalidated method twins) becomes one flat package around 3,000 lines:

```
optim/
  __init__.py     public surface
  manifolds.py    Manifold, Euclidean, SO3Manifold, SE3Manifold, RobotConfig, Bounds
  problem.py      Problem (with add_variable / add_residual), evaluation context
  lm.py           LevenbergMarquardt, GaussNewton, LMState (trimmed)
  first_order.py  the torch.optim adapter (~80 lines)
  temporal.py     block-banded normal assembly for trajectory problems
  implicit.py     slim implicit differentiation behind solve(differentiate="implicit")
  kernels.py      L2, Huber, Cauchy, Tukey, GemanMcClure — one file
  solvers.py      Cholesky, BandedCholesky — one file
```

Gone entirely: the legacy `LeastSquaresProblem`/`CostStack`/`Optimizer` stack
and `costs/`, the custom Adam and L-BFGS, the phase engine, the scalar
`ObjectiveItem` subsystem, the provider DAG, the matrix-free/`NormalCG` route,
the `_graph_executor` harness, and every `_prevalidated` shadow method.
Rationale for each cut is in the work orders; every cut is evidence-backed
(zero production callers, or one caller that inlines to a few lines).

### The user-facing story

Fitting a line — the smallest possible problem — should look like this:

```python
import torch
from better_robot.optim import Problem, LevenbergMarquardt

x = torch.tensor([0., 1., 2., 3.]); y = torch.tensor([1., 3., 5., 7.])

def fit(ctx):
    m, c = ctx["theta"][..., 0:1], ctx["theta"][..., 1:2]
    return m * x + c - y

problem = Problem()
problem.add_variable("theta", shape=(2,))               # Euclidean by default
problem.add_residual(fit, dim=4)                        # plain callable is enough
values, state = LevenbergMarquardt().run({"theta": torch.zeros(2)}, problem)
```

And IK without the facade should take one screen, not two:

```python
robot = RobotConfig(model)
problem = Problem()
problem.add_variable("q", manifold=robot, bounds=robot.joint_bounds())
problem.add_residual(PoseResidual(model, frame="body_panda_hand", target=T_goal))
values, state = LevenbergMarquardt().run({"q": q0}, problem)
```

(`joint_bounds()` lives on `RobotConfig`, not on `Model` — `Bounds` is an
optimizer-layer type and the layer DAG forbids `data_model` reaching up. It
replaces today's hand-masking ritual for quaternion coordinates.)

Design rules that make this possible:

- **Builder surface.** `add_variable(name, ...)` and `add_residual(...)` replace
  hand-assembled `VarSpec`/`ResidualItem` tuples. The record types still exist
  underneath; users never have to spell them for simple problems.
- **No double naming.** A residual's name defaults from the object (or the
  function name); the item wrapper never requires repeating it.
- **`reads` is structure, not police.** A residual may declare which variables
  it reads (this is what makes Jacobians block-sparse). If it doesn't and the
  problem has one variable, that variable is assumed. Undeclared access is not
  a runtime crime.
- **Context with a memo, not a graph engine.** The evaluation context exposes
  variable values plus lazily computed provider outputs through a
  per-evaluation memo, so FK runs once no matter how many residuals need it.
  Providers stay as a concept — three exist (`RobotStateProvider`,
  `SceneSDFProvider`, the contact-dynamics provider) — but shrink to "declared
  outputs + a compute function"; recursive memoization replaces the explicit
  cycle-detection/topological-sort machinery, and reads-to-variable dependency
  propagation (which Jacobian structure needs) becomes a simple union at
  problem freeze.
- **One residual protocol.** Residuals are callables over a context, with
  optional `jacobian_blocks` and temporal declarations. The parallel legacy
  protocol (`ResidualState`, per-residual `.jacobian()`, transpose-apply
  hooks, the `JacobianStrategy` enum) is deleted with the stack it served.

### Solvers

**LM / Gauss–Newton stay ours and keep their strengths:** batched independent
problems with per-element damping/acceptance/convergence, manifold retraction,
robust IRLS kernels, state-space bounds with projected-gradient KKT, dense and
block-banded normal routes, and the `init_state / update / run` lifecycle for
consumers who own the loop. Trimmed: `LMState` drops pure-diagnostic fields,
and the internal evaluation path is written once — not twice per method split
by trust level. Two **deliberate API removals** (reachable-but-unjustified
options, not dead code): the matrix-free/`NormalCG` route and the `"lstsq"`
linear-solver option with its rank-deficient `Cholesky.solve` fallback. Both
are resurrectable from history if a real caller appears; their tests and
result fields go with them.

**First-order optimization is `torch.optim`'s job.** The adapter is one small
loop: hold a persistent tangent delta per block, let any `torch.optim`
optimizer step it, retract onto the manifold, re-zero the delta so its chart
stays at the current point (optimizer state persists across rebasing — the
same approximation the hand-written Adam made, at a tenth of the code):

```python
delta = problem.zero_tangent(values)                    # leaf tensors
opt = torch.optim.Adam(list(delta.values()), lr=lr)
for _ in range(iters):
    opt.zero_grad()
    problem.objective(problem.retract(values, delta)).sum().backward()
    opt.step()
    values = problem.retract(values, detach(delta)); zero_(delta)
```

This is a **deliberate behavior change, not a drop-in swap**: the custom
Adam's per-element step counters, atomic trial rejection, and warm-start
moment validation are exactly the over-engineering being removed, and the
tests that pin those semantics are removed with it. What must survive is
solution quality: `solve_ik`'s `"adam"` and `"lm_then_adam"` modes keep their
regression tolerances. `"lm_then_adam"` becomes two solver calls around a
rebuilt refinement problem (preserving the `refine_disabled_items` weighting
the phase engine provided); then the phase engine is deleted.

**Differentiable solves are a feature, not a fortress.** The implicit-
differentiation path (adjoint through the optimality system) is kept and
slimmed from 822 lines to about 500. The math stays; so do the guards that
prevent *silently wrong* gradients (convergence, stable active set, the
quaternion-cut and robust-kink checks — they are correctness, not paranoia).
What goes is the apparatus around them: the nine-knob config, duplicated
state-layout validation, and error-formatting bloat. `solve_ik` gains a
`differentiable=True` option that uses it — the solver's `solve(...)` entry
is public today, but no task facade reaches it.

### What `solve_trajopt` takes

`solve_trajopt(model, horizon, dt, initial_q_traj, residuals=[...], ...)` —
the same residual items a `Problem` takes. The `CostStack` input dies with the
legacy stack; internally nothing changes (it already converts to a temporal
`Problem` and routes dense or banded automatically).

## 2. An open core

The audit confirmed the core *is* differentiable — FK, Jacobians, RNEA, ABA,
CRBA, centroidal, integrate/difference, and every residual back-propagate to
`q` **and** to model values. The problems are exposure and consistency:

- **Raw passes become first-class, uniformly.** One naming convention
  (`forward_kinematics_raw`, `frame_placements_raw`, `joint_jacobians_raw`,
  `rnea_raw`, `aba_raw`, `crba_raw`, `ccrba_raw`), all in their package
  `__all__`, all returning frozen `*Result` dataclasses. Today the same
  concept has three visibility tiers and two return styles.
- **The seam is public.** `ModelStructure` and `ModelValues` join the
  top-level exports — they are the documented kernel boundary and the way
  power users batch and differentiate model parameters.
- **`Data` becomes optional for dynamics.** `rnea(model, q, v, a)` returns the
  torques; passing `data=` remains available for callers who want the
  workspace filled. Nobody allocates a workspace to throw it away.
- **Facades stop detaching the world.** `solve_*` results stay detached by
  default (correct for a solver), but the differentiable path is wired and
  documented, and `solve_contact_forces` stops force-detaching its outputs.
- **The surface tells the truth.** The six exported dynamics stubs that only
  raise `NotImplementedError` (`compute_minverse`, `compute_coriolis_matrix`,
  three integrators, centroidal-derivatives) move to `for_future.md`; the
  deprecated `nle` alias dies; `br.spatial` becomes reachable like every other
  subpackage; the unused `ReferenceFrame` enum is reconciled with the
  `Literal` strings actually used.

## 3. One validation policy

The single biggest source of "wall of ifs" is duplicated trust logic. The
policy after polish:

1. **Public boundaries validate structure once** — shapes, dtypes, devices,
   with error messages that name the argument and the fix. This is where
   helpfulness lives.
2. **No value-content policing.** We do not `isfinite`-scan user tensors —
   torch itself doesn't. Numerical garbage propagates to an honest terminal
   status (`FAILED` when the current model is non-finite; a rejected trial
   may legitimately end `MAXITER`) instead of being hunted at every door.
   Structural validation (names, shapes, dtype, device) still happens — once,
   at `init_state`/entry — which is what lets the shadow twins collapse into
   a single evaluation path.
3. **Frozen objects are validated when attached, never re-validated when
   used.** `ModelValues` is checked where a `ModelStructure` is available to
   check against — at `Model` construction and `with_values` — and the
   per-call re-validation inside every public FK/dynamics entry goes away
   (the boundary counter contract is updated deliberately, not weakened
   silently). The raw `*_raw` seam trusts its caller by documented contract.
4. **Typed configuration is trusted.** A `Literal`-typed dataclass field is
   not re-policed with `isinstance` walls at every consumer; impossible states
   don't get error branches.

## 4. The style bar

Uniformity is a feature. After polish: sibling functions share return
conventions (raw → `*Result`, public → tensor or `Data`, stated per family);
one concept has one name everywhere; error messages share one voice
(`"<what> must <rule>, got <actual>"`); duplicated helper definitions are
single; comment density follows the surrounding file; every public symbol has
a docstring that would satisfy a stranger. The hot-path rules (no `.item()`
syncs, compile-friendly loops over static topology) stay contract-enforced.

## 5. Size accounting (evidence-based estimates)

| Area | Today | Target | How |
|---|---:|---:|---|
| `optim/` + `costs/` | 8,077 | ≤ 3,800 | phase 1 deletes 1,754 (legacy stack + graph executor) → ~6,300; phase 2 cuts ≈ −2,500: Adam→adapter (−215), implicit diet (−320), phase engine (−120), objectives (−120), shadow twins (−200), matrix-free + lstsq (−280), provider/kernels/solvers merges (−250), `problem.py`/`solver_lm.py`/`variables.py` internal rewrite shrinkage (−1,150, the builder+diet dividend), builder surface (+150) |
| residuals dual protocol | — | −400 | single protocol after legacy deletion |
| tasks validation walls | — | −250 | boundary policy, shared validators |
| stubs, aliases, dead branches | — | −200 | exports tell the truth (incl. residual stubs) |
| **src total** | **26,881** | **≈ 22,000** | |

Estimates are budgets: a work order that cannot meet its budget stops and
reports rather than redefining success.
