# M2b bounded least-squares survey

**Decision (accepted 2026-07-17):** implement a **projected active-set LM with
a model-selected projected-gradient safeguard**, reusing M2b's scaled
Madsen--Nielsen update. The owner authorized accepting recommended decisions
while away, so this recommendation was adopted. Full reflective TRF remains
the SciPy reference, not an M2b implementation.

This note is evidence for T2b.5, not solver code. It was written against the
landed M2a `VarSpec`/`RobotConfig` contracts and the local SciPy 1.17.1 TRF
source. No BHF or BVR source or state was changed.

## Evidence inspected

- SciPy 1.17.1
  `.venv/lib/python3.11/site-packages/scipy/optimize/_lsq/trf.py` and
  `_lsq/common.py`, including `select_step`, `trf_bounds`,
  `CL_scaling_vector`, `step_size_to_bound`, and
  `make_strictly_feasible`.
- Local JAXopt `references/optim/jaxopt/jaxopt/_src/lbfgsb.py`, especially
  `_get_error`, `_find_cauchy_point`, and `_minimize_subspace`. JAXopt has no
  bounded nonlinear-least-squares solver; only its projected-gradient/KKT
  fixed-point test is directly reusable here.
- `plan/03_architecture.md` section 4, the corrected failure account in
  `plan/01_assessment.md` section 1.4, and the T2b.5/P1--P3 requirements in
  `plan/for_agents/m2b_batched_second_order_solvers.md`.
- Landed M2a behavior in `optim/blocks/manifolds.py`, `variables.py`, and
  `problem.py`: bounds are state-space bounds; masks/scales are reduced
  tangent-space data; public retraction validates the starting value and then
  projects the result.

The Branch--Coleman--Li and Kanzow--Yamashita--Fukushima papers and Ceres
source are not vendored. SciPy is the inspected reflective reference. The
projected recommendation below is therefore a documented local design, not a
claim of being a line-for-line Kanzow or Ceres port.

## Comparison

| Criterion | Reflective trust region (SciPy TRF) | Projected/active-set LM (recommended) |
|---|---|---|
| Core step | Coleman--Li distance scaling, a trust-region solve, then choose the best quadratic model among the feasible truncated step, one-bound reflection, and constrained anti-gradient step. | Identify outward-active coordinates, solve the existing damped normal system on the inactive subspace, project/retract once, and compare its model prediction with a projected-gradient safeguard before evaluating one residual candidate. |
| Batchability | Most primitives can be tensorized, but the reference has a nested retry loop until positive actual reduction, per-element trust-region root solves, first-hit/reflection logic, and best-of-three selection. Fixed-trip masked emulation is possible but is a separate solver design. | Masks, restricted normal systems, projection, candidate selection, KKT tests, and damping updates are all fixed-shape `torch.where`/reduction work. One batched factorization and one residual candidate remain the iteration shape. |
| Capture suitability | Poor for a faithful first implementation. SciPy branches on feasibility and reduction, performs an element-dependent inner loop, and its two trust-region solvers use SVD/root/iterative machinery. Removing host control flow without changing the algorithm is substantial work. | Good by construction if the M2a validation path is kept outside `update`: no dynamic element selection, no inner loop, no data-dependent tensor shapes, and the same operations run for every batch element. M6 still owns actual capture certification. |
| Initial/boundary points | Requires strict interior iterates because Coleman--Li scaling is singular at a bound. SciPy rejects `lower >= upper`, shifts a feasible boundary `x0` inward, and uses `theta` to step back from bounds. | Naturally accepts feasible points exactly at a bound and fixed bounds (`lower == upper`). This matches M2a, which allows both and rejects only infeasible initial values. |
| M2a manifold fit | Coleman--Li assumes bounds and steps share one coordinate system. M2a deliberately stores `nq` state bounds separately from `nv` local tangents, so a faithful extension needs a state-to-local constraint Jacobian and strict-interior policy. | Feasibility and the actually taken step can be obtained through retraction/projection and `difference`, preserving `nq != nv`. Axis-aligned active masks still require an explicit capability fence for non-axis-aligned local coordinates (see edge cases). |
| Interaction with Madsen--Nielsen | Replaces the planned damping update with trust-radius and inner trust-region parameter state; combining both would be redundant and would need new evidence. | Reuses the already required scaled LM solve, per-element `mu`, gain ratio, `cholesky_ex` fallback, and status machinery. |
| Implementation/risk | High: it adds a second globalization mechanism and much more per-step state. A partial port risks being neither SciPy TRF nor the planned capture-ready LM. | Medium: the main new pieces are active-set construction, a restricted system, an honest projected model, projected KKT residual, and a safeguarded direction. These compose with T2b.1--T2b.4. |
| Measured evidence | SciPy TRF used 20 Jacobian iterations on P1 and 5 on P3. | BetterRobot used 40 iterations on P1 and 4 on P3; the restricted 2-DOF toy reached the analytic constrained optimum where clamp-only LM did not. P2 was not run because its proposed fixture does not establish pose unreachability. |

The decisive point is not that reflection is unbatchable. Its elementary
operations are batchable. The issue is that a faithful TRF port brings a
trust-region subsolver, strict-interior semantics, a retry loop, and a second
set of update rules into a milestone that already specifies scaled LM and
Madsen--Nielsen damping. The projected active-set design repairs the bounds
failure inside that existing update shape.

## Recommended step contract

Use one reduced physical tangent layout after M2a mask elimination. For an
IRLS-weighted residual/Jacobian, let

```text
g = J^T r                         physical reduced-tangent gradient
H = J^T J                         Gauss--Newton model
s > 0                             VarSpec free scales
J_s = J diag(s),  g_s = diag(s) g scaled system
F in {0,1}^nt                     1 for a currently movable coordinate
```

All of these carry arbitrary leading batch axes. An outward-active coordinate
is at its lower bound with `g_i > 0`, or at its upper bound with `g_i < 0`.
Fixed `VarSpec.mask` coordinates are already absent and must not be added back
as solver-active coordinates.

### 1. Restricted LM direction

Zeroing the step only *after* an unrestricted solve is wrong because active
columns still alter the free solution. Restrict both sides of the system:

```text
Ff  = F cast to the working dtype
H_A = Ff H_s Ff + diag(Ff * mu + (1 - Ff))
b_A = -Ff * g_s
d_s = solve(H_A, b_A)
d_lm = s * (Ff * d_s)
```

The unit diagonal on active coordinates keeps the fixed-shape matrix
nonsingular even for the GN preset's zero/tiny damping. Continue to fold the
`cholesky_ex` success mask into candidate acceptance.

Retract and project once, then recover the step that was actually taken:

```text
x_lm = feasible_retract(x, d_lm)
p_lm = gather_free(difference(x, x_lm))
```

This last line is mandatory. `p_lm` can differ from `d_lm` when a previously
inactive coordinate reaches a bound, and `RobotConfig` can have `nq != nv`.

### 2. Projected-gradient safeguard

An active-set Newton/LM direction can still be poor when the active set is
changing. Compute a projected steepest-descent path in fixed tensor work:

```text
x_pg1 = feasible_retract(x, -g)              # or a documented positive
p_pg1 = gather_free(difference(x, x_pg1))    # diagonal preconditioned -g
den   = p_pg1^T H p_pg1
beta  = clamp(-(g^T p_pg1) / max(den, eps), 0, 1)
x_pg  = feasible_retract(x, beta * p_pg1)
p_pg  = gather_free(difference(x, x_pg))
```

Evaluate the local quadratic prediction for both directions,

```text
pred(p) = -(g^T p + 0.5 * p^T H p),
```

and select the direction with the larger finite positive prediction using
`torch.where`. This selection happens before residual evaluation, so the
solver still evaluates exactly one candidate. If both predictions are
non-positive, take a zero step, reject, and escalate damping unless the KKT
test already terminates the element.

This safeguard is what distinguishes the proposal from the current
projection-only LM plus a new status label. The active mask changes the normal
system, the actual projected step changes the model prediction, and a
projected first-order direction remains available while the active set moves.

### 3. Projection-consistent gain ratio

For either selected actual step `p`, use

```text
actual = robust_cost(x) - robust_cost(x_candidate)
pred   = -(g^T p + 0.5 * ||J p||^2)
rho    = actual / clamp(pred, min=eps)
```

The familiar Madsen--Nielsen expression
`0.5 * p^T (mu*p - g)` is equivalent only when `p` exactly solves the
unprojected damped normal system. It is not valid after projection or
reflection. The robust cost supplies the actual reduction; the matching IRLS
quadratic supplies the prediction. A non-finite candidate or non-positive
prediction must reject.

### 4. KKT residual and statuses

The local fixed-point residual is the JAXopt/LBFGS-B test expressed through
M2a's feasible retraction:

```text
x_g     = feasible_retract(x, -g)
kkt_vec = gather_free(difference(x, x_g))
kkt_inf = max(abs(kkt_vec))
```

For an axis-aligned Euclidean coordinate this is
`P_[l,u](x - g) - x`; it is zero exactly when an interior gradient is zero,
a lower-bound gradient points outward (`g >= 0`), or an upper-bound gradient
points outward (`g <= 0`). A positive diagonal preconditioner may replace the
unit step, but the choice must be fixed and documented because it changes the
numerical tolerance scale, though not the exact fixed points.

Status precedence per element should be:

1. non-finite residual/model, or a factorization that still fails on its one
   actual solve attempt at `mu_max`: `failed`;
2. `kkt_inf <= gtol` and an active constraint is responsible
   (`||g||_inf > gtol`): `stalled_at_bounds`;
3. `kkt_inf <= gtol` otherwise: `converged`;
4. exhausted iteration budget: `maxiter`;
5. otherwise: `running`.

Step-size and relative-cost tests may terminate an accepted step only for an
unbounded problem. They must not report constrained convergence while the KKT
residual is above tolerance. A tolerance-only unbounded success is not
eligible for implicit differentiation unless final evaluation also passes KKT.

## Edge cases the implementation must settle explicitly

1. **Initial feasibility.** Follow landed M2a: reject an infeasible `x0` with
   an actionable error. Do not silently project it inside `init_state`.
   Panda probes must start from explicitly clamped neutral. Boundary starts
   are valid and must not be shifted inward.
2. **Fixed-width bounds.** M2a permits `lower == upper`; active-set LM can
   handle this, while SciPy TRF rejects it. Prefer eliminating intentional
   fixed coordinates with `VarSpec.mask`; otherwise treat a fixed-width free
   coordinate as active for either gradient sign.
3. **Free-flyer translation is a current contract discrepancy.** T2b.5 says
   free-flyer base coordinates have no bounds, but landed M2a allows boxes on
   the three free-flyer translation state coordinates. Its right-local SE(3)
   tangent translation rotates with the base, so a world-axis state box is
   not an axis-aligned tangent mask. For M2b v1, fail fast when a
   `RobotConfig` bound reaches a free-flyer (including inside a composite),
   and defer a constraint-normal/Jacobian treatment. Do not silently zero
   three local tangent coordinates. Planar translation is additive and does
   not have this issue.
4. **Other manifolds.** SO(3), SE(3), spherical, and unbounded-revolute
   coordinates have no boxes under M2a. Their KKT vector is the ordinary
   tangent gradient. Quaternion/unit-circle coordinates must never be
   clamped.
5. **RobotConfig mapping.** For supported bounded revolute, prismatic,
   helical, translation, and planar-translation coordinates, build and test a
   static q-coordinate-to-local-v map once. Replicate it knot-major for
   `(T, nq)` events and apply M2a's free-coordinate mask afterward.
6. **Scales.** Active-set signs and KKT conditions use the physical gradient
   `g`; scales precondition the solve but must not change which physical bound
   is outward-active. Predicted reduction can use `(J, p)` or the exactly
   equivalent scaled pair, never a mixture.
7. **Robust grouping.** Build `g` and `H` from the same IRLS semantic groups
   used by M2b, and use the true grouped `rho` objective for actual reduction.
8. **NaN and factorization masks.** NaN gain comparisons reject naturally,
   but sanitize candidate steps before manifold retraction. Damping keeps a
   strictly positive floor so zero damping cannot deadlock recovery, and a
   failed factorization gets one actual attempt at `mu_max` before `FAILED`.
   A failed element must not contaminate a successful neighbor in the batch.
9. **All coordinates eliminated/active.** Avoid reductions over an empty
   tangent axis. Define a zero KKT norm; if nonzero cost remains because no
   movement is permitted, report `stalled_at_bounds` only when a real bound
   is responsible, otherwise use an explicit no-free-variables failure or
   convergence policy and test it.
10. **Capture boundary.** `Problem.retract`/`VarSpec.retract` currently run
    Python validation and tensor-to-host predicates. `init_state` may validate;
    `update` needs a prevalidated internal feasible-retraction path before it
    can satisfy T2b.7. This is a known M2a-to-M2b seam, not proof that public
    M2a retraction is capture-ready.

## Prototype and measurements

The landed projected active-set implementation produced the following CPU,
float32 evidence. The P6 duration is pytest wall time, not a solver benchmark
or a batched speedup claim.

| Case | BetterRobot projected active-set LM | SciPy TRF reference |
|---|---|---|
| P1 bounded-interior Panda | `CONVERGED`, 40 iterations, final cost `1.306e-12`, position error `1.58e-6 m` | 20 Jacobian iterations, final cost `1.096e-10` |
| P3 feasible regularized Panda | `CONVERGED`, 4 iterations, final cost `1.96226e-6`, position error `1.15e-4 m` | 5 Jacobian iterations, final cost `1.96220e-6` |
| P6, 128 targets | One `B=128` call passed the stated status, cost, and task-space parity protocol against 128 `B=1` calls in 16.1 s | Not applicable |

The coupled 2-DOF regression supplies the algorithmic ablation. At an outward
active first coordinate, unrestricted LM followed by clamping lands near
`[1, 1]` with cost `1.0`. Restricting the normal system reaches the analytic
constrained solution `[1, 1.6]` with cost `0.1`, active mask `[True, False]`,
and `STALLED_AT_BOUNDS`. A mixed-batch test independently reports
`[CONVERGED, STALLED_AT_BOUNDS]` for interior and constrained elements.

P2 is deliberately deferred. For a redundant Panda arm, taking FK at one
configuration pushed outside selected joint limits proves only that this
particular inverse configuration is infeasible; it does not prove that the
same end-effector pose has no other feasible inverse. A diagnostic solve found
a different feasible near-zero-cost IK solution for the attempted fixture.
Asserting `STALLED_AT_BOUNDS` would therefore encode a false premise. A future
P2 must use a geometrically certified unreachable pose before the KKT/status
assertion can be meaningful.

Not all originally requested trajectories were persisted: P1/P3 final
measurements and statuses are recorded, but their per-iteration active-count
and `mu` trajectories and a standalone safeguard-selection ablation are not.
The committed damping, restricted-system, mixed-batch, robust-direction, and
KKT tests cover those formulas independently. If a certified P2 later fails
after implementation bugs are excluded, revisit reflective TRF rather than
tuning tolerances or relabeling `maxiter`.

## Owner decision

**Accepted:** projected active-set LM plus the projected-gradient safeguard,
under the owner's instruction to accept recommended decisions autonomously.
Full reflective TRF is retained as the SciPy benchmark/reference and as the
fallback design if a future geometrically valid P2 rejects this choice. The
selected method preserves M2b's batched, one-candidate, Madsen--Nielsen,
capture-ready update architecture.
