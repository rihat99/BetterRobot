# Named-block solver API

The symbols on this page are imported from `better_robot.optim`. They solve a
named-block `Problem`; the identically named classes under
`better_robot.optim.optimizers` are legacy `LeastSquaresProblem` optimizers.

```{py:module} better_robot.optim
:no-index:
```

## Levenberg–Marquardt

`````{py:class} LevenbergMarquardt(*, max_iter=50, gtol=1e-6, xtol=1e-9, ftol=1e-9, damping_parameter=1e-4, mu_min=1e-12, mu_max=4294967296.0, increase_factor_max=4294967296.0, bound_tolerance=1e-7, linear_solver=Cholesky(), kernel=L2(), jacobian_strategy="auto", fixed_damping=False, block_step_limits=())
:canonical: better_robot.optim.LevenbergMarquardt

```{autodoc2-docstring} better_robot.optim.blocks.solver_lm.LevenbergMarquardt
```

`mu_min` must be strictly positive. A failed factorization escalates damping
through `mu_max` and receives one solve attempt at that cap before the element
becomes `FAILED`.

`block_step_limits=(("translation", 0.2), ...)` caps each configured
variable's physical tangent L2 norm before both the normal-equation and
projected-gradient retractions. Names must identify blocks with at least one
free tangent coordinate. Gain prediction continues to use the actual
post-retraction tangent step.

````{py:method} init_state(values, problem, *, create_graph=False) -> LMState
:canonical: better_robot.optim.LevenbergMarquardt.init_state

Validate the solve boundary, build the static tangent/bound layout, and return
the initial tensor state. Set `create_graph=True` only for the explicit
small-problem unrolled differentiation oracle.
````

````{py:method} update(values, state, problem, *, create_graph=False) -> tuple[Values, LMState]
:canonical: better_robot.optim.LevenbergMarquardt.update

Apply one pure, sync-free, fixed-shape batched update. M6's internal
experimental CUDA-graph harness tests replay of fixed update groups; the
public `run` loop remains eager.
`create_graph=True` preserves the graph through this step for small reference
problems; the default does not retain the Jacobian graph.
````

````{py:method} finalize(values, state, problem, *, create_graph=False) -> tuple[Values, LMState]
:canonical: better_robot.optim.LevenbergMarquardt.finalize

Refresh residual, robust-weight, active-set, and KKT artifacts at the returned
point. Call this after a manually driven update loop, passing
`create_graph=True` when closing an explicit unrolled oracle.
````

````{py:method} run(values, problem, state=None) -> tuple[Values, LMState]
:canonical: better_robot.optim.LevenbergMarquardt.run

Run the detached eager loop. A supplied prior state warm-starts damping while
target-dependent artifacts are recomputed. `run` intentionally has no
`create_graph` mode and returns graph-free values and state. Warm starts reject
batch-shape, dtype, or device mismatches instead of coercing damping state.
````

````{py:method} solve(values, problem, state=None, *, differentiate="detached", implicit_config=None) -> tuple[Values, LMState]
:canonical: better_robot.optim.LevenbergMarquardt.solve

Use the detached solve by default, or pass `differentiate="implicit"` to
attach a first-order implicit backward to a converged dense optimum. Only
explicitly declared external parameters receive gradients; initialization,
warm state, bounds/masks, and hyperparameters do not. Invalid terminal states,
unstable active bounds, nonsmooth Huber points, terminal-manifold quaternion
representatives at the absolute-pi principal-log cut, and singular systems
raise `ImplicitDifferentiationError` during backward. An optimized input and
external parameter must not be the same tensor object. `ImplicitDiffConfig`
controls the dense size cap and the explicit small-banded dense-oracle opt-in.

Principal-log cuts hidden inside arbitrary residual/provider relative-rotation
code are not automatically certified; their smooth-domain guard remains the
custom author's responsibility.

Declared differentiable parameters must reach terminal optimality through a
stable named context read. Object identity with an item weight or kernel
attribute is not a binding and is rejected as disconnected rather than given a
silent zero gradient.
````

`````

## Gauss–Newton

`````{py:class} GaussNewton(**kwargs)
:canonical: better_robot.optim.GaussNewton

```{autodoc2-docstring} better_robot.optim.blocks.solver_lm.GaussNewton
```

This is the fixed-damping preset of the same guarded update, not an independent
solver loop.
`````

The `create_graph=True` lifecycle remains an explicitly unrolled correctness
oracle. It is distinct from `solve(..., differentiate="implicit")`, whose
forward trajectory is detached and whose custom backward differentiates the
terminal robust optimality system.

## Adam

`````{py:class} Adam(*, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-6, max_iter=100)
:canonical: better_robot.optim.Adam

```{autodoc2-docstring} better_robot.optim.blocks.solver_adam.Adam
```

````{py:method} init_state(values, problem) -> AdamState
:canonical: better_robot.optim.Adam.init_state

Validate the solve boundary and initialize one mask-reduced first/second moment
tensor per named block.
````

````{py:method} update(values, state, problem) -> tuple[Values, AdamState]
:canonical: better_robot.optim.Adam.update

Apply one host-sync-free tangent-gradient step. This path calls the
prevalidated objective VJP and feasible manifold retraction; it never calls
Jacobian-block, dense-Jacobian, or normal-matrix assembly.
````

````{py:method} run(values, problem, state=None) -> tuple[Values, AdamState]
:canonical: better_robot.optim.Adam.run

Run the detached eager loop. A compatible state retains reduced moments and
bias-correction step counts while current cost, gradient norm, convergence,
and status are refreshed. Name/order, reduced shape, batch shape, dtype, and
device mismatches are rejected.
````

`````

`````{py:class} AdamState
:canonical: better_robot.optim.AdamState

A fixed tensor pytree containing reduced `m`/`v` mappings plus per-element
step, cost, gradient norm, convergence, and status tensors.
`````

`````{py:class} AdamStatus
:canonical: better_robot.optim.AdamStatus

Per-element status enum: `RUNNING`, `CONVERGED`, `MAXITER`, or `FAILED`.
`````

Named-block LBFGS is deliberately unavailable: batching its per-element
histories, line search, and curvature-validity/history-reset behavior is a
deferred design problem. Use `Adam` for matrix-free first-order work or LM/GN
when dense second-order assembly is appropriate.

## Phases

`````{py:class} Phase(name, iters, optimizer, weight_overrides={}, mask_overrides={}, on_start=None)
:canonical: better_robot.optim.Phase

```{autodoc2-docstring} better_robot.optim.blocks.phase.Phase
`````

`````{py:function} run_phases(problem, values, phases) -> PhaseResult
:canonical: better_robot.optim.run_phases

Runs phase-local functional `Problem` views in order, carries only values
between phases, and creates fresh solver state for every nonempty phase. The
input problem is never mutated.
`````

`````{py:class} PhaseResult
:canonical: better_robot.optim.PhaseResult

Final values plus one solver state (or `None` for a zero-iteration phase) per
phase.
`````

## State and status

`````{py:class} LMState
:canonical: better_robot.optim.LMState

Fixed-structure tensor-only state. Its fields include residual and robust
weights, cost and damping, gain/step/convergence diagnostics, projected KKT
quantities, active/factorization masks, per-element status and iteration
counts, plus the static scale/bound layout needed by `update`.
`````

`````{py:class} LMStatus
:canonical: better_robot.optim.LMStatus

Per-element status enum: `RUNNING`, `CONVERGED`, `STALLED_AT_BOUNDS`,
`MAXITER`, or `FAILED`. The corresponding `LMState.status` leaf is a
`torch.int8` tensor with the solve's leading batch shape.

Accepted-step `xtol`/`ftol` termination is available only for unbounded
problems. A bounded problem requires the projected-gradient KKT test for every
success status. A tolerance-only unbounded `CONVERGED` result remains
`implicit_valid=False` unless final evaluation also satisfies KKT.
`````

See {doc}`/concepts/solver_stack` for robust grouping, supported bounds, warm
starts, custom-residual capture eligibility, and the legacy/new migration
boundary.
