# Named-block solver API

The symbols on this page are imported from `better_robot.optim`. They solve a
named-block `Problem`; the identically named classes under
`better_robot.optim.optimizers` are legacy `LeastSquaresProblem` optimizers.

```{py:module} better_robot.optim
:no-index:
```

## Levenberg–Marquardt

`````{py:class} LevenbergMarquardt(*, max_iter=50, gtol=1e-6, xtol=1e-9, ftol=1e-9, damping_parameter=1e-4, mu_min=1e-12, mu_max=4294967296.0, increase_factor_max=4294967296.0, bound_tolerance=1e-7, linear_solver=Cholesky(), kernel=L2(), jacobian_strategy="auto", fixed_damping=False)
:canonical: better_robot.optim.LevenbergMarquardt

```{autodoc2-docstring} better_robot.optim.blocks.solver_lm.LevenbergMarquardt
```

`mu_min` must be strictly positive. A failed factorization escalates damping
through `mu_max` and receives one solve attempt at that cap before the element
becomes `FAILED`.

````{py:method} init_state(values, problem, *, create_graph=False) -> LMState
:canonical: better_robot.optim.LevenbergMarquardt.init_state

Validate the solve boundary, build the static tangent/bound layout, and return
the initial tensor state. Set `create_graph=True` only for the explicit
small-problem unrolled differentiation oracle.
````

````{py:method} update(values, state, problem, *, create_graph=False) -> tuple[Values, LMState]
:canonical: better_robot.optim.LevenbergMarquardt.update

Apply one pure, sync-free, fixed-shape batched update. This is capture-ready by
construction; M6's actual CUDA capture/replay parity test owns certification.
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

`````

## Gauss–Newton

`````{py:class} GaussNewton(**kwargs)
:canonical: better_robot.optim.GaussNewton

```{autodoc2-docstring} better_robot.optim.blocks.solver_lm.GaussNewton
```

This is the fixed-damping preset of the same guarded update, not an independent
solver loop.
`````

The `create_graph=True` path is a correctness oracle for a few explicitly
unrolled steps, not a stable solver-differentiation guarantee. Implicit
backward, active-set stability checks, and production differentiation through
a complete solve remain M6 work.

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
