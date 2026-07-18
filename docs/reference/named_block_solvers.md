# Optimization solver API

The symbols on this page are imported from `better_robot.optim` and operate on
a named-block `Problem`.

```{py:module} better_robot.optim
:no-index:
```

## Levenberg--Marquardt

`````{py:class} LevenbergMarquardt(*, max_iter=50, gtol=1e-6, xtol=1e-9, ftol=1e-9, damping_parameter=1e-4, mu_min=1e-12, mu_max=4294967296.0, increase_factor_max=4294967296.0, bound_tolerance=1e-7, linear_solver=None, linearization="auto", kernel=L2(), jacobian_strategy="auto", fixed_damping=False, block_step_limits=())
:canonical: better_robot.optim.LevenbergMarquardt

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt
```

`linearization` is `"auto"`, `"dense"`, or `"structured"`. Automatic mode
uses block-banded assembly only when the complete problem is directly
eligible, and otherwise uses dense Cholesky. An explicit incompatible solver
or forced ineligible structured route raises.

`mu_min` must be strictly positive. A failed factorization escalates damping
through `mu_max` and receives one solve attempt at that cap before the element
becomes `FAILED`. `block_step_limits` caps named physical tangent-block norms.

````{py:method} init_state(values, problem, *, create_graph=False) -> LMState
:canonical: better_robot.optim.LevenbergMarquardt.init_state

Validate the solve boundary, build the static tangent/bound layout, and return
the initial tensor state.
````

````{py:method} update(values, state, problem, *, create_graph=False) -> tuple[Values, LMState]
:canonical: better_robot.optim.LevenbergMarquardt.update

Apply one fixed-shape batched update. `create_graph=True` is the small explicit
unrolled-differentiation oracle.
````

````{py:method} finalize(values, state, problem, *, create_graph=False) -> tuple[Values, LMState]
:canonical: better_robot.optim.LevenbergMarquardt.finalize

Refresh residual, robust-weight, active-set, and KKT artifacts at the returned
point. Call this after a manually driven update loop.
````

````{py:method} run(values, problem, state=None) -> tuple[Values, LMState]
:canonical: better_robot.optim.LevenbergMarquardt.run

Run the detached eager loop. A compatible prior state warm-starts damping;
target-dependent artifacts are recomputed.
````

````{py:method} solve(values, problem, state=None, *, differentiate="detached", implicit_config=None) -> tuple[Values, LMState]
:canonical: better_robot.optim.LevenbergMarquardt.solve

Use the detached solve by default, or pass `differentiate="implicit"` to
attach a guarded first-order implicit backward to an eligible converged
optimum. Invalid terminal states, unstable active bounds, robust kinks,
quaternion principal-log cuts, oversize/routing violations, and singular
systems raise `ImplicitDifferentiationError`.
````

`````

## Gauss--Newton

`````{py:class} GaussNewton(**kwargs)
:canonical: better_robot.optim.GaussNewton

```{autodoc2-docstring} better_robot.optim.lm.GaussNewton
```

The fixed-damping preset of the same guarded lifecycle, not an independent
solver loop.
`````

## First-order adapter

`````{py:function} run_first_order(values, problem, optimizer_factory, *, max_iter, tolerance, weights=None) -> tuple[Values, FirstOrderResult]
:canonical: better_robot.optim.run_first_order

```{autodoc2-docstring} better_robot.optim.first_order.run_first_order
```

The factory receives the persistent tangent parameters and returns an ordinary
`torch.optim.Optimizer`. After every step, the adapter retracts onto the
product manifold, applies bounds, and rebases the tangent buffers while
preserving optimizer state.
`````

`````{py:class} FirstOrderResult
:canonical: better_robot.optim.FirstOrderResult

Detached per-element `step`, `converged`, and `cost` tensors.
`````

## State and status

`````{py:class} LMState
:canonical: better_robot.optim.LMState

Fixed-structure tensor-only state containing the residual, robust weights,
cost and damping state, projected KKT quantities, factorization data,
per-element status and iteration counts, and the static bound layout needed by
`update`.
`````

`````{py:class} LMStatus
:canonical: better_robot.optim.LMStatus

Per-element status enum: `RUNNING`, `CONVERGED`, `STALLED_AT_BOUNDS`,
`MAXITER`, or `FAILED`. The corresponding state leaf is a `torch.int8` tensor
with the solve's leading batch shape.
`````

See {doc}`/concepts/solver_stack` for problem construction, robust grouping,
temporal routing, bounds, warm starts, and non-finite behavior.
