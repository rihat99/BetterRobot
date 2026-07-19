# {py:mod}`better_robot.optim.lm`

```{py:module} better_robot.optim.lm
```

```{autodoc2-docstring} better_robot.optim.lm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearizationDecision <better_robot.optim.lm.LinearizationDecision>`
  - ```{autodoc2-docstring} better_robot.optim.lm.LinearizationDecision
    :summary:
    ```
* - {py:obj}`LMStatus <better_robot.optim.lm.LMStatus>`
  - ```{autodoc2-docstring} better_robot.optim.lm.LMStatus
    :summary:
    ```
* - {py:obj}`LMState <better_robot.optim.lm.LMState>`
  - ```{autodoc2-docstring} better_robot.optim.lm.LMState
    :summary:
    ```
* - {py:obj}`LevenbergMarquardt <better_robot.optim.lm.LevenbergMarquardt>`
  - ```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt
    :summary:
    ```
* - {py:obj}`GaussNewton <better_robot.optim.lm.GaussNewton>`
  - ```{autodoc2-docstring} better_robot.optim.lm.GaussNewton
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearizationMode <better_robot.optim.lm.LinearizationMode>`
  - ```{autodoc2-docstring} better_robot.optim.lm.LinearizationMode
    :summary:
    ```
````

### API

````{py:data} LinearizationMode
:canonical: better_robot.optim.lm.LinearizationMode
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LinearizationMode
```

````

`````{py:class} LinearizationDecision
:canonical: better_robot.optim.lm.LinearizationDecision

```{autodoc2-docstring} better_robot.optim.lm.LinearizationDecision
```

````{py:attribute} requested
:canonical: better_robot.optim.lm.LinearizationDecision.requested
:type: better_robot.optim.lm.LinearizationMode
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LinearizationDecision.requested
```

````

````{py:attribute} used
:canonical: better_robot.optim.lm.LinearizationDecision.used
:type: typing.Literal[dense, banded]
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LinearizationDecision.used
```

````

````{py:attribute} reason
:canonical: better_robot.optim.lm.LinearizationDecision.reason
:type: better_robot.optim.temporal.LinearizationReason
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LinearizationDecision.reason
```

````

````{py:attribute} detail
:canonical: better_robot.optim.lm.LinearizationDecision.detail
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LinearizationDecision.detail
```

````

`````

`````{py:class} LMStatus()
:canonical: better_robot.optim.lm.LMStatus

Bases: {py:obj}`enum.IntEnum`

```{autodoc2-docstring} better_robot.optim.lm.LMStatus
```

````{py:attribute} RUNNING
:canonical: better_robot.optim.lm.LMStatus.RUNNING
:value: >
   0

```{autodoc2-docstring} better_robot.optim.lm.LMStatus.RUNNING
```

````

````{py:attribute} CONVERGED
:canonical: better_robot.optim.lm.LMStatus.CONVERGED
:value: >
   1

```{autodoc2-docstring} better_robot.optim.lm.LMStatus.CONVERGED
```

````

````{py:attribute} STALLED_AT_BOUNDS
:canonical: better_robot.optim.lm.LMStatus.STALLED_AT_BOUNDS
:value: >
   2

```{autodoc2-docstring} better_robot.optim.lm.LMStatus.STALLED_AT_BOUNDS
```

````

````{py:attribute} MAXITER
:canonical: better_robot.optim.lm.LMStatus.MAXITER
:value: >
   3

```{autodoc2-docstring} better_robot.optim.lm.LMStatus.MAXITER
```

````

````{py:attribute} FAILED
:canonical: better_robot.optim.lm.LMStatus.FAILED
:value: >
   4

```{autodoc2-docstring} better_robot.optim.lm.LMStatus.FAILED
```

````

`````

`````{py:class} LMState
:canonical: better_robot.optim.lm.LMState

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} better_robot.optim.lm.LMState
```

````{py:attribute} residual
:canonical: better_robot.optim.lm.LMState.residual
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.residual
```

````

````{py:attribute} robust_weights
:canonical: better_robot.optim.lm.LMState.robust_weights
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.robust_weights
```

````

````{py:attribute} cost
:canonical: better_robot.optim.lm.LMState.cost
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.cost
```

````

````{py:attribute} mu
:canonical: better_robot.optim.lm.LMState.mu
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.mu
```

````

````{py:attribute} increase_factor
:canonical: better_robot.optim.lm.LMState.increase_factor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.increase_factor
```

````

````{py:attribute} gain_ratio
:canonical: better_robot.optim.lm.LMState.gain_ratio
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.gain_ratio
```

````

````{py:attribute} gradient
:canonical: better_robot.optim.lm.LMState.gradient
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.gradient
```

````

````{py:attribute} grad_norm
:canonical: better_robot.optim.lm.LMState.grad_norm
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.grad_norm
```

````

````{py:attribute} projected_grad_norm
:canonical: better_robot.optim.lm.LMState.projected_grad_norm
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.projected_grad_norm
```

````

````{py:attribute} step_norm
:canonical: better_robot.optim.lm.LMState.step_norm
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.step_norm
```

````

````{py:attribute} relative_decrease
:canonical: better_robot.optim.lm.LMState.relative_decrease
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.relative_decrease
```

````

````{py:attribute} active_mask
:canonical: better_robot.optim.lm.LMState.active_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.active_mask
```

````

````{py:attribute} factorization_ok
:canonical: better_robot.optim.lm.LMState.factorization_ok
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.factorization_ok
```

````

````{py:attribute} converged
:canonical: better_robot.optim.lm.LMState.converged
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.converged
```

````

````{py:attribute} implicit_valid
:canonical: better_robot.optim.lm.LMState.implicit_valid
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.implicit_valid
```

````

````{py:attribute} status
:canonical: better_robot.optim.lm.LMState.status
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.status
```

````

````{py:attribute} iterations
:canonical: better_robot.optim.lm.LMState.iterations
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.iterations
```

````

````{py:attribute} scale
:canonical: better_robot.optim.lm.LMState.scale
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.scale
```

````

````{py:attribute} bound_state_index
:canonical: better_robot.optim.lm.LMState.bound_state_index
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.bound_state_index
```

````

````{py:attribute} bound_lower
:canonical: better_robot.optim.lm.LMState.bound_lower
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.bound_lower
```

````

````{py:attribute} bound_upper
:canonical: better_robot.optim.lm.LMState.bound_upper
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.bound_upper
```

````

````{py:attribute} bounded_mask
:canonical: better_robot.optim.lm.LMState.bounded_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LMState.bounded_mask
```

````

````{py:property} kkt_norm
:canonical: better_robot.optim.lm.LMState.kkt_norm
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim.lm.LMState.kkt_norm
```

````

`````

`````{py:class} LevenbergMarquardt
:canonical: better_robot.optim.lm.LevenbergMarquardt

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt
```

````{py:attribute} max_iter
:canonical: better_robot.optim.lm.LevenbergMarquardt.max_iter
:type: int
:value: >
   50

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.max_iter
```

````

````{py:attribute} gtol
:canonical: better_robot.optim.lm.LevenbergMarquardt.gtol
:type: float
:value: >
   1e-06

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.gtol
```

````

````{py:attribute} xtol
:canonical: better_robot.optim.lm.LevenbergMarquardt.xtol
:type: float
:value: >
   1e-09

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.xtol
```

````

````{py:attribute} ftol
:canonical: better_robot.optim.lm.LevenbergMarquardt.ftol
:type: float
:value: >
   1e-09

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.ftol
```

````

````{py:attribute} damping_parameter
:canonical: better_robot.optim.lm.LevenbergMarquardt.damping_parameter
:type: float
:value: >
   0.0001

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.damping_parameter
```

````

````{py:attribute} mu_min
:canonical: better_robot.optim.lm.LevenbergMarquardt.mu_min
:type: float
:value: >
   1e-12

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.mu_min
```

````

````{py:attribute} mu_max
:canonical: better_robot.optim.lm.LevenbergMarquardt.mu_max
:type: float
:value: >
   'float(...)'

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.mu_max
```

````

````{py:attribute} increase_factor_max
:canonical: better_robot.optim.lm.LevenbergMarquardt.increase_factor_max
:type: float
:value: >
   'float(...)'

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.increase_factor_max
```

````

````{py:attribute} bound_tolerance
:canonical: better_robot.optim.lm.LevenbergMarquardt.bound_tolerance
:type: float
:value: >
   1e-07

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.bound_tolerance
```

````

````{py:attribute} linear_solver
:canonical: better_robot.optim.lm.LevenbergMarquardt.linear_solver
:type: better_robot.optim.solvers.LinearSolver | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.linear_solver
```

````

````{py:attribute} linearization
:canonical: better_robot.optim.lm.LevenbergMarquardt.linearization
:type: better_robot.optim.lm.LinearizationMode
:value: >
   'auto'

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.linearization
```

````

````{py:attribute} kernel
:canonical: better_robot.optim.lm.LevenbergMarquardt.kernel
:type: better_robot.optim.kernels.RobustKernel
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.kernel
```

````

````{py:attribute} jacobian_strategy
:canonical: better_robot.optim.lm.LevenbergMarquardt.jacobian_strategy
:type: better_robot.optim.problem.JacobianStrategy
:value: >
   'auto'

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.jacobian_strategy
```

````

````{py:attribute} fixed_damping
:canonical: better_robot.optim.lm.LevenbergMarquardt.fixed_damping
:type: bool
:value: >
   False

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.fixed_damping
```

````

````{py:attribute} block_step_limits
:canonical: better_robot.optim.lm.LevenbergMarquardt.block_step_limits
:type: tuple[tuple[str, float], ...]
:value: >
   ()

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.block_step_limits
```

````

````{py:method} resolve_linearization(problem: better_robot.optim.problem.Problem) -> better_robot.optim.lm.LinearizationDecision
:canonical: better_robot.optim.lm.LevenbergMarquardt.resolve_linearization

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.resolve_linearization
```

````

````{py:method} init_state(values: better_robot.optim.variables.Values, problem: better_robot.optim.problem.Problem, *, create_graph: bool = False) -> better_robot.optim.lm.LMState
:canonical: better_robot.optim.lm.LevenbergMarquardt.init_state

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.init_state
```

````

````{py:method} update(values: better_robot.optim.variables.Values, state: better_robot.optim.lm.LMState, problem: better_robot.optim.problem.Problem, *, create_graph: bool = False) -> tuple[better_robot.optim.variables.Values, better_robot.optim.lm.LMState]
:canonical: better_robot.optim.lm.LevenbergMarquardt.update

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.update
```

````

````{py:method} finalize(values: better_robot.optim.variables.Values, state: better_robot.optim.lm.LMState, problem: better_robot.optim.problem.Problem, *, create_graph: bool = False) -> tuple[better_robot.optim.variables.Values, better_robot.optim.lm.LMState]
:canonical: better_robot.optim.lm.LevenbergMarquardt.finalize

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.finalize
```

````

````{py:method} run(values: better_robot.optim.variables.Values, problem: better_robot.optim.problem.Problem, state: better_robot.optim.lm.LMState | None = None) -> tuple[better_robot.optim.variables.Values, better_robot.optim.lm.LMState]
:canonical: better_robot.optim.lm.LevenbergMarquardt.run

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.run
```

````

````{py:method} solve(values: better_robot.optim.variables.Values, problem: better_robot.optim.problem.Problem, state: better_robot.optim.lm.LMState | None = None, *, differentiate: typing.Literal[detached, implicit] = 'detached', implicit_config: better_robot.optim.implicit.ImplicitDiffConfig | None = None) -> tuple[better_robot.optim.variables.Values, better_robot.optim.lm.LMState]
:canonical: better_robot.optim.lm.LevenbergMarquardt.solve

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.solve
```

````

`````

`````{py:class} GaussNewton
:canonical: better_robot.optim.lm.GaussNewton

Bases: {py:obj}`better_robot.optim.lm.LevenbergMarquardt`

```{autodoc2-docstring} better_robot.optim.lm.GaussNewton
```

````{py:attribute} damping_parameter
:canonical: better_robot.optim.lm.GaussNewton.damping_parameter
:type: float
:value: >
   1e-09

```{autodoc2-docstring} better_robot.optim.lm.GaussNewton.damping_parameter
```

````

````{py:attribute} fixed_damping
:canonical: better_robot.optim.lm.GaussNewton.fixed_damping
:type: bool
:value: >
   True

```{autodoc2-docstring} better_robot.optim.lm.GaussNewton.fixed_damping
```

````

`````
