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

`````{py:class} LevenbergMarquardt(problem: better_robot.optim.problem.Problem, *, solver: typing.Literal[auto] | better_robot.optim.solvers.LinearSolver = 'auto', max_iterations: int = 50, tolerance: float = 1e-06, step_tolerance: float = 1e-09, relative_tolerance: float = 1e-09, damping: float = 0.0001, mu_min: float = 1e-12, mu_max: float = float(2**32), increase_factor_max: float = float(2**32), bound_tolerance: float = 1e-07, linearization: better_robot.optim.lm.LinearizationMode = 'auto', jacobian_strategy: better_robot.optim.problem.JacobianStrategy = 'auto', fixed_damping: bool = False)
:canonical: better_robot.optim.lm.LevenbergMarquardt

Bases: {py:obj}`better_robot.optim.optimizers.Optimizer`

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt
```

````{py:method} resolve_linearization(problem: better_robot.optim.problem.Problem) -> better_robot.optim.lm.LinearizationDecision
:canonical: better_robot.optim.lm.LevenbergMarquardt.resolve_linearization

```{autodoc2-docstring} better_robot.optim.lm.LevenbergMarquardt.resolve_linearization
```

````

````{py:method} reset() -> None
:canonical: better_robot.optim.lm.LevenbergMarquardt.reset

````

````{py:method} step() -> better_robot.optim.optimizers.OptimizerInfo
:canonical: better_robot.optim.lm.LevenbergMarquardt.step

````

````{py:method} optimize(*, verbose: bool = False, differentiate: typing.Literal[implicit] | None = None, implicit_config: better_robot.optim.implicit.ImplicitDiffConfig | None = None) -> better_robot.optim.optimizers.OptimizerInfo
:canonical: better_robot.optim.lm.LevenbergMarquardt.optimize

````

`````

````{py:class} GaussNewton(problem: better_robot.optim.problem.Problem, **kwargs)
:canonical: better_robot.optim.lm.GaussNewton

Bases: {py:obj}`better_robot.optim.lm.LevenbergMarquardt`

```{autodoc2-docstring} better_robot.optim.lm.GaussNewton
```

````
