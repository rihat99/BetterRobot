# {py:mod}`better_robot.optim.state`

```{py:module} better_robot.optim.state
```

```{autodoc2-docstring} better_robot.optim.state
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SolverState <better_robot.optim.state.SolverState>`
  - ```{autodoc2-docstring} better_robot.optim.state.SolverState
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SolverStatus <better_robot.optim.state.SolverStatus>`
  - ```{autodoc2-docstring} better_robot.optim.state.SolverStatus
    :summary:
    ```
````

### API

````{py:data} SolverStatus
:canonical: better_robot.optim.state.SolverStatus
:value: >
   None

```{autodoc2-docstring} better_robot.optim.state.SolverStatus
```

````

`````{py:class} SolverState
:canonical: better_robot.optim.state.SolverState

```{autodoc2-docstring} better_robot.optim.state.SolverState
```

````{py:attribute} x
:canonical: better_robot.optim.state.SolverState.x
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.state.SolverState.x
```

````

````{py:attribute} residual
:canonical: better_robot.optim.state.SolverState.residual
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.state.SolverState.residual
```

````

````{py:attribute} residual_norm
:canonical: better_robot.optim.state.SolverState.residual_norm
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.state.SolverState.residual_norm
```

````

````{py:attribute} iters
:canonical: better_robot.optim.state.SolverState.iters
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.optim.state.SolverState.iters
```

````

````{py:attribute} damping
:canonical: better_robot.optim.state.SolverState.damping
:type: float
:value: >
   0.0

```{autodoc2-docstring} better_robot.optim.state.SolverState.damping
```

````

````{py:attribute} gain_ratio
:canonical: better_robot.optim.state.SolverState.gain_ratio
:type: float | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.state.SolverState.gain_ratio
```

````

````{py:attribute} status
:canonical: better_robot.optim.state.SolverState.status
:type: better_robot.optim.state.SolverStatus
:value: >
   'running'

```{autodoc2-docstring} better_robot.optim.state.SolverState.status
```

````

````{py:attribute} history
:canonical: better_robot.optim.state.SolverState.history
:type: list[dict]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.optim.state.SolverState.history
```

````

````{py:property} converged
:canonical: better_robot.optim.state.SolverState.converged
:type: bool

```{autodoc2-docstring} better_robot.optim.state.SolverState.converged
```

````

````{py:method} from_problem(problem: better_robot.optim.problem.LeastSquaresProblem) -> better_robot.optim.state.SolverState
:canonical: better_robot.optim.state.SolverState.from_problem
:classmethod:

```{autodoc2-docstring} better_robot.optim.state.SolverState.from_problem
```

````

`````
