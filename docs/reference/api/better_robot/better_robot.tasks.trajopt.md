# {py:mod}`better_robot.tasks.trajopt`

```{py:module} better_robot.tasks.trajopt
```

```{autodoc2-docstring} better_robot.tasks.trajopt
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrajOptResult <better_robot.tasks.trajopt.TrajOptResult>`
  - ```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`solve_trajopt <better_robot.tasks.trajopt.solve_trajopt>`
  - ```{autodoc2-docstring} better_robot.tasks.trajopt.solve_trajopt
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ResidualFactory <better_robot.tasks.trajopt.ResidualFactory>`
  - ```{autodoc2-docstring} better_robot.tasks.trajopt.ResidualFactory
    :summary:
    ```
* - {py:obj}`OptimizerFactory <better_robot.tasks.trajopt.OptimizerFactory>`
  - ```{autodoc2-docstring} better_robot.tasks.trajopt.OptimizerFactory
    :summary:
    ```
````

### API

`````{py:class} TrajOptResult
:canonical: better_robot.tasks.trajopt.TrajOptResult

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult
```

````{py:attribute} trajectory
:canonical: better_robot.tasks.trajopt.TrajOptResult.trajectory
:type: better_robot.tasks.trajectory.Trajectory
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.trajectory
```

````

````{py:attribute} residual
:canonical: better_robot.tasks.trajopt.TrajOptResult.residual
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.residual
```

````

````{py:attribute} iters
:canonical: better_robot.tasks.trajopt.TrajOptResult.iters
:type: int | torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.iters
```

````

````{py:attribute} converged
:canonical: better_robot.tasks.trajopt.TrajOptResult.converged
:type: bool | torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.converged
```

````

````{py:attribute} status
:canonical: better_robot.tasks.trajopt.TrajOptResult.status
:type: int | torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.status
```

````

````{py:attribute} model
:canonical: better_robot.tasks.trajopt.TrajOptResult.model
:type: better_robot.data_model.model.Model
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.model
```

````

````{py:attribute} linearization_requested
:canonical: better_robot.tasks.trajopt.TrajOptResult.linearization_requested
:type: better_robot.optim.LinearizationMode
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.linearization_requested
```

````

````{py:attribute} linearization_used
:canonical: better_robot.tasks.trajopt.TrajOptResult.linearization_used
:type: typing.Literal[dense, banded]
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.linearization_used
```

````

````{py:attribute} linearization_reason
:canonical: better_robot.tasks.trajopt.TrajOptResult.linearization_reason
:type: better_robot.optim.LinearizationReason
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.linearization_reason
```

````

````{py:attribute} linearization_detail
:canonical: better_robot.tasks.trajopt.TrajOptResult.linearization_detail
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.linearization_detail
```

````

`````

````{py:data} ResidualFactory
:canonical: better_robot.tasks.trajopt.ResidualFactory
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.ResidualFactory
```

````

````{py:data} OptimizerFactory
:canonical: better_robot.tasks.trajopt.OptimizerFactory
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.OptimizerFactory
```

````

````{py:function} solve_trajopt(model: better_robot.data_model.model.Model, *, dt: float, initial_q_traj: torch.Tensor | better_robot.optim.RobotVariable, residuals: collections.abc.Sequence[better_robot.optim.Residual | better_robot.tasks.trajopt.ResidualFactory], optimizer: better_robot.tasks.trajopt.OptimizerFactory | None = None, max_iter: int = 50, jacobian_strategy: better_robot.optim.JacobianStrategy = 'auto', lower: torch.Tensor | None = None, upper: torch.Tensor | None = None) -> better_robot.tasks.trajopt.TrajOptResult
:canonical: better_robot.tasks.trajopt.solve_trajopt

```{autodoc2-docstring} better_robot.tasks.trajopt.solve_trajopt
```
````
