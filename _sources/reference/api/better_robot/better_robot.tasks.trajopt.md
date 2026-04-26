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
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.iters
```

````

````{py:attribute} converged
:canonical: better_robot.tasks.trajopt.TrajOptResult.converged
:type: bool
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajopt.TrajOptResult.converged
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

`````

````{py:function} solve_trajopt(model: better_robot.data_model.model.Model, *, horizon: int, dt: float, initial_q_traj: torch.Tensor, cost_stack: better_robot.costs.stack.CostStack, optimizer: better_robot.optim.optimizers.base.Optimizer, max_iter: int = 50, jacobian_strategy: better_robot.kinematics.jacobian_strategy.JacobianStrategy = JacobianStrategy.AUTO, lower: torch.Tensor | None = None, upper: torch.Tensor | None = None, parameterization: better_robot.tasks.parameterization.TrajectoryParameterization | None = None) -> better_robot.tasks.trajopt.TrajOptResult
:canonical: better_robot.tasks.trajopt.solve_trajopt

```{autodoc2-docstring} better_robot.tasks.trajopt.solve_trajopt
```
````
