# {py:mod}`better_robot.tasks.ik`

```{py:module} better_robot.tasks.ik
```

```{autodoc2-docstring} better_robot.tasks.ik
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IKCostConfig <better_robot.tasks.ik.IKCostConfig>`
  - ```{autodoc2-docstring} better_robot.tasks.ik.IKCostConfig
    :summary:
    ```
* - {py:obj}`OptimizerConfig <better_robot.tasks.ik.OptimizerConfig>`
  - ```{autodoc2-docstring} better_robot.tasks.ik.OptimizerConfig
    :summary:
    ```
* - {py:obj}`IKResult <better_robot.tasks.ik.IKResult>`
  - ```{autodoc2-docstring} better_robot.tasks.ik.IKResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`solve_ik <better_robot.tasks.ik.solve_ik>`
  - ```{autodoc2-docstring} better_robot.tasks.ik.solve_ik
    :summary:
    ```
````

### API

`````{py:class} IKCostConfig
:canonical: better_robot.tasks.ik.IKCostConfig

```{autodoc2-docstring} better_robot.tasks.ik.IKCostConfig
```

````{py:attribute} pos_weight
:canonical: better_robot.tasks.ik.IKCostConfig.pos_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} better_robot.tasks.ik.IKCostConfig.pos_weight
```

````

````{py:attribute} ori_weight
:canonical: better_robot.tasks.ik.IKCostConfig.ori_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} better_robot.tasks.ik.IKCostConfig.ori_weight
```

````

````{py:attribute} pose_weight
:canonical: better_robot.tasks.ik.IKCostConfig.pose_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} better_robot.tasks.ik.IKCostConfig.pose_weight
```

````

````{py:attribute} limit_weight
:canonical: better_robot.tasks.ik.IKCostConfig.limit_weight
:type: float
:value: >
   0.1

```{autodoc2-docstring} better_robot.tasks.ik.IKCostConfig.limit_weight
```

````

````{py:attribute} rest_weight
:canonical: better_robot.tasks.ik.IKCostConfig.rest_weight
:type: float
:value: >
   0.01

```{autodoc2-docstring} better_robot.tasks.ik.IKCostConfig.rest_weight
```

````

````{py:attribute} collision_margin
:canonical: better_robot.tasks.ik.IKCostConfig.collision_margin
:type: float
:value: >
   0.02

```{autodoc2-docstring} better_robot.tasks.ik.IKCostConfig.collision_margin
```

````

````{py:attribute} collision_weight
:canonical: better_robot.tasks.ik.IKCostConfig.collision_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} better_robot.tasks.ik.IKCostConfig.collision_weight
```

````

````{py:attribute} q_rest
:canonical: better_robot.tasks.ik.IKCostConfig.q_rest
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.ik.IKCostConfig.q_rest
```

````

`````

`````{py:class} OptimizerConfig
:canonical: better_robot.tasks.ik.OptimizerConfig

```{autodoc2-docstring} better_robot.tasks.ik.OptimizerConfig
```

````{py:attribute} optimizer
:canonical: better_robot.tasks.ik.OptimizerConfig.optimizer
:type: typing.Literal[lm, gn, adam, lbfgs, lm_then_lbfgs]
:value: >
   'lm'

```{autodoc2-docstring} better_robot.tasks.ik.OptimizerConfig.optimizer
```

````

````{py:attribute} max_iter
:canonical: better_robot.tasks.ik.OptimizerConfig.max_iter
:type: int
:value: >
   100

```{autodoc2-docstring} better_robot.tasks.ik.OptimizerConfig.max_iter
```

````

````{py:attribute} jacobian_strategy
:canonical: better_robot.tasks.ik.OptimizerConfig.jacobian_strategy
:type: better_robot.kinematics.jacobian_strategy.JacobianStrategy
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.ik.OptimizerConfig.jacobian_strategy
```

````

````{py:attribute} linear_solver
:canonical: better_robot.tasks.ik.OptimizerConfig.linear_solver
:type: typing.Literal[cholesky, lstsq, cg]
:value: >
   'cholesky'

```{autodoc2-docstring} better_robot.tasks.ik.OptimizerConfig.linear_solver
```

````

````{py:attribute} kernel
:canonical: better_robot.tasks.ik.OptimizerConfig.kernel
:type: typing.Literal[l2, huber, cauchy, tukey]
:value: >
   'l2'

```{autodoc2-docstring} better_robot.tasks.ik.OptimizerConfig.kernel
```

````

````{py:attribute} damping
:canonical: better_robot.tasks.ik.OptimizerConfig.damping
:type: typing.Literal[constant, adaptive, trust_region]
:value: >
   'adaptive'

```{autodoc2-docstring} better_robot.tasks.ik.OptimizerConfig.damping
```

````

````{py:attribute} tol
:canonical: better_robot.tasks.ik.OptimizerConfig.tol
:type: float
:value: >
   1e-06

```{autodoc2-docstring} better_robot.tasks.ik.OptimizerConfig.tol
```

````

````{py:attribute} refine_disabled_items
:canonical: better_robot.tasks.ik.OptimizerConfig.refine_disabled_items
:type: tuple[str, ...]
:value: >
   ()

```{autodoc2-docstring} better_robot.tasks.ik.OptimizerConfig.refine_disabled_items
```

````

`````

`````{py:class} IKResult
:canonical: better_robot.tasks.ik.IKResult

```{autodoc2-docstring} better_robot.tasks.ik.IKResult
```

````{py:attribute} q
:canonical: better_robot.tasks.ik.IKResult.q
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.ik.IKResult.q
```

````

````{py:attribute} residual
:canonical: better_robot.tasks.ik.IKResult.residual
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.ik.IKResult.residual
```

````

````{py:attribute} iters
:canonical: better_robot.tasks.ik.IKResult.iters
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.ik.IKResult.iters
```

````

````{py:attribute} converged
:canonical: better_robot.tasks.ik.IKResult.converged
:type: bool
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.ik.IKResult.converged
```

````

````{py:attribute} model
:canonical: better_robot.tasks.ik.IKResult.model
:type: better_robot.data_model.model.Model
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.ik.IKResult.model
```

````

````{py:method} fk() -> better_robot.data_model.data.Data
:canonical: better_robot.tasks.ik.IKResult.fk

```{autodoc2-docstring} better_robot.tasks.ik.IKResult.fk
```

````

````{py:method} frame_pose(name: str) -> torch.Tensor
:canonical: better_robot.tasks.ik.IKResult.frame_pose

```{autodoc2-docstring} better_robot.tasks.ik.IKResult.frame_pose
```

````

````{py:method} q_only() -> torch.Tensor
:canonical: better_robot.tasks.ik.IKResult.q_only

```{autodoc2-docstring} better_robot.tasks.ik.IKResult.q_only
```

````

`````

````{py:function} solve_ik(model: better_robot.data_model.model.Model, targets: dict[str, torch.Tensor], *, initial_q: torch.Tensor | None = None, cost_cfg: better_robot.tasks.ik.IKCostConfig | None = None, optimizer_cfg: better_robot.tasks.ik.OptimizerConfig | None = None, robot_collision: RobotCollision | None = None) -> better_robot.tasks.ik.IKResult
:canonical: better_robot.tasks.ik.solve_ik

```{autodoc2-docstring} better_robot.tasks.ik.solve_ik
```
````
