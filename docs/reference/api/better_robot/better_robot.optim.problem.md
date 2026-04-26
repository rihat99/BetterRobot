# {py:mod}`better_robot.optim.problem`

```{py:module} better_robot.optim.problem
```

```{autodoc2-docstring} better_robot.optim.problem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LeastSquaresProblem <better_robot.optim.problem.LeastSquaresProblem>`
  - ```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem
    :summary:
    ```
````

### API

`````{py:class} LeastSquaresProblem
:canonical: better_robot.optim.problem.LeastSquaresProblem

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem
```

````{py:attribute} cost_stack
:canonical: better_robot.optim.problem.LeastSquaresProblem.cost_stack
:type: better_robot.costs.stack.CostStack
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.cost_stack
```

````

````{py:attribute} state_factory
:canonical: better_robot.optim.problem.LeastSquaresProblem.state_factory
:type: typing.Callable[[torch.Tensor], better_robot.residuals.base.ResidualState]
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.state_factory
```

````

````{py:attribute} x0
:canonical: better_robot.optim.problem.LeastSquaresProblem.x0
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.x0
```

````

````{py:attribute} lower
:canonical: better_robot.optim.problem.LeastSquaresProblem.lower
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.lower
```

````

````{py:attribute} upper
:canonical: better_robot.optim.problem.LeastSquaresProblem.upper
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.upper
```

````

````{py:attribute} jacobian_strategy
:canonical: better_robot.optim.problem.LeastSquaresProblem.jacobian_strategy
:type: better_robot.kinematics.jacobian_strategy.JacobianStrategy
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.jacobian_strategy
```

````

````{py:attribute} nv
:canonical: better_robot.optim.problem.LeastSquaresProblem.nv
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.nv
```

````

````{py:attribute} retract
:canonical: better_robot.optim.problem.LeastSquaresProblem.retract
:type: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.retract
```

````

````{py:method} residual(x: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.problem.LeastSquaresProblem.residual

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.residual
```

````

````{py:method} jacobian(x: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.problem.LeastSquaresProblem.jacobian

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.jacobian
```

````

````{py:method} step(x: torch.Tensor, delta_v: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.problem.LeastSquaresProblem.step

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.step
```

````

````{py:method} gradient(x: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.problem.LeastSquaresProblem.gradient

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.gradient
```

````

````{py:method} jacobian_blocks(x: torch.Tensor) -> dict[str, torch.Tensor]
:canonical: better_robot.optim.problem.LeastSquaresProblem.jacobian_blocks

```{autodoc2-docstring} better_robot.optim.problem.LeastSquaresProblem.jacobian_blocks
```

````

`````
