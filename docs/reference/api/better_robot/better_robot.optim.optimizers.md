# {py:mod}`better_robot.optim.optimizers`

```{py:module} better_robot.optim.optimizers
```

```{autodoc2-docstring} better_robot.optim.optimizers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OptimizerStatus <better_robot.optim.optimizers.OptimizerStatus>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerStatus
    :summary:
    ```
* - {py:obj}`OptimizerInfo <better_robot.optim.optimizers.OptimizerInfo>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerInfo
    :summary:
    ```
* - {py:obj}`Optimizer <better_robot.optim.optimizers.Optimizer>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.Optimizer
    :summary:
    ```
* - {py:obj}`TorchOptimizer <better_robot.optim.optimizers.TorchOptimizer>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.TorchOptimizer
    :summary:
    ```
````

### API

`````{py:class} OptimizerStatus()
:canonical: better_robot.optim.optimizers.OptimizerStatus

Bases: {py:obj}`enum.IntEnum`

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerStatus
```

````{py:attribute} RUNNING
:canonical: better_robot.optim.optimizers.OptimizerStatus.RUNNING
:value: >
   0

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerStatus.RUNNING
```

````

````{py:attribute} CONVERGED
:canonical: better_robot.optim.optimizers.OptimizerStatus.CONVERGED
:value: >
   1

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerStatus.CONVERGED
```

````

````{py:attribute} STALLED_AT_BOUNDS
:canonical: better_robot.optim.optimizers.OptimizerStatus.STALLED_AT_BOUNDS
:value: >
   2

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerStatus.STALLED_AT_BOUNDS
```

````

````{py:attribute} MAXITER
:canonical: better_robot.optim.optimizers.OptimizerStatus.MAXITER
:value: >
   3

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerStatus.MAXITER
```

````

````{py:attribute} FAILED
:canonical: better_robot.optim.optimizers.OptimizerStatus.FAILED
:value: >
   4

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerStatus.FAILED
```

````

`````

`````{py:class} OptimizerInfo
:canonical: better_robot.optim.optimizers.OptimizerInfo

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerInfo
```

````{py:attribute} status
:canonical: better_robot.optim.optimizers.OptimizerInfo.status
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerInfo.status
```

````

````{py:attribute} iterations
:canonical: better_robot.optim.optimizers.OptimizerInfo.iterations
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerInfo.iterations
```

````

````{py:attribute} cost
:canonical: better_robot.optim.optimizers.OptimizerInfo.cost
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerInfo.cost
```

````

````{py:property} converged
:canonical: better_robot.optim.optimizers.OptimizerInfo.converged
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim.optimizers.OptimizerInfo.converged
```

````

`````

`````{py:class} Optimizer(problem: better_robot.optim.problem.Problem, *, max_iterations: int = 50, tolerance: float = 1e-08)
:canonical: better_robot.optim.optimizers.Optimizer

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} better_robot.optim.optimizers.Optimizer
```

````{py:method} step() -> better_robot.optim.optimizers.OptimizerInfo
:canonical: better_robot.optim.optimizers.Optimizer.step
:abstractmethod:

```{autodoc2-docstring} better_robot.optim.optimizers.Optimizer.step
```

````

````{py:method} optimize(*, verbose: bool = False) -> better_robot.optim.optimizers.OptimizerInfo
:canonical: better_robot.optim.optimizers.Optimizer.optimize

```{autodoc2-docstring} better_robot.optim.optimizers.Optimizer.optimize
```

````

````{py:method} reset() -> None
:canonical: better_robot.optim.optimizers.Optimizer.reset
:abstractmethod:

```{autodoc2-docstring} better_robot.optim.optimizers.Optimizer.reset
```

````

`````

`````{py:class} TorchOptimizer(problem: better_robot.optim.problem.Problem, optimizer_cls: type[torch.optim.Optimizer] | better_robot.optim.optimizers._TorchOptimizerFactory, *, max_iterations: int = 100, tolerance: float = 0.0, **optimizer_kwargs: typing.Any)
:canonical: better_robot.optim.optimizers.TorchOptimizer

Bases: {py:obj}`better_robot.optim.optimizers.Optimizer`

```{autodoc2-docstring} better_robot.optim.optimizers.TorchOptimizer
```

````{py:method} reset() -> None
:canonical: better_robot.optim.optimizers.TorchOptimizer.reset

````

````{py:method} step() -> better_robot.optim.optimizers.OptimizerInfo
:canonical: better_robot.optim.optimizers.TorchOptimizer.step

````

````{py:method} optimize(*, verbose: bool = False, differentiate: str | None = None) -> better_robot.optim.optimizers.OptimizerInfo
:canonical: better_robot.optim.optimizers.TorchOptimizer.optimize

````

`````
