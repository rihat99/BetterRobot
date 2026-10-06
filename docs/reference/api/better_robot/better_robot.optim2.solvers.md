# {py:mod}`better_robot.optim2.solvers`

```{py:module} better_robot.optim2.solvers
```

```{autodoc2-docstring} better_robot.optim2.solvers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`State <better_robot.optim2.solvers.State>`
  - ```{autodoc2-docstring} better_robot.optim2.solvers.State
    :summary:
    ```
* - {py:obj}`Solver <better_robot.optim2.solvers.Solver>`
  - ```{autodoc2-docstring} better_robot.optim2.solvers.Solver
    :summary:
    ```
* - {py:obj}`LevenbergMarquardt <better_robot.optim2.solvers.LevenbergMarquardt>`
  - ```{autodoc2-docstring} better_robot.optim2.solvers.LevenbergMarquardt
    :summary:
    ```
* - {py:obj}`GaussNewton <better_robot.optim2.solvers.GaussNewton>`
  - ```{autodoc2-docstring} better_robot.optim2.solvers.GaussNewton
    :summary:
    ```
* - {py:obj}`LBFGS <better_robot.optim2.solvers.LBFGS>`
  - ```{autodoc2-docstring} better_robot.optim2.solvers.LBFGS
    :summary:
    ```
* - {py:obj}`TorchOptim <better_robot.optim2.solvers.TorchOptim>`
  - ```{autodoc2-docstring} better_robot.optim2.solvers.TorchOptim
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`select <better_robot.optim2.solvers.select>`
  - ```{autodoc2-docstring} better_robot.optim2.solvers.select
    :summary:
    ```
````

### API

`````{py:class} State
:canonical: better_robot.optim2.solvers.State

```{autodoc2-docstring} better_robot.optim2.solvers.State
```

````{py:attribute} x
:canonical: better_robot.optim2.solvers.State.x
:type: dict[str, torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.solvers.State.x
```

````

````{py:attribute} loss
:canonical: better_robot.optim2.solvers.State.loss
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.solvers.State.loss
```

````

````{py:attribute} done
:canonical: better_robot.optim2.solvers.State.done
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.solvers.State.done
```

````

````{py:attribute} it
:canonical: better_robot.optim2.solvers.State.it
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.optim2.solvers.State.it
```

````

````{py:attribute} extra
:canonical: better_robot.optim2.solvers.State.extra
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.optim2.solvers.State.extra
```

````

`````

`````{py:class} Solver
:canonical: better_robot.optim2.solvers.Solver

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.optim2.solvers.Solver
```

````{py:method} init(problem: better_robot.optim2.problem.Problem, x: dict[str, torch.Tensor]) -> better_robot.optim2.solvers.State
:canonical: better_robot.optim2.solvers.Solver.init

```{autodoc2-docstring} better_robot.optim2.solvers.Solver.init
```

````

````{py:method} step(problem: better_robot.optim2.problem.Problem, state: better_robot.optim2.solvers.State) -> better_robot.optim2.solvers.State
:canonical: better_robot.optim2.solvers.Solver.step

```{autodoc2-docstring} better_robot.optim2.solvers.Solver.step
```

````

`````

````{py:function} select(mask: torch.Tensor, new: dict[str, torch.Tensor], old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]
:canonical: better_robot.optim2.solvers.select

```{autodoc2-docstring} better_robot.optim2.solvers.select
```
````

`````{py:class} LevenbergMarquardt(*, damping: float = 0.001, gtol: float | None = None, ftol: float | None = None, xtol: float | None = None)
:canonical: better_robot.optim2.solvers.LevenbergMarquardt

```{autodoc2-docstring} better_robot.optim2.solvers.LevenbergMarquardt
```

````{py:method} init(problem: better_robot.optim2.problem.Problem, x: dict[str, torch.Tensor]) -> better_robot.optim2.solvers.State
:canonical: better_robot.optim2.solvers.LevenbergMarquardt.init

```{autodoc2-docstring} better_robot.optim2.solvers.LevenbergMarquardt.init
```

````

````{py:method} step(problem: better_robot.optim2.problem.Problem, state: better_robot.optim2.solvers.State) -> better_robot.optim2.solvers.State
:canonical: better_robot.optim2.solvers.LevenbergMarquardt.step

```{autodoc2-docstring} better_robot.optim2.solvers.LevenbergMarquardt.step
```

````

`````

`````{py:class} GaussNewton(*, damping: float = 0.0, gtol: float | None = None, xtol: float | None = None)
:canonical: better_robot.optim2.solvers.GaussNewton

```{autodoc2-docstring} better_robot.optim2.solvers.GaussNewton
```

````{py:method} init(problem: better_robot.optim2.problem.Problem, x: dict[str, torch.Tensor]) -> better_robot.optim2.solvers.State
:canonical: better_robot.optim2.solvers.GaussNewton.init

```{autodoc2-docstring} better_robot.optim2.solvers.GaussNewton.init
```

````

````{py:method} step(problem: better_robot.optim2.problem.Problem, state: better_robot.optim2.solvers.State) -> better_robot.optim2.solvers.State
:canonical: better_robot.optim2.solvers.GaussNewton.step

```{autodoc2-docstring} better_robot.optim2.solvers.GaussNewton.step
```

````

`````

`````{py:class} LBFGS(*, history: int = 10, line_search: int = 8, armijo: float = 0.0001, gtol: float | None = None)
:canonical: better_robot.optim2.solvers.LBFGS

```{autodoc2-docstring} better_robot.optim2.solvers.LBFGS
```

````{py:method} init(problem: better_robot.optim2.problem.Problem, x: dict[str, torch.Tensor]) -> better_robot.optim2.solvers.State
:canonical: better_robot.optim2.solvers.LBFGS.init

```{autodoc2-docstring} better_robot.optim2.solvers.LBFGS.init
```

````

````{py:method} step(problem: better_robot.optim2.problem.Problem, state: better_robot.optim2.solvers.State) -> better_robot.optim2.solvers.State
:canonical: better_robot.optim2.solvers.LBFGS.step

```{autodoc2-docstring} better_robot.optim2.solvers.LBFGS.step
```

````

`````

`````{py:class} TorchOptim(optimizer_cls: type[torch.optim.Optimizer], *, scheduler: collections.abc.Callable[[torch.optim.Optimizer], typing.Any] | None = None, **kwargs: typing.Any)
:canonical: better_robot.optim2.solvers.TorchOptim

```{autodoc2-docstring} better_robot.optim2.solvers.TorchOptim
```

````{py:method} init(problem: better_robot.optim2.problem.Problem, x: dict[str, torch.Tensor]) -> better_robot.optim2.solvers.State
:canonical: better_robot.optim2.solvers.TorchOptim.init

```{autodoc2-docstring} better_robot.optim2.solvers.TorchOptim.init
```

````

````{py:method} step(problem: better_robot.optim2.problem.Problem, state: better_robot.optim2.solvers.State) -> better_robot.optim2.solvers.State
:canonical: better_robot.optim2.solvers.TorchOptim.step

```{autodoc2-docstring} better_robot.optim2.solvers.TorchOptim.step
```

````

`````
