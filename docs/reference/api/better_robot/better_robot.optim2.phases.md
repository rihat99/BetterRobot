# {py:mod}`better_robot.optim2.phases`

```{py:module} better_robot.optim2.phases
```

```{autodoc2-docstring} better_robot.optim2.phases
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Phase <better_robot.optim2.phases.Phase>`
  - ```{autodoc2-docstring} better_robot.optim2.phases.Phase
    :summary:
    ```
* - {py:obj}`Result <better_robot.optim2.phases.Result>`
  - ```{autodoc2-docstring} better_robot.optim2.phases.Result
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run <better_robot.optim2.phases.run>`
  - ```{autodoc2-docstring} better_robot.optim2.phases.run
    :summary:
    ```
* - {py:obj}`solve <better_robot.optim2.phases.solve>`
  - ```{autodoc2-docstring} better_robot.optim2.phases.solve
    :summary:
    ```
* - {py:obj}`solve_layer <better_robot.optim2.phases.solve_layer>`
  - ```{autodoc2-docstring} better_robot.optim2.phases.solve_layer
    :summary:
    ```
````

### API

`````{py:class} Phase
:canonical: better_robot.optim2.phases.Phase

```{autodoc2-docstring} better_robot.optim2.phases.Phase
```

````{py:attribute} solver
:canonical: better_robot.optim2.phases.Phase.solver
:type: better_robot.optim2.solvers.Solver
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.phases.Phase.solver
```

````

````{py:attribute} iters
:canonical: better_robot.optim2.phases.Phase.iters
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.phases.Phase.iters
```

````

````{py:attribute} name
:canonical: better_robot.optim2.phases.Phase.name
:type: str
:value: >
   'phase'

```{autodoc2-docstring} better_robot.optim2.phases.Phase.name
```

````

````{py:attribute} weights
:canonical: better_robot.optim2.phases.Phase.weights
:type: collections.abc.Mapping[str, float] | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.phases.Phase.weights
```

````

````{py:attribute} free
:canonical: better_robot.optim2.phases.Phase.free
:type: collections.abc.Mapping[str, torch.Tensor] | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.phases.Phase.free
```

````

````{py:attribute} on_start
:canonical: better_robot.optim2.phases.Phase.on_start
:type: collections.abc.Callable[[better_robot.optim2.problem.Problem, dict[str, torch.Tensor]], dict[str, torch.Tensor] | None] | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.phases.Phase.on_start
```

````

````{py:attribute} on_step
:canonical: better_robot.optim2.phases.Phase.on_step
:type: collections.abc.Callable[[better_robot.optim2.problem.Problem, better_robot.optim2.solvers.State], None] | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.phases.Phase.on_step
```

````

````{py:attribute} stop_every
:canonical: better_robot.optim2.phases.Phase.stop_every
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.optim2.phases.Phase.stop_every
```

````

`````

`````{py:class} Result
:canonical: better_robot.optim2.phases.Result

```{autodoc2-docstring} better_robot.optim2.phases.Result
```

````{py:attribute} x
:canonical: better_robot.optim2.phases.Result.x
:type: dict[str, torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.phases.Result.x
```

````

````{py:attribute} loss
:canonical: better_robot.optim2.phases.Result.loss
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.phases.Result.loss
```

````

````{py:attribute} done
:canonical: better_robot.optim2.phases.Result.done
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.phases.Result.done
```

````

````{py:attribute} iters
:canonical: better_robot.optim2.phases.Result.iters
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.phases.Result.iters
```

````

````{py:attribute} history
:canonical: better_robot.optim2.phases.Result.history
:type: dict[str, torch.Tensor]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.optim2.phases.Result.history
```

````

`````

````{py:function} run(problem: better_robot.optim2.problem.Problem, phases: collections.abc.Sequence[better_robot.optim2.phases.Phase], x: collections.abc.Mapping[str, torch.Tensor] | None = None) -> better_robot.optim2.phases.Result
:canonical: better_robot.optim2.phases.run

```{autodoc2-docstring} better_robot.optim2.phases.run
```
````

````{py:function} solve(problem: better_robot.optim2.problem.Problem, solver: better_robot.optim2.solvers.Solver, *, iters: int = 50, x: collections.abc.Mapping[str, torch.Tensor] | None = None) -> better_robot.optim2.phases.Result
:canonical: better_robot.optim2.phases.solve

```{autodoc2-docstring} better_robot.optim2.phases.solve
```
````

````{py:function} solve_layer(problem: better_robot.optim2.problem.Problem, solver: better_robot.optim2.solvers.Solver, *, iters: int = 50, x: collections.abc.Mapping[str, torch.Tensor] | None = None, damping: float = 1e-06) -> better_robot.optim2.phases.Result
:canonical: better_robot.optim2.phases.solve_layer

```{autodoc2-docstring} better_robot.optim2.phases.solve_layer
```
````
