# {py:mod}`better_robot.optim.solvers.base`

```{py:module} better_robot.optim.solvers.base
```

```{autodoc2-docstring} better_robot.optim.solvers.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearSolveStatus <better_robot.optim.solvers.base.LinearSolveStatus>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveStatus
    :summary:
    ```
* - {py:obj}`LinearSolveResult <better_robot.optim.solvers.base.LinearSolveResult>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveResult
    :summary:
    ```
* - {py:obj}`LinearSolver <better_robot.optim.solvers.base.LinearSolver>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolver
    :summary:
    ```
* - {py:obj}`InformativeLinearSolver <better_robot.optim.solvers.base.InformativeLinearSolver>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.base.InformativeLinearSolver
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearSystem <better_robot.optim.solvers.base.LinearSystem>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSystem
    :summary:
    ```
````

### API

````{py:data} LinearSystem
:canonical: better_robot.optim.solvers.base.LinearSystem
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSystem
```

````

`````{py:class} LinearSolveStatus()
:canonical: better_robot.optim.solvers.base.LinearSolveStatus

Bases: {py:obj}`enum.IntEnum`

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveStatus
```

````{py:attribute} SUCCESS
:canonical: better_robot.optim.solvers.base.LinearSolveStatus.SUCCESS
:value: >
   0

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveStatus.SUCCESS
```

````

````{py:attribute} MAX_ITER
:canonical: better_robot.optim.solvers.base.LinearSolveStatus.MAX_ITER
:value: >
   1

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveStatus.MAX_ITER
```

````

````{py:attribute} NONFINITE
:canonical: better_robot.optim.solvers.base.LinearSolveStatus.NONFINITE
:value: >
   2

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveStatus.NONFINITE
```

````

````{py:attribute} BREAKDOWN
:canonical: better_robot.optim.solvers.base.LinearSolveStatus.BREAKDOWN
:value: >
   3

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveStatus.BREAKDOWN
```

````

````{py:attribute} NOT_SPD
:canonical: better_robot.optim.solvers.base.LinearSolveStatus.NOT_SPD
:value: >
   4

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveStatus.NOT_SPD
```

````

`````

`````{py:class} LinearSolveResult
:canonical: better_robot.optim.solvers.base.LinearSolveResult

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveResult
```

````{py:attribute} solution
:canonical: better_robot.optim.solvers.base.LinearSolveResult.solution
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveResult.solution
```

````

````{py:attribute} converged
:canonical: better_robot.optim.solvers.base.LinearSolveResult.converged
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveResult.converged
```

````

````{py:attribute} finite
:canonical: better_robot.optim.solvers.base.LinearSolveResult.finite
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveResult.finite
```

````

````{py:attribute} ok
:canonical: better_robot.optim.solvers.base.LinearSolveResult.ok
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveResult.ok
```

````

````{py:attribute} iterations
:canonical: better_robot.optim.solvers.base.LinearSolveResult.iterations
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveResult.iterations
```

````

````{py:attribute} residual_norm
:canonical: better_robot.optim.solvers.base.LinearSolveResult.residual_norm
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveResult.residual_norm
```

````

````{py:attribute} relative_residual
:canonical: better_robot.optim.solvers.base.LinearSolveResult.relative_residual
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveResult.relative_residual
```

````

````{py:attribute} status
:canonical: better_robot.optim.solvers.base.LinearSolveResult.status
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolveResult.status
```

````

`````

`````{py:class} LinearSolver
:canonical: better_robot.optim.solvers.base.LinearSolver

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolver
```

````{py:method} solve(A: better_robot.optim.solvers.base.LinearSystem, b: torch.Tensor, ridge: torch.Tensor | float | None = None) -> torch.Tensor
:canonical: better_robot.optim.solvers.base.LinearSolver.solve

```{autodoc2-docstring} better_robot.optim.solvers.base.LinearSolver.solve
```

````

`````

`````{py:class} InformativeLinearSolver
:canonical: better_robot.optim.solvers.base.InformativeLinearSolver

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.optim.solvers.base.InformativeLinearSolver
```

````{py:method} solve_with_info(A: better_robot.optim.solvers.base.LinearSystem, b: torch.Tensor, ridge: torch.Tensor | float | None = None, *, initial: torch.Tensor | None = None) -> better_robot.optim.solvers.base.LinearSolveResult
:canonical: better_robot.optim.solvers.base.InformativeLinearSolver.solve_with_info

```{autodoc2-docstring} better_robot.optim.solvers.base.InformativeLinearSolver.solve_with_info
```

````

`````
