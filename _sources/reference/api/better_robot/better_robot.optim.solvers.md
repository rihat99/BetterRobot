# {py:mod}`better_robot.optim.solvers`

```{py:module} better_robot.optim.solvers
```

```{autodoc2-docstring} better_robot.optim.solvers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearSolveStatus <better_robot.optim.solvers.LinearSolveStatus>`
  -
* - {py:obj}`LinearSolveResult <better_robot.optim.solvers.LinearSolveResult>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveResult
    :summary:
    ```
* - {py:obj}`LinearSolver <better_robot.optim.solvers.LinearSolver>`
  -
* - {py:obj}`InformativeLinearSolver <better_robot.optim.solvers.InformativeLinearSolver>`
  -
* - {py:obj}`Cholesky <better_robot.optim.solvers.Cholesky>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.Cholesky
    :summary:
    ```
* - {py:obj}`BandedCholesky <better_robot.optim.solvers.BandedCholesky>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.BandedCholesky
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearSystem <better_robot.optim.solvers.LinearSystem>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.LinearSystem
    :summary:
    ```
````

### API

````{py:data} LinearSystem
:canonical: better_robot.optim.solvers.LinearSystem
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.LinearSystem
```

````

`````{py:class} LinearSolveStatus()
:canonical: better_robot.optim.solvers.LinearSolveStatus

Bases: {py:obj}`enum.IntEnum`

````{py:attribute} SUCCESS
:canonical: better_robot.optim.solvers.LinearSolveStatus.SUCCESS
:value: >
   0

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveStatus.SUCCESS
```

````

````{py:attribute} MAX_ITER
:canonical: better_robot.optim.solvers.LinearSolveStatus.MAX_ITER
:value: >
   1

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveStatus.MAX_ITER
```

````

````{py:attribute} NONFINITE
:canonical: better_robot.optim.solvers.LinearSolveStatus.NONFINITE
:value: >
   2

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveStatus.NONFINITE
```

````

````{py:attribute} BREAKDOWN
:canonical: better_robot.optim.solvers.LinearSolveStatus.BREAKDOWN
:value: >
   3

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveStatus.BREAKDOWN
```

````

````{py:attribute} NOT_SPD
:canonical: better_robot.optim.solvers.LinearSolveStatus.NOT_SPD
:value: >
   4

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveStatus.NOT_SPD
```

````

`````

`````{py:class} LinearSolveResult
:canonical: better_robot.optim.solvers.LinearSolveResult

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveResult
```

````{py:attribute} solution
:canonical: better_robot.optim.solvers.LinearSolveResult.solution
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveResult.solution
```

````

````{py:attribute} converged
:canonical: better_robot.optim.solvers.LinearSolveResult.converged
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveResult.converged
```

````

````{py:attribute} finite
:canonical: better_robot.optim.solvers.LinearSolveResult.finite
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveResult.finite
```

````

````{py:attribute} ok
:canonical: better_robot.optim.solvers.LinearSolveResult.ok
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveResult.ok
```

````

````{py:attribute} iterations
:canonical: better_robot.optim.solvers.LinearSolveResult.iterations
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveResult.iterations
```

````

````{py:attribute} residual_norm
:canonical: better_robot.optim.solvers.LinearSolveResult.residual_norm
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveResult.residual_norm
```

````

````{py:attribute} relative_residual
:canonical: better_robot.optim.solvers.LinearSolveResult.relative_residual
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveResult.relative_residual
```

````

````{py:attribute} status
:canonical: better_robot.optim.solvers.LinearSolveResult.status
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolveResult.status
```

````

`````

`````{py:class} LinearSolver
:canonical: better_robot.optim.solvers.LinearSolver

Bases: {py:obj}`typing.Protocol`

````{py:method} solve(A: better_robot.optim.solvers.LinearSystem, b: torch.Tensor, ridge: torch.Tensor | float | None = None) -> torch.Tensor
:canonical: better_robot.optim.solvers.LinearSolver.solve

```{autodoc2-docstring} better_robot.optim.solvers.LinearSolver.solve
```

````

`````

`````{py:class} InformativeLinearSolver
:canonical: better_robot.optim.solvers.InformativeLinearSolver

Bases: {py:obj}`typing.Protocol`

````{py:method} solve_with_info(A: better_robot.optim.solvers.LinearSystem, b: torch.Tensor, ridge: torch.Tensor | float | None = None, *, initial: torch.Tensor | None = None) -> better_robot.optim.solvers.LinearSolveResult
:canonical: better_robot.optim.solvers.InformativeLinearSolver.solve_with_info

```{autodoc2-docstring} better_robot.optim.solvers.InformativeLinearSolver.solve_with_info
```

````

`````

`````{py:class} Cholesky
:canonical: better_robot.optim.solvers.Cholesky

```{autodoc2-docstring} better_robot.optim.solvers.Cholesky
```

````{py:attribute} supported_systems
:canonical: better_robot.optim.solvers.Cholesky.supported_systems
:value: >
   'frozenset(...)'

```{autodoc2-docstring} better_robot.optim.solvers.Cholesky.supported_systems
```

````

````{py:method} solve(A: torch.Tensor, b: torch.Tensor, ridge: torch.Tensor | float | None = None) -> torch.Tensor
:canonical: better_robot.optim.solvers.Cholesky.solve

```{autodoc2-docstring} better_robot.optim.solvers.Cholesky.solve
```

````

````{py:method} solve_with_info(A: torch.Tensor, b: torch.Tensor, ridge: torch.Tensor | float | None = None, *, initial: torch.Tensor | None = None) -> better_robot.optim.solvers.LinearSolveResult
:canonical: better_robot.optim.solvers.Cholesky.solve_with_info

```{autodoc2-docstring} better_robot.optim.solvers.Cholesky.solve_with_info
```

````

`````

`````{py:class} BandedCholesky
:canonical: better_robot.optim.solvers.BandedCholesky

```{autodoc2-docstring} better_robot.optim.solvers.BandedCholesky
```

````{py:attribute} supported_systems
:canonical: better_robot.optim.solvers.BandedCholesky.supported_systems
:value: >
   'frozenset(...)'

```{autodoc2-docstring} better_robot.optim.solvers.BandedCholesky.supported_systems
```

````

````{py:attribute} supports_initial
:canonical: better_robot.optim.solvers.BandedCholesky.supports_initial
:value: >
   False

```{autodoc2-docstring} better_robot.optim.solvers.BandedCholesky.supports_initial
```

````

````{py:method} solve(A: better_robot.optim.temporal.BlockBandedMatrix, b: torch.Tensor, ridge: torch.Tensor | float | None = None) -> torch.Tensor
:canonical: better_robot.optim.solvers.BandedCholesky.solve

```{autodoc2-docstring} better_robot.optim.solvers.BandedCholesky.solve
```

````

````{py:method} solve_with_info(A: better_robot.optim.temporal.BlockBandedMatrix, b: torch.Tensor, ridge: torch.Tensor | float | None = None, *, initial: torch.Tensor | None = None) -> better_robot.optim.solvers.LinearSolveResult
:canonical: better_robot.optim.solvers.BandedCholesky.solve_with_info

```{autodoc2-docstring} better_robot.optim.solvers.BandedCholesky.solve_with_info
```

````

`````
