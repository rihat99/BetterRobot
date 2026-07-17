# {py:mod}`better_robot.optim.solvers.cholesky`

```{py:module} better_robot.optim.solvers.cholesky
```

```{autodoc2-docstring} better_robot.optim.solvers.cholesky
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Cholesky <better_robot.optim.solvers.cholesky.Cholesky>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.cholesky.Cholesky
    :summary:
    ```
````

### API

`````{py:class} Cholesky
:canonical: better_robot.optim.solvers.cholesky.Cholesky

```{autodoc2-docstring} better_robot.optim.solvers.cholesky.Cholesky
```

````{py:attribute} supported_systems
:canonical: better_robot.optim.solvers.cholesky.Cholesky.supported_systems
:value: >
   'frozenset(...)'

```{autodoc2-docstring} better_robot.optim.solvers.cholesky.Cholesky.supported_systems
```

````

````{py:method} solve(A: torch.Tensor, b: torch.Tensor, ridge: torch.Tensor | float | None = None) -> torch.Tensor
:canonical: better_robot.optim.solvers.cholesky.Cholesky.solve

```{autodoc2-docstring} better_robot.optim.solvers.cholesky.Cholesky.solve
```

````

````{py:method} solve_with_info(A: torch.Tensor, b: torch.Tensor, ridge: torch.Tensor | float | None = None, *, initial: torch.Tensor | None = None) -> better_robot.optim.solvers.base.LinearSolveResult
:canonical: better_robot.optim.solvers.cholesky.Cholesky.solve_with_info

```{autodoc2-docstring} better_robot.optim.solvers.cholesky.Cholesky.solve_with_info
```

````

`````
