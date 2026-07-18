# {py:mod}`better_robot.optim.solvers.banded_cholesky`

```{py:module} better_robot.optim.solvers.banded_cholesky
```

```{autodoc2-docstring} better_robot.optim.solvers.banded_cholesky
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BandedCholesky <better_robot.optim.solvers.banded_cholesky.BandedCholesky>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.banded_cholesky.BandedCholesky
    :summary:
    ```
````

### API

`````{py:class} BandedCholesky
:canonical: better_robot.optim.solvers.banded_cholesky.BandedCholesky

```{autodoc2-docstring} better_robot.optim.solvers.banded_cholesky.BandedCholesky
```

````{py:attribute} supported_systems
:canonical: better_robot.optim.solvers.banded_cholesky.BandedCholesky.supported_systems
:value: >
   'frozenset(...)'

```{autodoc2-docstring} better_robot.optim.solvers.banded_cholesky.BandedCholesky.supported_systems
```

````

````{py:attribute} supports_initial
:canonical: better_robot.optim.solvers.banded_cholesky.BandedCholesky.supports_initial
:value: >
   False

```{autodoc2-docstring} better_robot.optim.solvers.banded_cholesky.BandedCholesky.supports_initial
```

````

````{py:method} solve(A: better_robot.optim.structure.BlockBandedMatrix, b: torch.Tensor, ridge: torch.Tensor | float | None = None) -> torch.Tensor
:canonical: better_robot.optim.solvers.banded_cholesky.BandedCholesky.solve

```{autodoc2-docstring} better_robot.optim.solvers.banded_cholesky.BandedCholesky.solve
```

````

````{py:method} solve_with_info(A: better_robot.optim.structure.BlockBandedMatrix, b: torch.Tensor, ridge: torch.Tensor | float | None = None, *, initial: torch.Tensor | None = None) -> better_robot.optim.solvers.base.LinearSolveResult
:canonical: better_robot.optim.solvers.banded_cholesky.BandedCholesky.solve_with_info

```{autodoc2-docstring} better_robot.optim.solvers.banded_cholesky.BandedCholesky.solve_with_info
```

````

`````
