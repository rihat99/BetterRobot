# {py:mod}`better_robot.optim.solvers.normal_cg`

```{py:module} better_robot.optim.solvers.normal_cg
```

```{autodoc2-docstring} better_robot.optim.solvers.normal_cg
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NormalCG <better_robot.optim.solvers.normal_cg.NormalCG>`
  - ```{autodoc2-docstring} better_robot.optim.solvers.normal_cg.NormalCG
    :summary:
    ```
````

### API

`````{py:class} NormalCG
:canonical: better_robot.optim.solvers.normal_cg.NormalCG

```{autodoc2-docstring} better_robot.optim.solvers.normal_cg.NormalCG
```

````{py:attribute} max_iter
:canonical: better_robot.optim.solvers.normal_cg.NormalCG.max_iter
:type: int
:value: >
   100

```{autodoc2-docstring} better_robot.optim.solvers.normal_cg.NormalCG.max_iter
```

````

````{py:attribute} rtol
:canonical: better_robot.optim.solvers.normal_cg.NormalCG.rtol
:type: float
:value: >
   1e-05

```{autodoc2-docstring} better_robot.optim.solvers.normal_cg.NormalCG.rtol
```

````

````{py:attribute} atol
:canonical: better_robot.optim.solvers.normal_cg.NormalCG.atol
:type: float
:value: >
   0.0

```{autodoc2-docstring} better_robot.optim.solvers.normal_cg.NormalCG.atol
```

````

````{py:attribute} supported_systems
:canonical: better_robot.optim.solvers.normal_cg.NormalCG.supported_systems
:type: typing.ClassVar[frozenset[str]]
:value: >
   'frozenset(...)'

```{autodoc2-docstring} better_robot.optim.solvers.normal_cg.NormalCG.supported_systems
```

````

````{py:attribute} supports_initial
:canonical: better_robot.optim.solvers.normal_cg.NormalCG.supports_initial
:type: typing.ClassVar[bool]
:value: >
   True

```{autodoc2-docstring} better_robot.optim.solvers.normal_cg.NormalCG.supports_initial
```

````

````{py:method} solve(A: better_robot.optim.structure.NormalOperator, b: torch.Tensor, ridge: torch.Tensor | float | None = None) -> torch.Tensor
:canonical: better_robot.optim.solvers.normal_cg.NormalCG.solve

```{autodoc2-docstring} better_robot.optim.solvers.normal_cg.NormalCG.solve
```

````

````{py:method} solve_with_info(A: better_robot.optim.structure.NormalOperator, b: torch.Tensor, ridge: torch.Tensor | float | None = None, *, initial: torch.Tensor | None = None) -> better_robot.optim.solvers.base.LinearSolveResult
:canonical: better_robot.optim.solvers.normal_cg.NormalCG.solve_with_info

```{autodoc2-docstring} better_robot.optim.solvers.normal_cg.NormalCG.solve_with_info
```

````

`````
