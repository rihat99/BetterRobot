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

* - {py:obj}`Problem <better_robot.optim.problem.Problem>`
  - ```{autodoc2-docstring} better_robot.optim.problem.Problem
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JacobianStrategy <better_robot.optim.problem.JacobianStrategy>`
  - ```{autodoc2-docstring} better_robot.optim.problem.JacobianStrategy
    :summary:
    ```
````

### API

````{py:data} JacobianStrategy
:canonical: better_robot.optim.problem.JacobianStrategy
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.JacobianStrategy
```

````

````{py:exception} AutodiffFallbackWarning()
:canonical: better_robot.optim.problem.AutodiffFallbackWarning

Bases: {py:obj}`RuntimeWarning`

```{autodoc2-docstring} better_robot.optim.problem.AutodiffFallbackWarning
```

````

`````{py:class} Problem(residuals: collections.abc.Sequence[better_robot.residuals.base.Residual] = ())
:canonical: better_robot.optim.problem.Problem

```{autodoc2-docstring} better_robot.optim.problem.Problem
```

````{py:property} frozen
:canonical: better_robot.optim.problem.Problem.frozen
:type: bool

```{autodoc2-docstring} better_robot.optim.problem.Problem.frozen
```

````

````{py:method} add_residual(item: better_robot.residuals.base.Residual) -> better_robot.residuals.base.Residual
:canonical: better_robot.optim.problem.Problem.add_residual

```{autodoc2-docstring} better_robot.optim.problem.Problem.add_residual
```

````

````{py:method} update(values: collections.abc.Mapping[str, torch.Tensor]) -> None
:canonical: better_robot.optim.problem.Problem.update

```{autodoc2-docstring} better_robot.optim.problem.Problem.update
```

````

````{py:method} error() -> torch.Tensor
:canonical: better_robot.optim.problem.Problem.error

```{autodoc2-docstring} better_robot.optim.problem.Problem.error
```

````

````{py:method} objective(values: collections.abc.Mapping[str, torch.Tensor] | None = None) -> torch.Tensor
:canonical: better_robot.optim.problem.Problem.objective

```{autodoc2-docstring} better_robot.optim.problem.Problem.objective
```

````

````{py:method} gradient(*, create_graph: bool = False) -> better_robot.optim.problem._TensorMap
:canonical: better_robot.optim.problem.Problem.gradient

```{autodoc2-docstring} better_robot.optim.problem.Problem.gradient
```

````

````{py:method} jacobian_blocks(*, strategy: better_robot.optim.problem.JacobianStrategy = 'auto', create_graph: bool = False, fd_eps: float = 0.0001) -> dict[tuple[str, str], torch.Tensor]
:canonical: better_robot.optim.problem.Problem.jacobian_blocks

```{autodoc2-docstring} better_robot.optim.problem.Problem.jacobian_blocks
```

````

````{py:method} dense_jacobian(*, strategy: better_robot.optim.problem.JacobianStrategy = 'auto', create_graph: bool = False) -> torch.Tensor
:canonical: better_robot.optim.problem.Problem.dense_jacobian

```{autodoc2-docstring} better_robot.optim.problem.Problem.dense_jacobian
```

````

````{py:method} retract(values: collections.abc.Mapping[str, torch.Tensor], steps: collections.abc.Mapping[str, torch.Tensor]) -> better_robot.optim.problem._TensorMap
:canonical: better_robot.optim.problem.Problem.retract

```{autodoc2-docstring} better_robot.optim.problem.Problem.retract
```

````

````{py:method} difference(x0: collections.abc.Mapping[str, torch.Tensor], x1: collections.abc.Mapping[str, torch.Tensor]) -> better_robot.optim.problem._TensorMap
:canonical: better_robot.optim.problem.Problem.difference

```{autodoc2-docstring} better_robot.optim.problem.Problem.difference
```

````

````{py:method} structured_normal(values: collections.abc.Mapping[str, torch.Tensor] | None = None, *, row_scale: torch.Tensor | None = None, residual: torch.Tensor | None = None, create_graph: bool = False)
:canonical: better_robot.optim.problem.Problem.structured_normal

```{autodoc2-docstring} better_robot.optim.problem.Problem.structured_normal
```

````

`````
