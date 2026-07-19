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

* - {py:obj}`Residual <better_robot.optim.problem.Residual>`
  -
* - {py:obj}`ResidualItem <better_robot.optim.problem.ResidualItem>`
  - ```{autodoc2-docstring} better_robot.optim.problem.ResidualItem
    :summary:
    ```
* - {py:obj}`Problem <better_robot.optim.problem.Problem>`
  - ```{autodoc2-docstring} better_robot.optim.problem.Problem
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Weight <better_robot.optim.problem.Weight>`
  - ```{autodoc2-docstring} better_robot.optim.problem.Weight
    :summary:
    ```
* - {py:obj}`JacobianStrategy <better_robot.optim.problem.JacobianStrategy>`
  - ```{autodoc2-docstring} better_robot.optim.problem.JacobianStrategy
    :summary:
    ```
````

### API

````{py:data} Weight
:canonical: better_robot.optim.problem.Weight
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.Weight
```

````

````{py:data} JacobianStrategy
:canonical: better_robot.optim.problem.JacobianStrategy
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.JacobianStrategy
```

````

`````{py:class} Residual
:canonical: better_robot.optim.problem.Residual

Bases: {py:obj}`typing.Protocol`

````{py:attribute} name
:canonical: better_robot.optim.problem.Residual.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.Residual.name
```

````

````{py:attribute} reads
:canonical: better_robot.optim.problem.Residual.reads
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.Residual.reads
```

````

````{py:attribute} dim
:canonical: better_robot.optim.problem.Residual.dim
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.Residual.dim
```

````

`````

`````{py:class} ResidualItem
:canonical: better_robot.optim.problem.ResidualItem

```{autodoc2-docstring} better_robot.optim.problem.ResidualItem
```

````{py:attribute} name
:canonical: better_robot.optim.problem.ResidualItem.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.ResidualItem.name
```

````

````{py:attribute} residual
:canonical: better_robot.optim.problem.ResidualItem.residual
:type: better_robot.optim.problem.Residual
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.ResidualItem.residual
```

````

````{py:attribute} weight
:canonical: better_robot.optim.problem.ResidualItem.weight
:type: better_robot.optim.problem.Weight
:value: >
   1.0

```{autodoc2-docstring} better_robot.optim.problem.ResidualItem.weight
```

````

````{py:attribute} kernel
:canonical: better_robot.optim.problem.ResidualItem.kernel
:type: better_robot.optim.kernels.RobustKernel | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.problem.ResidualItem.kernel
```

````

````{py:attribute} group_size
:canonical: better_robot.optim.problem.ResidualItem.group_size
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.optim.problem.ResidualItem.group_size
```

````

`````

`````{py:class} Problem(*, vars: collections.abc.Sequence[better_robot.optim.variables.VarSpec] = (), residuals: collections.abc.Sequence[better_robot.optim.problem.ResidualItem] = (), providers: collections.abc.Sequence[better_robot.optim.providers.Provider] = (), parameters: collections.abc.Mapping[str, torch.Tensor] | None = None, differentiable_parameters: collections.abc.Sequence[str] = ())
:canonical: better_robot.optim.problem.Problem

```{autodoc2-docstring} better_robot.optim.problem.Problem
```

````{py:method} add_variable(name: str, *, shape: tuple[int, ...] | None = None, manifold: better_robot.optim.manifolds.Manifold = Euclidean(), bounds: better_robot.optim.manifolds.Bounds | None = None, mask: torch.Tensor | None = None, scale: torch.Tensor | None = None, time_axis: int | None = None) -> better_robot.optim.variables.VarSpec
:canonical: better_robot.optim.problem.Problem.add_variable

```{autodoc2-docstring} better_robot.optim.problem.Problem.add_variable
```

````

````{py:method} add_residual(residual: collections.abc.Callable[[collections.abc.Mapping[str, typing.Any]], torch.Tensor], *, weight: better_robot.optim.problem.Weight = 1.0, kernel: better_robot.optim.kernels.RobustKernel | None = None, name: str | None = None, dim: int | None = None) -> better_robot.optim.problem.ResidualItem
:canonical: better_robot.optim.problem.Problem.add_residual

```{autodoc2-docstring} better_robot.optim.problem.Problem.add_residual
```

````

````{py:method} residual(values: collections.abc.Mapping[str, torch.Tensor], *, weights: collections.abc.Mapping[str, better_robot.optim.problem.Weight] | None = None) -> torch.Tensor
:canonical: better_robot.optim.problem.Problem.residual

```{autodoc2-docstring} better_robot.optim.problem.Problem.residual
```

````

````{py:method} objective(values: collections.abc.Mapping[str, torch.Tensor], *, weights: collections.abc.Mapping[str, better_robot.optim.problem.Weight] | None = None) -> torch.Tensor
:canonical: better_robot.optim.problem.Problem.objective

```{autodoc2-docstring} better_robot.optim.problem.Problem.objective
```

````

````{py:method} gradient(values: collections.abc.Mapping[str, torch.Tensor], *, weights: collections.abc.Mapping[str, better_robot.optim.problem.Weight] | None = None, create_graph: bool = False) -> better_robot.optim.variables.Values
:canonical: better_robot.optim.problem.Problem.gradient

```{autodoc2-docstring} better_robot.optim.problem.Problem.gradient
```

````

````{py:method} structured_normal(values: collections.abc.Mapping[str, torch.Tensor], *, weights: collections.abc.Mapping[str, better_robot.optim.problem.Weight] | None = None, row_scale: torch.Tensor | None = None, residual: torch.Tensor | None = None, create_graph: bool = False) -> better_robot.optim.temporal.StructuredNormal
:canonical: better_robot.optim.problem.Problem.structured_normal

```{autodoc2-docstring} better_robot.optim.problem.Problem.structured_normal
```

````

````{py:method} retract(values: collections.abc.Mapping[str, torch.Tensor], steps: collections.abc.Mapping[str, torch.Tensor]) -> better_robot.optim.variables.Values
:canonical: better_robot.optim.problem.Problem.retract

```{autodoc2-docstring} better_robot.optim.problem.Problem.retract
```

````

````{py:method} difference(x0: collections.abc.Mapping[str, torch.Tensor], x1: collections.abc.Mapping[str, torch.Tensor]) -> better_robot.optim.variables.Values
:canonical: better_robot.optim.problem.Problem.difference

```{autodoc2-docstring} better_robot.optim.problem.Problem.difference
```

````

````{py:method} jacobian_blocks(values: collections.abc.Mapping[str, torch.Tensor], *, weights: collections.abc.Mapping[str, better_robot.optim.problem.Weight] | None = None, strategy: better_robot.optim.problem.JacobianStrategy = 'auto', create_graph: bool = False, fd_eps: float = 0.0001) -> dict[tuple[str, str], torch.Tensor]
:canonical: better_robot.optim.problem.Problem.jacobian_blocks

```{autodoc2-docstring} better_robot.optim.problem.Problem.jacobian_blocks
```

````

````{py:method} dense_jacobian(values: collections.abc.Mapping[str, torch.Tensor], *, weights: collections.abc.Mapping[str, better_robot.optim.problem.Weight] | None = None, strategy: better_robot.optim.problem.JacobianStrategy = 'auto', create_graph: bool = False) -> torch.Tensor
:canonical: better_robot.optim.problem.Problem.dense_jacobian

```{autodoc2-docstring} better_robot.optim.problem.Problem.dense_jacobian
```

````

````{py:property} external_parameters
:canonical: better_robot.optim.problem.Problem.external_parameters
:type: collections.abc.Mapping[str, torch.Tensor]

```{autodoc2-docstring} better_robot.optim.problem.Problem.external_parameters
```

````

````{py:property} differentiable_external_parameters
:canonical: better_robot.optim.problem.Problem.differentiable_external_parameters
:type: collections.abc.Mapping[str, torch.Tensor]

```{autodoc2-docstring} better_robot.optim.problem.Problem.differentiable_external_parameters
```

````

`````
