# {py:mod}`better_robot.residuals.base`

```{py:module} better_robot.residuals.base
```

```{autodoc2-docstring} better_robot.residuals.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Weight <better_robot.residuals.base.Weight>`
  - ```{autodoc2-docstring} better_robot.residuals.base.Weight
    :summary:
    ```
* - {py:obj}`ScaleWeight <better_robot.residuals.base.ScaleWeight>`
  - ```{autodoc2-docstring} better_robot.residuals.base.ScaleWeight
    :summary:
    ```
* - {py:obj}`DiagonalWeight <better_robot.residuals.base.DiagonalWeight>`
  - ```{autodoc2-docstring} better_robot.residuals.base.DiagonalWeight
    :summary:
    ```
* - {py:obj}`Residual <better_robot.residuals.base.Residual>`
  - ```{autodoc2-docstring} better_robot.residuals.base.Residual
    :summary:
    ```
* - {py:obj}`ScalarCost <better_robot.residuals.base.ScalarCost>`
  - ```{autodoc2-docstring} better_robot.residuals.base.ScalarCost
    :summary:
    ```
* - {py:obj}`Difference <better_robot.residuals.base.Difference>`
  - ```{autodoc2-docstring} better_robot.residuals.base.Difference
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`residual <better_robot.residuals.base.residual>`
  - ```{autodoc2-docstring} better_robot.residuals.base.residual
    :summary:
    ```
````

### API

`````{py:class} Weight
:canonical: better_robot.residuals.base.Weight

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} better_robot.residuals.base.Weight
```

````{py:method} apply(error: torch.Tensor) -> torch.Tensor
:canonical: better_robot.residuals.base.Weight.apply
:abstractmethod:

```{autodoc2-docstring} better_robot.residuals.base.Weight.apply
```

````

````{py:method} apply_jacobian(blocks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]
:canonical: better_robot.residuals.base.Weight.apply_jacobian
:abstractmethod:

```{autodoc2-docstring} better_robot.residuals.base.Weight.apply_jacobian
```

````

````{py:method} is_inactive() -> bool
:canonical: better_robot.residuals.base.Weight.is_inactive

```{autodoc2-docstring} better_robot.residuals.base.Weight.is_inactive
```

````

`````

`````{py:class} ScaleWeight(value: numbers.Real | torch.Tensor)
:canonical: better_robot.residuals.base.ScaleWeight

Bases: {py:obj}`better_robot.residuals.base.Weight`

```{autodoc2-docstring} better_robot.residuals.base.ScaleWeight
```

````{py:method} apply(error: torch.Tensor) -> torch.Tensor
:canonical: better_robot.residuals.base.ScaleWeight.apply

````

````{py:method} apply_jacobian(blocks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]
:canonical: better_robot.residuals.base.ScaleWeight.apply_jacobian

````

````{py:method} is_inactive() -> bool
:canonical: better_robot.residuals.base.ScaleWeight.is_inactive

````

`````

`````{py:class} DiagonalWeight(diagonal: torch.Tensor)
:canonical: better_robot.residuals.base.DiagonalWeight

Bases: {py:obj}`better_robot.residuals.base.Weight`

```{autodoc2-docstring} better_robot.residuals.base.DiagonalWeight
```

````{py:method} apply(error: torch.Tensor) -> torch.Tensor
:canonical: better_robot.residuals.base.DiagonalWeight.apply

````

````{py:method} apply_jacobian(blocks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]
:canonical: better_robot.residuals.base.DiagonalWeight.apply_jacobian

````

`````

`````{py:class} Residual(*variables: better_robot.residuals.utils.VariableLike, dim: int, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, reduce: typing.Literal[sum, mean, mean_active] = 'sum', kernel: object | None = None, group_size: int = 1, name: str | None = None, enabled: bool = True)
:canonical: better_robot.residuals.base.Residual

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} better_robot.residuals.base.Residual
```

````{py:property} weight
:canonical: better_robot.residuals.base.Residual.weight
:type: numbers.Real | torch.Tensor

```{autodoc2-docstring} better_robot.residuals.base.Residual.weight
```

````

````{py:property} row_weight
:canonical: better_robot.residuals.base.Residual.row_weight
:type: better_robot.residuals.base.Weight

```{autodoc2-docstring} better_robot.residuals.base.Residual.row_weight
```

````

````{py:property} reduce
:canonical: better_robot.residuals.base.Residual.reduce
:type: typing.Literal[sum, mean, mean_active]

```{autodoc2-docstring} better_robot.residuals.base.Residual.reduce
```

````

````{py:property} enabled
:canonical: better_robot.residuals.base.Residual.enabled
:type: bool

```{autodoc2-docstring} better_robot.residuals.base.Residual.enabled
```

````

````{py:method} is_inactive() -> bool
:canonical: better_robot.residuals.base.Residual.is_inactive

```{autodoc2-docstring} better_robot.residuals.base.Residual.is_inactive
```

````

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.base.Residual.error
:abstractmethod:

```{autodoc2-docstring} better_robot.residuals.base.Residual.error
```

````

````{py:method} jacobian() -> tuple[torch.Tensor, ...] | None
:canonical: better_robot.residuals.base.Residual.jacobian

```{autodoc2-docstring} better_robot.residuals.base.Residual.jacobian
```

````

````{py:method} active_groups() -> torch.Tensor | None
:canonical: better_robot.residuals.base.Residual.active_groups

```{autodoc2-docstring} better_robot.residuals.base.Residual.active_groups
```

````

````{py:method} weighted_error() -> torch.Tensor
:canonical: better_robot.residuals.base.Residual.weighted_error

```{autodoc2-docstring} better_robot.residuals.base.Residual.weighted_error
```

````

`````

`````{py:class} ScalarCost(fn: collections.abc.Callable[..., torch.Tensor], *reads: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node, weight: numbers.Real | torch.Tensor = 1.0, name: str | None = None)
:canonical: better_robot.residuals.base.ScalarCost

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.base.ScalarCost
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.base.ScalarCost.error

````

`````

````{py:function} residual(*variables_or_fn: better_robot.residuals.utils.VariableLike | collections.abc.Callable[..., torch.Tensor], dim: int, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, reduce: typing.Literal[sum, mean, mean_active] = 'sum', kernel: object | None = None, group_size: int = 1, name: str | None = None, enabled: bool = True)
:canonical: better_robot.residuals.base.residual

```{autodoc2-docstring} better_robot.residuals.base.residual
```
````

`````{py:class} Difference(variable: better_robot.residuals.utils.VariableLike, target: torch.Tensor, *, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, reduce: typing.Literal[sum, mean, mean_active] = 'sum', kernel: object | None = None, name: str | None = None, enabled: bool = True)
:canonical: better_robot.residuals.base.Difference

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.base.Difference
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.base.Difference.error

````

`````
