# {py:mod}`better_robot.optim.variables`

```{py:module} better_robot.optim.variables
```

```{autodoc2-docstring} better_robot.optim.variables
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VarSpec <better_robot.optim.variables.VarSpec>`
  - ```{autodoc2-docstring} better_robot.optim.variables.VarSpec
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`detach_values <better_robot.optim.variables.detach_values>`
  - ```{autodoc2-docstring} better_robot.optim.variables.detach_values
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Values <better_robot.optim.variables.Values>`
  - ```{autodoc2-docstring} better_robot.optim.variables.Values
    :summary:
    ```
````

### API

````{py:data} Values
:canonical: better_robot.optim.variables.Values
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.variables.Values
```

````

`````{py:class} VarSpec
:canonical: better_robot.optim.variables.VarSpec

```{autodoc2-docstring} better_robot.optim.variables.VarSpec
```

````{py:attribute} name
:canonical: better_robot.optim.variables.VarSpec.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.name
```

````

````{py:attribute} shape
:canonical: better_robot.optim.variables.VarSpec.shape
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.shape
```

````

````{py:attribute} manifold
:canonical: better_robot.optim.variables.VarSpec.manifold
:type: better_robot.optim.manifolds.Manifold
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.manifold
```

````

````{py:attribute} bounds
:canonical: better_robot.optim.variables.VarSpec.bounds
:type: better_robot.optim.manifolds.Bounds | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.bounds
```

````

````{py:attribute} scale
:canonical: better_robot.optim.variables.VarSpec.scale
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.scale
```

````

````{py:attribute} mask
:canonical: better_robot.optim.variables.VarSpec.mask
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.mask
```

````

````{py:attribute} time_axis
:canonical: better_robot.optim.variables.VarSpec.time_axis
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.time_axis
```

````

````{py:property} tangent_dim
:canonical: better_robot.optim.variables.VarSpec.tangent_dim
:type: int

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.tangent_dim
```

````

````{py:property} free_dim
:canonical: better_robot.optim.variables.VarSpec.free_dim
:type: int

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.free_dim
```

````

````{py:property} free_indices
:canonical: better_robot.optim.variables.VarSpec.free_indices
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.free_indices
```

````

````{py:property} free_scale
:canonical: better_robot.optim.variables.VarSpec.free_scale
:type: torch.Tensor | None

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.free_scale
```

````

````{py:property} time_length
:canonical: better_robot.optim.variables.VarSpec.time_length
:type: int

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.time_length
```

````

````{py:property} temporal_tangent_width
:canonical: better_robot.optim.variables.VarSpec.temporal_tangent_width
:type: int

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.temporal_tangent_width
```

````

````{py:property} temporal_mask_is_separable
:canonical: better_robot.optim.variables.VarSpec.temporal_mask_is_separable
:type: bool

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.temporal_mask_is_separable
```

````

````{py:property} temporal_free_indices
:canonical: better_robot.optim.variables.VarSpec.temporal_free_indices
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.temporal_free_indices
```

````

````{py:property} temporal_reduced_width
:canonical: better_robot.optim.variables.VarSpec.temporal_reduced_width
:type: int

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.temporal_reduced_width
```

````

````{py:method} batch_shape(value: torch.Tensor) -> tuple[int, ...]
:canonical: better_robot.optim.variables.VarSpec.batch_shape

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.batch_shape
```

````

````{py:method} validate_value(value: torch.Tensor) -> None
:canonical: better_robot.optim.variables.VarSpec.validate_value

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.validate_value
```

````

````{py:method} gather_tangent(full: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.variables.VarSpec.gather_tangent

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.gather_tangent
```

````

````{py:method} expand_tangent(reduced: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.variables.VarSpec.expand_tangent

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.expand_tangent
```

````

````{py:method} retract(value: torch.Tensor, reduced_delta: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.variables.VarSpec.retract

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.retract
```

````

````{py:method} difference(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.variables.VarSpec.difference

```{autodoc2-docstring} better_robot.optim.variables.VarSpec.difference
```

````

`````

````{py:function} detach_values(values: better_robot.optim.variables.Values) -> better_robot.optim.variables.Values
:canonical: better_robot.optim.variables.detach_values

```{autodoc2-docstring} better_robot.optim.variables.detach_values
```
````
