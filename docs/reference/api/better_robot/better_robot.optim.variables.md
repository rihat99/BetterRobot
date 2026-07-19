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

* - {py:obj}`Variable <better_robot.optim.variables.Variable>`
  - ```{autodoc2-docstring} better_robot.optim.variables.Variable
    :summary:
    ```
* - {py:obj}`SO3Variable <better_robot.optim.variables.SO3Variable>`
  - ```{autodoc2-docstring} better_robot.optim.variables.SO3Variable
    :summary:
    ```
* - {py:obj}`SE3Variable <better_robot.optim.variables.SE3Variable>`
  - ```{autodoc2-docstring} better_robot.optim.variables.SE3Variable
    :summary:
    ```
* - {py:obj}`RobotVariable <better_robot.optim.variables.RobotVariable>`
  - ```{autodoc2-docstring} better_robot.optim.variables.RobotVariable
    :summary:
    ```
````

### API

`````{py:class} Variable(tensor: torch.Tensor, *, name: str | None = None, trainable: bool = True, bounds: better_robot.optim.manifolds.Bounds | None = None, mask: torch.Tensor | None = None, scale: torch.Tensor | None = None, batch_ndim: int = 0, time_axis: int | None = None)
:canonical: better_robot.optim.variables.Variable

```{autodoc2-docstring} better_robot.optim.variables.Variable
```

````{py:method} tangent_dim() -> int
:canonical: better_robot.optim.variables.Variable.tangent_dim

```{autodoc2-docstring} better_robot.optim.variables.Variable.tangent_dim
```

````

````{py:property} free_dim
:canonical: better_robot.optim.variables.Variable.free_dim
:type: int

```{autodoc2-docstring} better_robot.optim.variables.Variable.free_dim
```

````

````{py:property} free_indices
:canonical: better_robot.optim.variables.Variable.free_indices
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim.variables.Variable.free_indices
```

````

````{py:property} free_scale
:canonical: better_robot.optim.variables.Variable.free_scale
:type: torch.Tensor | None

```{autodoc2-docstring} better_robot.optim.variables.Variable.free_scale
```

````

````{py:property} batch_shape
:canonical: better_robot.optim.variables.Variable.batch_shape
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.optim.variables.Variable.batch_shape
```

````

````{py:method} batch_shape_of(value: torch.Tensor) -> tuple[int, ...]
:canonical: better_robot.optim.variables.Variable.batch_shape_of

```{autodoc2-docstring} better_robot.optim.variables.Variable.batch_shape_of
```

````

````{py:property} time_length
:canonical: better_robot.optim.variables.Variable.time_length
:type: int

```{autodoc2-docstring} better_robot.optim.variables.Variable.time_length
```

````

````{py:property} temporal_tangent_width
:canonical: better_robot.optim.variables.Variable.temporal_tangent_width
:type: int

```{autodoc2-docstring} better_robot.optim.variables.Variable.temporal_tangent_width
```

````

````{py:property} temporal_mask_is_separable
:canonical: better_robot.optim.variables.Variable.temporal_mask_is_separable
:type: bool

```{autodoc2-docstring} better_robot.optim.variables.Variable.temporal_mask_is_separable
```

````

````{py:property} temporal_free_indices
:canonical: better_robot.optim.variables.Variable.temporal_free_indices
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim.variables.Variable.temporal_free_indices
```

````

````{py:property} temporal_reduced_width
:canonical: better_robot.optim.variables.Variable.temporal_reduced_width
:type: int

```{autodoc2-docstring} better_robot.optim.variables.Variable.temporal_reduced_width
```

````

````{py:method} validate_value(value: torch.Tensor) -> None
:canonical: better_robot.optim.variables.Variable.validate_value

```{autodoc2-docstring} better_robot.optim.variables.Variable.validate_value
```

````

````{py:method} gather_tangent(full: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.variables.Variable.gather_tangent

```{autodoc2-docstring} better_robot.optim.variables.Variable.gather_tangent
```

````

````{py:method} expand_tangent(reduced: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.variables.Variable.expand_tangent

```{autodoc2-docstring} better_robot.optim.variables.Variable.expand_tangent
```

````

````{py:method} project(value: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.variables.Variable.project

```{autodoc2-docstring} better_robot.optim.variables.Variable.project
```

````

````{py:method} retract(reduced_delta: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.variables.Variable.retract

```{autodoc2-docstring} better_robot.optim.variables.Variable.retract
```

````

````{py:method} difference(other: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.variables.Variable.difference

```{autodoc2-docstring} better_robot.optim.variables.Variable.difference
```

````

`````

`````{py:class} SO3Variable(tensor: torch.Tensor, *, name: str | None = None, trainable: bool = True, bounds: better_robot.optim.manifolds.Bounds | None = None, mask: torch.Tensor | None = None, scale: torch.Tensor | None = None, batch_ndim: int = 0, time_axis: int | None = None)
:canonical: better_robot.optim.variables.SO3Variable

Bases: {py:obj}`better_robot.optim.variables.Variable`

```{autodoc2-docstring} better_robot.optim.variables.SO3Variable
```

````{py:method} tangent_dim() -> int
:canonical: better_robot.optim.variables.SO3Variable.tangent_dim

````

`````

`````{py:class} SE3Variable(tensor: torch.Tensor, *, name: str | None = None, trainable: bool = True, bounds: better_robot.optim.manifolds.Bounds | None = None, mask: torch.Tensor | None = None, scale: torch.Tensor | None = None, batch_ndim: int = 0, time_axis: int | None = None)
:canonical: better_robot.optim.variables.SE3Variable

Bases: {py:obj}`better_robot.optim.variables.Variable`

```{autodoc2-docstring} better_robot.optim.variables.SE3Variable
```

````{py:method} tangent_dim() -> int
:canonical: better_robot.optim.variables.SE3Variable.tangent_dim

````

`````

`````{py:class} RobotVariable(model: better_robot.data_model.model.Model, tensor: torch.Tensor | None = None, *, name: str | None = None, trainable: bool = True, bounds: better_robot.optim.manifolds.Bounds | bool | None = None, mask: torch.Tensor | None = None, scale: torch.Tensor | None = None, batch_ndim: int = 0, time_axis: int | None = None)
:canonical: better_robot.optim.variables.RobotVariable

Bases: {py:obj}`better_robot.optim.variables.Variable`

```{autodoc2-docstring} better_robot.optim.variables.RobotVariable
```

````{py:property} box_mask
:canonical: better_robot.optim.variables.RobotVariable.box_mask
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim.variables.RobotVariable.box_mask
```

````

````{py:method} joint_bounds() -> better_robot.optim.manifolds.Bounds
:canonical: better_robot.optim.variables.RobotVariable.joint_bounds

```{autodoc2-docstring} better_robot.optim.variables.RobotVariable.joint_bounds
```

````

````{py:property} unit_coordinate_slices
:canonical: better_robot.optim.variables.RobotVariable.unit_coordinate_slices
:type: tuple[slice, ...]

```{autodoc2-docstring} better_robot.optim.variables.RobotVariable.unit_coordinate_slices
```

````

````{py:method} tangent_dim() -> int
:canonical: better_robot.optim.variables.RobotVariable.tangent_dim

````

`````
