# {py:mod}`better_robot.tasks.trajectory`

```{py:module} better_robot.tasks.trajectory
```

```{autodoc2-docstring} better_robot.tasks.trajectory
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Trajectory <better_robot.tasks.trajectory.Trajectory>`
  - ```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory
    :summary:
    ```
````

### API

`````{py:class} Trajectory
:canonical: better_robot.tasks.trajectory.Trajectory

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory
```

````{py:attribute} t
:canonical: better_robot.tasks.trajectory.Trajectory.t
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.t
```

````

````{py:attribute} q
:canonical: better_robot.tasks.trajectory.Trajectory.q
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.q
```

````

````{py:attribute} v
:canonical: better_robot.tasks.trajectory.Trajectory.v
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.v
```

````

````{py:attribute} a
:canonical: better_robot.tasks.trajectory.Trajectory.a
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.a
```

````

````{py:attribute} tau
:canonical: better_robot.tasks.trajectory.Trajectory.tau
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.tau
```

````

````{py:attribute} extras
:canonical: better_robot.tasks.trajectory.Trajectory.extras
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.extras
```

````

````{py:attribute} metadata
:canonical: better_robot.tasks.trajectory.Trajectory.metadata
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.metadata
```

````

````{py:attribute} model_id
:canonical: better_robot.tasks.trajectory.Trajectory.model_id
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.model_id
```

````

````{py:property} batch_shape
:canonical: better_robot.tasks.trajectory.Trajectory.batch_shape
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.batch_shape
```

````

````{py:property} num_knots
:canonical: better_robot.tasks.trajectory.Trajectory.num_knots
:type: int

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.num_knots
```

````

````{py:property} horizon
:canonical: better_robot.tasks.trajectory.Trajectory.horizon
:type: int

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.horizon
```

````

````{py:property} batch_size
:canonical: better_robot.tasks.trajectory.Trajectory.batch_size
:type: int

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.batch_size
```

````

````{py:method} with_batch_dims(n: int = 1) -> better_robot.tasks.trajectory.Trajectory
:canonical: better_robot.tasks.trajectory.Trajectory.with_batch_dims

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.with_batch_dims
```

````

````{py:method} slice(t_start: float, t_end: float) -> better_robot.tasks.trajectory.Trajectory
:canonical: better_robot.tasks.trajectory.Trajectory.slice

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.slice
```

````

````{py:method} resample(new_t: torch.Tensor, *, kind: typing.Literal[linear, sclerp] = 'linear') -> better_robot.tasks.trajectory.Trajectory
:canonical: better_robot.tasks.trajectory.Trajectory.resample

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.resample
```

````

````{py:method} downsample(factor: int) -> better_robot.tasks.trajectory.Trajectory
:canonical: better_robot.tasks.trajectory.Trajectory.downsample

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.downsample
```

````

````{py:method} to_data(model: better_robot.data_model.model.Model) -> better_robot.data_model.data.Data
:canonical: better_robot.tasks.trajectory.Trajectory.to_data

```{autodoc2-docstring} better_robot.tasks.trajectory.Trajectory.to_data
```

````

`````
