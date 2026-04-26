# {py:mod}`better_robot.optim.jacobian_spec`

```{py:module} better_robot.optim.jacobian_spec
```

```{autodoc2-docstring} better_robot.optim.jacobian_spec
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ResidualSpec <better_robot.optim.jacobian_spec.ResidualSpec>`
  - ```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ResidualStructure <better_robot.optim.jacobian_spec.ResidualStructure>`
  - ```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualStructure
    :summary:
    ```
* - {py:obj}`TimeCoupling <better_robot.optim.jacobian_spec.TimeCoupling>`
  - ```{autodoc2-docstring} better_robot.optim.jacobian_spec.TimeCoupling
    :summary:
    ```
````

### API

````{py:data} ResidualStructure
:canonical: better_robot.optim.jacobian_spec.ResidualStructure
:value: >
   None

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualStructure
```

````

````{py:data} TimeCoupling
:canonical: better_robot.optim.jacobian_spec.TimeCoupling
:value: >
   None

```{autodoc2-docstring} better_robot.optim.jacobian_spec.TimeCoupling
```

````

`````{py:class} ResidualSpec
:canonical: better_robot.optim.jacobian_spec.ResidualSpec

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec
```

````{py:attribute} dim
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.dim
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.dim
```

````

````{py:attribute} output_dim
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.output_dim
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.output_dim
```

````

````{py:attribute} tangent_dim
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.tangent_dim
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.tangent_dim
```

````

````{py:attribute} structure
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.structure
:type: better_robot.optim.jacobian_spec.ResidualStructure
:value: >
   'dense'

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.structure
```

````

````{py:attribute} time_coupling
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.time_coupling
:type: better_robot.optim.jacobian_spec.TimeCoupling
:value: >
   'single'

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.time_coupling
```

````

````{py:attribute} affected_knots
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.affected_knots
:type: tuple[int, ...]
:value: >
   ()

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.affected_knots
```

````

````{py:attribute} affected_joints
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.affected_joints
:type: tuple[int, ...]
:value: >
   ()

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.affected_joints
```

````

````{py:attribute} affected_frames
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.affected_frames
:type: tuple[int, ...]
:value: >
   ()

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.affected_frames
```

````

````{py:attribute} dynamic_dim
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.dynamic_dim
:type: bool
:value: >
   False

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.dynamic_dim
```

````

````{py:attribute} input_indices
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.input_indices
:type: tuple[int, ...] | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.input_indices
```

````

````{py:attribute} is_diagonal
:canonical: better_robot.optim.jacobian_spec.ResidualSpec.is_diagonal
:type: bool
:value: >
   False

```{autodoc2-docstring} better_robot.optim.jacobian_spec.ResidualSpec.is_diagonal
```

````

`````
