# {py:mod}`better_robot.data_model.model_values`

```{py:module} better_robot.data_model.model_values
```

```{autodoc2-docstring} better_robot.data_model.model_values
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModelValues <better_robot.data_model.model_values.ModelValues>`
  - ```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`packed_inertias_to_6x6 <better_robot.data_model.model_values.packed_inertias_to_6x6>`
  - ```{autodoc2-docstring} better_robot.data_model.model_values.packed_inertias_to_6x6
    :summary:
    ```
````

### API

````{py:function} packed_inertias_to_6x6(body_inertias: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.model_values.packed_inertias_to_6x6

```{autodoc2-docstring} better_robot.data_model.model_values.packed_inertias_to_6x6
```
````

`````{py:class} ModelValues
:canonical: better_robot.data_model.model_values.ModelValues

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues
```

````{py:attribute} joint_placements
:canonical: better_robot.data_model.model_values.ModelValues.joint_placements
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.joint_placements
```

````

````{py:attribute} body_inertias
:canonical: better_robot.data_model.model_values.ModelValues.body_inertias
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.body_inertias
```

````

````{py:attribute} frame_placements
:canonical: better_robot.data_model.model_values.ModelValues.frame_placements
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.frame_placements
```

````

````{py:attribute} lower_pos_limit
:canonical: better_robot.data_model.model_values.ModelValues.lower_pos_limit
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.lower_pos_limit
```

````

````{py:attribute} upper_pos_limit
:canonical: better_robot.data_model.model_values.ModelValues.upper_pos_limit
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.upper_pos_limit
```

````

````{py:attribute} velocity_limit
:canonical: better_robot.data_model.model_values.ModelValues.velocity_limit
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.velocity_limit
```

````

````{py:attribute} effort_limit
:canonical: better_robot.data_model.model_values.ModelValues.effort_limit
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.effort_limit
```

````

````{py:attribute} rotor_inertia
:canonical: better_robot.data_model.model_values.ModelValues.rotor_inertia
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.rotor_inertia
```

````

````{py:attribute} armature
:canonical: better_robot.data_model.model_values.ModelValues.armature
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.armature
```

````

````{py:attribute} friction
:canonical: better_robot.data_model.model_values.ModelValues.friction
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.friction
```

````

````{py:attribute} damping
:canonical: better_robot.data_model.model_values.ModelValues.damping
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.damping
```

````

````{py:attribute} gravity
:canonical: better_robot.data_model.model_values.ModelValues.gravity
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.gravity
```

````

````{py:attribute} mimic_multiplier
:canonical: better_robot.data_model.model_values.ModelValues.mimic_multiplier
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.mimic_multiplier
```

````

````{py:attribute} mimic_offset
:canonical: better_robot.data_model.model_values.ModelValues.mimic_offset
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.mimic_offset
```

````

````{py:attribute} q_neutral
:canonical: better_robot.data_model.model_values.ModelValues.q_neutral
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.q_neutral
```

````

````{py:method} validate(structure: better_robot.data_model.model_structure.ModelStructure) -> None
:canonical: better_robot.data_model.model_values.ModelValues.validate

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.validate
```

````

````{py:method} from_model(model: better_robot.data_model.model.Model) -> better_robot.data_model.model_values.ModelValues
:canonical: better_robot.data_model.model_values.ModelValues.from_model
:classmethod:

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.from_model
```

````

````{py:method} spatial_inertias() -> torch.Tensor
:canonical: better_robot.data_model.model_values.ModelValues.spatial_inertias

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.spatial_inertias
```

````

````{py:method} to(device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> better_robot.data_model.model_values.ModelValues
:canonical: better_robot.data_model.model_values.ModelValues.to

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.to
```

````

````{py:method} tree_flatten() -> tuple[list[torch.Tensor], tuple[str, ...]]
:canonical: better_robot.data_model.model_values.ModelValues.tree_flatten

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.tree_flatten
```

````

````{py:method} tree_unflatten(leaves: list[torch.Tensor], context: tuple[str, ...]) -> better_robot.data_model.model_values.ModelValues
:canonical: better_robot.data_model.model_values.ModelValues.tree_unflatten
:classmethod:

```{autodoc2-docstring} better_robot.data_model.model_values.ModelValues.tree_unflatten
```

````

`````
