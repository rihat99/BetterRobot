# {py:mod}`better_robot.data_model.joint_models.translation`

```{py:module} better_robot.data_model.joint_models.translation
```

```{autodoc2-docstring} better_robot.data_model.joint_models.translation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointTranslation <better_robot.data_model.joint_models.translation.JointTranslation>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation
    :summary:
    ```
````

### API

`````{py:class} JointTranslation
:canonical: better_robot.data_model.joint_models.translation.JointTranslation

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.kind
:type: str
:value: >
   'translation'

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.nq
:type: int
:value: >
   3

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.nv
:type: int
:value: >
   3

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.axis
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.integrate
```

````

````{py:method} difference(q0_slice, q1_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.difference
```

````

````{py:method} random_configuration(generator, lower, upper) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.translation.JointTranslation.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.translation.JointTranslation.neutral
```

````

`````
