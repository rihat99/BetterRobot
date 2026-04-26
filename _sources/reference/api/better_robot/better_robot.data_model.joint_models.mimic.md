# {py:mod}`better_robot.data_model.joint_models.mimic`

```{py:module} better_robot.data_model.joint_models.mimic
```

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointMimic <better_robot.data_model.joint_models.mimic.JointMimic>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic
    :summary:
    ```
````

### API

`````{py:class} JointMimic
:canonical: better_robot.data_model.joint_models.mimic.JointMimic

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.kind
:type: str
:value: >
   'mimic'

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.nq
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.nv
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.axis
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.integrate
```

````

````{py:method} difference(q0_slice, q1_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.difference
```

````

````{py:method} random_configuration(generator, lower, upper) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.mimic.JointMimic.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.mimic.JointMimic.neutral
```

````

`````
