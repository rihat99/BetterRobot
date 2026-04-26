# {py:mod}`better_robot.data_model.joint_models.helical`

```{py:module} better_robot.data_model.joint_models.helical
```

```{autodoc2-docstring} better_robot.data_model.joint_models.helical
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointHelical <better_robot.data_model.joint_models.helical.JointHelical>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical
    :summary:
    ```
````

### API

`````{py:class} JointHelical
:canonical: better_robot.data_model.joint_models.helical.JointHelical

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical
```

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.helical.JointHelical.axis
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.axis
```

````

````{py:attribute} pitch
:canonical: better_robot.data_model.joint_models.helical.JointHelical.pitch
:type: float
:value: >
   0.0

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.pitch
```

````

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.helical.JointHelical.kind
:type: str
:value: >
   'helical'

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.helical.JointHelical.nq
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.helical.JointHelical.nv
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.nv
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.helical.JointHelical.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.helical.JointHelical.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.helical.JointHelical.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.helical.JointHelical.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.integrate
```

````

````{py:method} difference(q0_slice, q1_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.helical.JointHelical.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.difference
```

````

````{py:method} random_configuration(generator, lower, upper) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.helical.JointHelical.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.helical.JointHelical.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.helical.JointHelical.neutral
```

````

`````
