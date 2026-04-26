# {py:mod}`better_robot.data_model.joint_models.spherical`

```{py:module} better_robot.data_model.joint_models.spherical
```

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointSpherical <better_robot.data_model.joint_models.spherical.JointSpherical>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical
    :summary:
    ```
````

### API

`````{py:class} JointSpherical
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.kind
:type: str
:value: >
   'spherical'

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.nq
:type: int
:value: >
   4

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.nv
:type: int
:value: >
   3

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.axis
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.joint_velocity
```

````

````{py:method} integrate(q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.integrate
```

````

````{py:method} difference(q0_slice: torch.Tensor, q1_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.difference
```

````

````{py:method} random_configuration(generator, lower, upper) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.spherical.JointSpherical.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.spherical.JointSpherical.neutral
```

````

`````
