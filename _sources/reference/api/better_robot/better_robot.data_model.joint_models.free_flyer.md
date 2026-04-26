# {py:mod}`better_robot.data_model.joint_models.free_flyer`

```{py:module} better_robot.data_model.joint_models.free_flyer
```

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointFreeFlyer <better_robot.data_model.joint_models.free_flyer.JointFreeFlyer>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer
    :summary:
    ```
````

### API

`````{py:class} JointFreeFlyer
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.kind
:type: str
:value: >
   'free_flyer'

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.nq
:type: int
:value: >
   7

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.nv
:type: int
:value: >
   6

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.axis
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.joint_velocity
```

````

````{py:method} integrate(q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.integrate
```

````

````{py:method} difference(q0_slice: torch.Tensor, q1_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.difference
```

````

````{py:method} random_configuration(generator, lower, upper) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.free_flyer.JointFreeFlyer.neutral
```

````

`````
