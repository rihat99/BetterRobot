# {py:mod}`better_robot.data_model.joint_models.fixed`

```{py:module} better_robot.data_model.joint_models.fixed
```

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointFixed <better_robot.data_model.joint_models.fixed.JointFixed>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed
    :summary:
    ```
* - {py:obj}`JointUniverse <better_robot.data_model.joint_models.fixed.JointUniverse>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse
    :summary:
    ```
````

### API

`````{py:class} JointFixed
:canonical: better_robot.data_model.joint_models.fixed.JointFixed

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.kind
:type: str
:value: >
   'fixed'

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.nq
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.nv
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.axis
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.integrate
```

````

````{py:method} difference(q0_slice, q1_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.difference
```

````

````{py:method} random_configuration(generator, lower, upper) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointFixed.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointFixed.neutral
```

````

`````

`````{py:class} JointUniverse
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.kind
:type: str
:value: >
   'universe'

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.nq
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.nv
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.axis
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.integrate
```

````

````{py:method} difference(q0_slice, q1_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.difference
```

````

````{py:method} random_configuration(generator, lower, upper) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.fixed.JointUniverse.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.fixed.JointUniverse.neutral
```

````

`````
