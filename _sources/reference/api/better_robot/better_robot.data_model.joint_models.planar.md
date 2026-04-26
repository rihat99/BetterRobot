# {py:mod}`better_robot.data_model.joint_models.planar`

```{py:module} better_robot.data_model.joint_models.planar
```

```{autodoc2-docstring} better_robot.data_model.joint_models.planar
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointPlanar <better_robot.data_model.joint_models.planar.JointPlanar>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar
    :summary:
    ```
````

### API

`````{py:class} JointPlanar
:canonical: better_robot.data_model.joint_models.planar.JointPlanar

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.kind
:type: str
:value: >
   'planar'

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.nq
:type: int
:value: >
   4

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.nv
:type: int
:value: >
   3

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.axis
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.joint_velocity
```

````

````{py:method} integrate(q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.integrate
```

````

````{py:method} difference(q0_slice: torch.Tensor, q1_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.difference
```

````

````{py:method} random_configuration(generator, lower, upper) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.planar.JointPlanar.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.planar.JointPlanar.neutral
```

````

`````
