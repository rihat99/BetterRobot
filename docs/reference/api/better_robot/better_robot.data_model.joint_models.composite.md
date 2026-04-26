# {py:mod}`better_robot.data_model.joint_models.composite`

```{py:module} better_robot.data_model.joint_models.composite
```

```{autodoc2-docstring} better_robot.data_model.joint_models.composite
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointComposite <better_robot.data_model.joint_models.composite.JointComposite>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite
    :summary:
    ```
````

### API

`````{py:class} JointComposite
:canonical: better_robot.data_model.joint_models.composite.JointComposite

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite
```

````{py:attribute} sub_joints
:canonical: better_robot.data_model.joint_models.composite.JointComposite.sub_joints
:type: tuple[typing.Any, ...]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.sub_joints
```

````

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.composite.JointComposite.kind
:type: str
:value: >
   'composite'

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.composite.JointComposite.nq
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.composite.JointComposite.nv
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.composite.JointComposite.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.axis
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.composite.JointComposite.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.composite.JointComposite.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.composite.JointComposite.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.composite.JointComposite.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.integrate
```

````

````{py:method} difference(q0_slice, q1_slice) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.composite.JointComposite.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.difference
```

````

````{py:method} random_configuration(generator, lower, upper) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.composite.JointComposite.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.composite.JointComposite.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.composite.JointComposite.neutral
```

````

`````
