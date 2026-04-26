# {py:mod}`better_robot.data_model.joint_models.base`

```{py:module} better_robot.data_model.joint_models.base
```

```{autodoc2-docstring} better_robot.data_model.joint_models.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointModel <better_robot.data_model.joint_models.base.JointModel>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`zero_joint_bias_acceleration <better_robot.data_model.joint_models.base.zero_joint_bias_acceleration>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.base.zero_joint_bias_acceleration
    :summary:
    ```
* - {py:obj}`zero_joint_motion_subspace_derivative <better_robot.data_model.joint_models.base.zero_joint_motion_subspace_derivative>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.base.zero_joint_motion_subspace_derivative
    :summary:
    ```
* - {py:obj}`joint_bias_acceleration <better_robot.data_model.joint_models.base.joint_bias_acceleration>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.base.joint_bias_acceleration
    :summary:
    ```
* - {py:obj}`joint_motion_subspace_derivative <better_robot.data_model.joint_models.base.joint_motion_subspace_derivative>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.base.joint_motion_subspace_derivative
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointKind <better_robot.data_model.joint_models.base.JointKind>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointKind
    :summary:
    ```
````

### API

````{py:data} JointKind
:canonical: better_robot.data_model.joint_models.base.JointKind
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointKind
```

````

`````{py:class} JointModel
:canonical: better_robot.data_model.joint_models.base.JointModel

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.base.JointModel.kind
:type: better_robot.data_model.joint_models.base.JointKind
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.base.JointModel.nq
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.base.JointModel.nv
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.base.JointModel.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.axis
```

````

````{py:method} joint_transform(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.JointModel.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.JointModel.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.JointModel.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.joint_velocity
```

````

````{py:method} integrate(q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.JointModel.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.integrate
```

````

````{py:method} difference(q0_slice: torch.Tensor, q1_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.JointModel.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.difference
```

````

````{py:method} random_configuration(generator: torch.Generator | None, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.JointModel.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.random_configuration
```

````

````{py:method} neutral() -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.JointModel.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.base.JointModel.neutral
```

````

`````

````{py:function} zero_joint_bias_acceleration(q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.zero_joint_bias_acceleration

```{autodoc2-docstring} better_robot.data_model.joint_models.base.zero_joint_bias_acceleration
```
````

````{py:function} zero_joint_motion_subspace_derivative(q_slice: torch.Tensor, v_slice: torch.Tensor, nv: int) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.zero_joint_motion_subspace_derivative

```{autodoc2-docstring} better_robot.data_model.joint_models.base.zero_joint_motion_subspace_derivative
```
````

````{py:function} joint_bias_acceleration(jm: better_robot.data_model.joint_models.base.JointModel, q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.joint_bias_acceleration

```{autodoc2-docstring} better_robot.data_model.joint_models.base.joint_bias_acceleration
```
````

````{py:function} joint_motion_subspace_derivative(jm: better_robot.data_model.joint_models.base.JointModel, q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.joint_models.base.joint_motion_subspace_derivative

```{autodoc2-docstring} better_robot.data_model.joint_models.base.joint_motion_subspace_derivative
```
````
