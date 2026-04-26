# {py:mod}`better_robot.data_model.data`

```{py:module} better_robot.data_model.data
```

```{autodoc2-docstring} better_robot.data_model.data
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Data <better_robot.data_model.data.Data>`
  - ```{autodoc2-docstring} better_robot.data_model.data.Data
    :summary:
    ```
````

### API

`````{py:class} Data
:canonical: better_robot.data_model.data.Data

```{autodoc2-docstring} better_robot.data_model.data.Data
```

````{py:attribute} q
:canonical: better_robot.data_model.data.Data.q
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.q
```

````

````{py:attribute} v
:canonical: better_robot.data_model.data.Data.v
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.v
```

````

````{py:attribute} a
:canonical: better_robot.data_model.data.Data.a
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.a
```

````

````{py:attribute} tau
:canonical: better_robot.data_model.data.Data.tau
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.tau
```

````

````{py:attribute} joint_pose_local
:canonical: better_robot.data_model.data.Data.joint_pose_local
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.joint_pose_local
```

````

````{py:attribute} joint_pose_world
:canonical: better_robot.data_model.data.Data.joint_pose_world
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.joint_pose_world
```

````

````{py:attribute} frame_pose_world
:canonical: better_robot.data_model.data.Data.frame_pose_world
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.frame_pose_world
```

````

````{py:attribute} joint_velocity_world
:canonical: better_robot.data_model.data.Data.joint_velocity_world
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.joint_velocity_world
```

````

````{py:attribute} joint_velocity_local
:canonical: better_robot.data_model.data.Data.joint_velocity_local
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.joint_velocity_local
```

````

````{py:attribute} joint_acceleration_world
:canonical: better_robot.data_model.data.Data.joint_acceleration_world
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.joint_acceleration_world
```

````

````{py:attribute} joint_acceleration_local
:canonical: better_robot.data_model.data.Data.joint_acceleration_local
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.joint_acceleration_local
```

````

````{py:attribute} joint_forces
:canonical: better_robot.data_model.data.Data.joint_forces
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.joint_forces
```

````

````{py:attribute} joint_jacobians
:canonical: better_robot.data_model.data.Data.joint_jacobians
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.joint_jacobians
```

````

````{py:attribute} joint_jacobians_dot
:canonical: better_robot.data_model.data.Data.joint_jacobians_dot
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.joint_jacobians_dot
```

````

````{py:attribute} mass_matrix
:canonical: better_robot.data_model.data.Data.mass_matrix
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.mass_matrix
```

````

````{py:attribute} coriolis_matrix
:canonical: better_robot.data_model.data.Data.coriolis_matrix
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.coriolis_matrix
```

````

````{py:attribute} gravity_torque
:canonical: better_robot.data_model.data.Data.gravity_torque
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.gravity_torque
```

````

````{py:attribute} bias_forces
:canonical: better_robot.data_model.data.Data.bias_forces
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.bias_forces
```

````

````{py:attribute} ddq
:canonical: better_robot.data_model.data.Data.ddq
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.ddq
```

````

````{py:attribute} centroidal_momentum_matrix
:canonical: better_robot.data_model.data.Data.centroidal_momentum_matrix
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.centroidal_momentum_matrix
```

````

````{py:attribute} centroidal_momentum
:canonical: better_robot.data_model.data.Data.centroidal_momentum
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.centroidal_momentum
```

````

````{py:attribute} com_position
:canonical: better_robot.data_model.data.Data.com_position
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.com_position
```

````

````{py:attribute} com_velocity
:canonical: better_robot.data_model.data.Data.com_velocity
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.com_velocity
```

````

````{py:attribute} com_acceleration
:canonical: better_robot.data_model.data.Data.com_acceleration
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.data.Data.com_acceleration
```

````

````{py:method} reset() -> None
:canonical: better_robot.data_model.data.Data.reset

```{autodoc2-docstring} better_robot.data_model.data.Data.reset
```

````

````{py:method} invalidate(level: better_robot.data_model._kinematics_level.KinematicsLevel = KinematicsLevel.NONE) -> None
:canonical: better_robot.data_model.data.Data.invalidate

```{autodoc2-docstring} better_robot.data_model.data.Data.invalidate
```

````

````{py:method} require(level: better_robot.data_model._kinematics_level.KinematicsLevel) -> None
:canonical: better_robot.data_model.data.Data.require

```{autodoc2-docstring} better_robot.data_model.data.Data.require
```

````

````{py:method} clone() -> better_robot.data_model.data.Data
:canonical: better_robot.data_model.data.Data.clone

```{autodoc2-docstring} better_robot.data_model.data.Data.clone
```

````

````{py:property} batch_shape
:canonical: better_robot.data_model.data.Data.batch_shape
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.data.Data.batch_shape
```

````

````{py:method} joint_pose(joint_id: int)
:canonical: better_robot.data_model.data.Data.joint_pose

```{autodoc2-docstring} better_robot.data_model.data.Data.joint_pose
```

````

````{py:method} frame_pose(frame_id: int)
:canonical: better_robot.data_model.data.Data.frame_pose

```{autodoc2-docstring} better_robot.data_model.data.Data.frame_pose
```

````

`````
