# {py:mod}`better_robot.kinematics.state`

```{py:module} better_robot.kinematics.state
```

```{autodoc2-docstring} better_robot.kinematics.state
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KinematicState <better_robot.kinematics.state.KinematicState>`
  - ```{autodoc2-docstring} better_robot.kinematics.state.KinematicState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`forward_kinematics_state <better_robot.kinematics.state.forward_kinematics_state>`
  - ```{autodoc2-docstring} better_robot.kinematics.state.forward_kinematics_state
    :summary:
    ```
* - {py:obj}`frame_jacobian <better_robot.kinematics.state.frame_jacobian>`
  - ```{autodoc2-docstring} better_robot.kinematics.state.frame_jacobian
    :summary:
    ```
````

### API

`````{py:class} KinematicState
:canonical: better_robot.kinematics.state.KinematicState

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState
```

````{py:attribute} q
:canonical: better_robot.kinematics.state.KinematicState.q
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState.q
```

````

````{py:attribute} joint_transform
:canonical: better_robot.kinematics.state.KinematicState.joint_transform
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState.joint_transform
```

````

````{py:attribute} frame_transform
:canonical: better_robot.kinematics.state.KinematicState.frame_transform
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState.frame_transform
```

````

````{py:property} joint_translation
:canonical: better_robot.kinematics.state.KinematicState.joint_translation
:type: torch.Tensor

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState.joint_translation
```

````

````{py:property} joint_rotation
:canonical: better_robot.kinematics.state.KinematicState.joint_rotation
:type: torch.Tensor

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState.joint_rotation
```

````

````{py:property} frame_translation
:canonical: better_robot.kinematics.state.KinematicState.frame_translation
:type: torch.Tensor

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState.frame_translation
```

````

````{py:property} frame_rotation
:canonical: better_robot.kinematics.state.KinematicState.frame_rotation
:type: torch.Tensor

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState.frame_rotation
```

````

````{py:property} joint_pose
:canonical: better_robot.kinematics.state.KinematicState.joint_pose
:type: torch.Tensor

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState.joint_pose
```

````

````{py:property} frame_pose
:canonical: better_robot.kinematics.state.KinematicState.frame_pose
:type: torch.Tensor

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState.frame_pose
```

````

````{py:method} frame_pose_of(frame_id: int) -> torch.Tensor
:canonical: better_robot.kinematics.state.KinematicState.frame_pose_of

```{autodoc2-docstring} better_robot.kinematics.state.KinematicState.frame_pose_of
```

````

`````

````{py:function} forward_kinematics_state(model: better_robot.data_model.model.Model, q: torch.Tensor) -> better_robot.kinematics.state.KinematicState
:canonical: better_robot.kinematics.state.forward_kinematics_state

```{autodoc2-docstring} better_robot.kinematics.state.forward_kinematics_state
```
````

````{py:function} frame_jacobian(model: better_robot.data_model.model.Model, state: better_robot.kinematics.state.KinematicState, frame: int | str, *, reference: typing.Literal[better_robot.kinematics.state.frame_jacobian.world, local, local_world_aligned] = 'local_world_aligned') -> torch.Tensor
:canonical: better_robot.kinematics.state.frame_jacobian

```{autodoc2-docstring} better_robot.kinematics.state.frame_jacobian
```
````
