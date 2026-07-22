# {py:mod}`better_robot.kinematics.forward`

```{py:module} better_robot.kinematics.forward
```

```{autodoc2-docstring} better_robot.kinematics.forward
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FKResult <better_robot.kinematics.forward.FKResult>`
  - ```{autodoc2-docstring} better_robot.kinematics.forward.FKResult
    :summary:
    ```
* - {py:obj}`FramePlacementsResult <better_robot.kinematics.forward.FramePlacementsResult>`
  - ```{autodoc2-docstring} better_robot.kinematics.forward.FramePlacementsResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`forward_kinematics_raw <better_robot.kinematics.forward.forward_kinematics_raw>`
  - ```{autodoc2-docstring} better_robot.kinematics.forward.forward_kinematics_raw
    :summary:
    ```
* - {py:obj}`forward_kinematics <better_robot.kinematics.forward.forward_kinematics>`
  - ```{autodoc2-docstring} better_robot.kinematics.forward.forward_kinematics
    :summary:
    ```
* - {py:obj}`update_frame_placements <better_robot.kinematics.forward.update_frame_placements>`
  - ```{autodoc2-docstring} better_robot.kinematics.forward.update_frame_placements
    :summary:
    ```
* - {py:obj}`frame_placements_raw <better_robot.kinematics.forward.frame_placements_raw>`
  - ```{autodoc2-docstring} better_robot.kinematics.forward.frame_placements_raw
    :summary:
    ```
````

### API

`````{py:class} FKResult
:canonical: better_robot.kinematics.forward.FKResult

```{autodoc2-docstring} better_robot.kinematics.forward.FKResult
```

````{py:attribute} joint_pose_world
:canonical: better_robot.kinematics.forward.FKResult.joint_pose_world
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.kinematics.forward.FKResult.joint_pose_world
```

````

````{py:attribute} joint_pose_local
:canonical: better_robot.kinematics.forward.FKResult.joint_pose_local
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.kinematics.forward.FKResult.joint_pose_local
```

````

`````

`````{py:class} FramePlacementsResult
:canonical: better_robot.kinematics.forward.FramePlacementsResult

```{autodoc2-docstring} better_robot.kinematics.forward.FramePlacementsResult
```

````{py:attribute} frame_pose_world
:canonical: better_robot.kinematics.forward.FramePlacementsResult.frame_pose_world
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.kinematics.forward.FramePlacementsResult.frame_pose_world
```

````

`````

````{py:function} forward_kinematics_raw(structure: better_robot.data_model.model_structure.ModelStructure, values: better_robot.data_model.model_values.ModelValues, q: torch.Tensor) -> better_robot.kinematics.forward.FKResult
:canonical: better_robot.kinematics.forward.forward_kinematics_raw

```{autodoc2-docstring} better_robot.kinematics.forward.forward_kinematics_raw
```
````

````{py:function} forward_kinematics(model: better_robot.data_model.model.Model, q_or_data: torch.Tensor | better_robot.data_model.data.Data, *, compute_frames: bool = False, check_quaternion_norm: bool = False, use_warp: bool = False, use_compile: bool = False) -> better_robot.data_model.data.Data
:canonical: better_robot.kinematics.forward.forward_kinematics

```{autodoc2-docstring} better_robot.kinematics.forward.forward_kinematics
```
````

````{py:function} update_frame_placements(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> better_robot.data_model.data.Data
:canonical: better_robot.kinematics.forward.update_frame_placements

```{autodoc2-docstring} better_robot.kinematics.forward.update_frame_placements
```
````

````{py:function} frame_placements_raw(structure: better_robot.data_model.model_structure.ModelStructure, values: better_robot.data_model.model_values.ModelValues, joint_pose_world: torch.Tensor) -> better_robot.kinematics.forward.FramePlacementsResult
:canonical: better_robot.kinematics.forward.frame_placements_raw

```{autodoc2-docstring} better_robot.kinematics.forward.frame_placements_raw
```
````
