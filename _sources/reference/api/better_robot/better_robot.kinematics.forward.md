# {py:mod}`better_robot.kinematics.forward`

```{py:module} better_robot.kinematics.forward
```

```{autodoc2-docstring} better_robot.kinematics.forward
:allowtitles:
```

## Module Contents

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

````{py:function} forward_kinematics_raw(structure: better_robot.data_model.model_structure.ModelStructure, values: better_robot.data_model.model_values.ModelValues, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.kinematics.forward.forward_kinematics_raw

```{autodoc2-docstring} better_robot.kinematics.forward.forward_kinematics_raw
```
````

````{py:function} forward_kinematics(model: better_robot.data_model.model.Model, q_or_data: torch.Tensor | better_robot.data_model.data.Data, *, compute_frames: bool = False, check_quaternion_norm: bool = False, use_warp: bool = False) -> better_robot.data_model.data.Data
:canonical: better_robot.kinematics.forward.forward_kinematics

```{autodoc2-docstring} better_robot.kinematics.forward.forward_kinematics
```
````

````{py:function} update_frame_placements(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> better_robot.data_model.data.Data
:canonical: better_robot.kinematics.forward.update_frame_placements

```{autodoc2-docstring} better_robot.kinematics.forward.update_frame_placements
```
````

````{py:function} frame_placements_raw(structure: better_robot.data_model.model_structure.ModelStructure, values: better_robot.data_model.model_values.ModelValues, joint_pose_world: torch.Tensor) -> torch.Tensor
:canonical: better_robot.kinematics.forward.frame_placements_raw

```{autodoc2-docstring} better_robot.kinematics.forward.frame_placements_raw
```
````
