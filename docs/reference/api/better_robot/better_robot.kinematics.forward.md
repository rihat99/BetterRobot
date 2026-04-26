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
````

### API

````{py:function} forward_kinematics_raw(model: better_robot.data_model.model.Model, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.kinematics.forward.forward_kinematics_raw

```{autodoc2-docstring} better_robot.kinematics.forward.forward_kinematics_raw
```
````

````{py:function} forward_kinematics(model: better_robot.data_model.model.Model, q_or_data: torch.Tensor | better_robot.data_model.data.Data, *, compute_frames: bool = False, backend: Backend | None = None) -> better_robot.data_model.data.Data
:canonical: better_robot.kinematics.forward.forward_kinematics

```{autodoc2-docstring} better_robot.kinematics.forward.forward_kinematics
```
````

````{py:function} update_frame_placements(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> better_robot.data_model.data.Data
:canonical: better_robot.kinematics.forward.update_frame_placements

```{autodoc2-docstring} better_robot.kinematics.forward.update_frame_placements
```
````
