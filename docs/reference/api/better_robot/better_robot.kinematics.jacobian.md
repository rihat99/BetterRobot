# {py:mod}`better_robot.kinematics.jacobian`

```{py:module} better_robot.kinematics.jacobian
```

```{autodoc2-docstring} better_robot.kinematics.jacobian
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointJacobiansResult <better_robot.kinematics.jacobian.JointJacobiansResult>`
  - ```{autodoc2-docstring} better_robot.kinematics.jacobian.JointJacobiansResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`joint_jacobians_raw <better_robot.kinematics.jacobian.joint_jacobians_raw>`
  - ```{autodoc2-docstring} better_robot.kinematics.jacobian.joint_jacobians_raw
    :summary:
    ```
* - {py:obj}`compute_joint_jacobians <better_robot.kinematics.jacobian.compute_joint_jacobians>`
  - ```{autodoc2-docstring} better_robot.kinematics.jacobian.compute_joint_jacobians
    :summary:
    ```
* - {py:obj}`get_joint_jacobian <better_robot.kinematics.jacobian.get_joint_jacobian>`
  - ```{autodoc2-docstring} better_robot.kinematics.jacobian.get_joint_jacobian
    :summary:
    ```
* - {py:obj}`get_frame_jacobian <better_robot.kinematics.jacobian.get_frame_jacobian>`
  - ```{autodoc2-docstring} better_robot.kinematics.jacobian.get_frame_jacobian
    :summary:
    ```
````

### API

`````{py:class} JointJacobiansResult
:canonical: better_robot.kinematics.jacobian.JointJacobiansResult

```{autodoc2-docstring} better_robot.kinematics.jacobian.JointJacobiansResult
```

````{py:attribute} joint_jacobians
:canonical: better_robot.kinematics.jacobian.JointJacobiansResult.joint_jacobians
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.kinematics.jacobian.JointJacobiansResult.joint_jacobians
```

````

`````

````{py:function} joint_jacobians_raw(structure: better_robot.data_model.model_structure.ModelStructure, q: torch.Tensor, joint_pose_world: torch.Tensor) -> better_robot.kinematics.jacobian.JointJacobiansResult
:canonical: better_robot.kinematics.jacobian.joint_jacobians_raw

```{autodoc2-docstring} better_robot.kinematics.jacobian.joint_jacobians_raw
```
````

````{py:function} compute_joint_jacobians(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> better_robot.data_model.data.Data
:canonical: better_robot.kinematics.jacobian.compute_joint_jacobians

```{autodoc2-docstring} better_robot.kinematics.jacobian.compute_joint_jacobians
```
````

````{py:function} get_joint_jacobian(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, joint_id: int, *, reference: better_robot.kinematics.jacobian._ReferenceFrame = 'world') -> torch.Tensor
:canonical: better_robot.kinematics.jacobian.get_joint_jacobian

```{autodoc2-docstring} better_robot.kinematics.jacobian.get_joint_jacobian
```
````

````{py:function} get_frame_jacobian(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, frame_id: int, *, reference: better_robot.kinematics.jacobian._ReferenceFrame = 'local_world_aligned') -> torch.Tensor
:canonical: better_robot.kinematics.jacobian.get_frame_jacobian

```{autodoc2-docstring} better_robot.kinematics.jacobian.get_frame_jacobian
```
````
