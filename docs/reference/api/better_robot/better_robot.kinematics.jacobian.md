# {py:mod}`better_robot.kinematics.jacobian`

```{py:module} better_robot.kinematics.jacobian
```

```{autodoc2-docstring} better_robot.kinematics.jacobian
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

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
* - {py:obj}`residual_jacobian <better_robot.kinematics.jacobian.residual_jacobian>`
  - ```{autodoc2-docstring} better_robot.kinematics.jacobian.residual_jacobian
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ReferenceFrame <better_robot.kinematics.jacobian.ReferenceFrame>`
  - ```{autodoc2-docstring} better_robot.kinematics.jacobian.ReferenceFrame
    :summary:
    ```
````

### API

````{py:data} ReferenceFrame
:canonical: better_robot.kinematics.jacobian.ReferenceFrame
:value: >
   None

```{autodoc2-docstring} better_robot.kinematics.jacobian.ReferenceFrame
```

````

````{py:function} compute_joint_jacobians(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, *, backend: Backend | None = None) -> better_robot.data_model.data.Data
:canonical: better_robot.kinematics.jacobian.compute_joint_jacobians

```{autodoc2-docstring} better_robot.kinematics.jacobian.compute_joint_jacobians
```
````

````{py:function} get_joint_jacobian(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, joint_id: int, *, reference: better_robot.kinematics.jacobian.ReferenceFrame = 'world') -> torch.Tensor
:canonical: better_robot.kinematics.jacobian.get_joint_jacobian

```{autodoc2-docstring} better_robot.kinematics.jacobian.get_joint_jacobian
```
````

````{py:function} get_frame_jacobian(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, frame_id: int, *, reference: better_robot.kinematics.jacobian.ReferenceFrame = 'local_world_aligned') -> torch.Tensor
:canonical: better_robot.kinematics.jacobian.get_frame_jacobian

```{autodoc2-docstring} better_robot.kinematics.jacobian.get_frame_jacobian
```
````

````{py:function} residual_jacobian(residual: better_robot.residuals.base.Residual, state: better_robot.residuals.base.ResidualState, *, strategy: better_robot.kinematics.jacobian_strategy.JacobianStrategy = JacobianStrategy.AUTO) -> torch.Tensor
:canonical: better_robot.kinematics.jacobian.residual_jacobian

```{autodoc2-docstring} better_robot.kinematics.jacobian.residual_jacobian
```
````
