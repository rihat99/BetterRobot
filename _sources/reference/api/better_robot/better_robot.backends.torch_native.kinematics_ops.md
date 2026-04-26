# {py:mod}`better_robot.backends.torch_native.kinematics_ops`

```{py:module} better_robot.backends.torch_native.kinematics_ops
```

```{autodoc2-docstring} better_robot.backends.torch_native.kinematics_ops
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TorchNativeKinematicsOps <better_robot.backends.torch_native.kinematics_ops.TorchNativeKinematicsOps>`
  - ```{autodoc2-docstring} better_robot.backends.torch_native.kinematics_ops.TorchNativeKinematicsOps
    :summary:
    ```
````

### API

`````{py:class} TorchNativeKinematicsOps
:canonical: better_robot.backends.torch_native.kinematics_ops.TorchNativeKinematicsOps

```{autodoc2-docstring} better_robot.backends.torch_native.kinematics_ops.TorchNativeKinematicsOps
```

````{py:method} forward_kinematics(model: better_robot.data_model.model.Model, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.backends.torch_native.kinematics_ops.TorchNativeKinematicsOps.forward_kinematics

```{autodoc2-docstring} better_robot.backends.torch_native.kinematics_ops.TorchNativeKinematicsOps.forward_kinematics
```

````

````{py:method} compute_joint_jacobians(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> torch.Tensor
:canonical: better_robot.backends.torch_native.kinematics_ops.TorchNativeKinematicsOps.compute_joint_jacobians

```{autodoc2-docstring} better_robot.backends.torch_native.kinematics_ops.TorchNativeKinematicsOps.compute_joint_jacobians
```

````

`````
