# {py:mod}`better_robot.backends.torch_native.dynamics_ops`

```{py:module} better_robot.backends.torch_native.dynamics_ops
```

```{autodoc2-docstring} better_robot.backends.torch_native.dynamics_ops
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TorchNativeDynamicsOps <better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps>`
  - ```{autodoc2-docstring} better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps
    :summary:
    ```
````

### API

`````{py:class} TorchNativeDynamicsOps
:canonical: better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps

```{autodoc2-docstring} better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps
```

````{py:method} rnea(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps.rnea

```{autodoc2-docstring} better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps.rnea
```

````

````{py:method} aba(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps.aba

```{autodoc2-docstring} better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps.aba
```

````

````{py:method} crba(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps.crba

```{autodoc2-docstring} better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps.crba
```

````

````{py:method} center_of_mass(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor | None = None, a: torch.Tensor | None = None) -> torch.Tensor
:canonical: better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps.center_of_mass

```{autodoc2-docstring} better_robot.backends.torch_native.dynamics_ops.TorchNativeDynamicsOps.center_of_mass
```

````

`````
