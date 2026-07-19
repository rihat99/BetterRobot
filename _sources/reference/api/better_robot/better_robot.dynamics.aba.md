# {py:mod}`better_robot.dynamics.aba`

```{py:module} better_robot.dynamics.aba
```

```{autodoc2-docstring} better_robot.dynamics.aba
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ABAResult <better_robot.dynamics.aba.ABAResult>`
  - ```{autodoc2-docstring} better_robot.dynamics.aba.ABAResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`aba_raw <better_robot.dynamics.aba.aba_raw>`
  - ```{autodoc2-docstring} better_robot.dynamics.aba.aba_raw
    :summary:
    ```
* - {py:obj}`aba <better_robot.dynamics.aba.aba>`
  - ```{autodoc2-docstring} better_robot.dynamics.aba.aba
    :summary:
    ```
````

### API

`````{py:class} ABAResult
:canonical: better_robot.dynamics.aba.ABAResult

```{autodoc2-docstring} better_robot.dynamics.aba.ABAResult
```

````{py:attribute} ddq
:canonical: better_robot.dynamics.aba.ABAResult.ddq
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.aba.ABAResult.ddq
```

````

````{py:attribute} joint_pose_world
:canonical: better_robot.dynamics.aba.ABAResult.joint_pose_world
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.aba.ABAResult.joint_pose_world
```

````

````{py:attribute} joint_pose_local
:canonical: better_robot.dynamics.aba.ABAResult.joint_pose_local
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.aba.ABAResult.joint_pose_local
```

````

````{py:attribute} joint_velocity_local
:canonical: better_robot.dynamics.aba.ABAResult.joint_velocity_local
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.aba.ABAResult.joint_velocity_local
```

````

````{py:attribute} joint_acceleration_local
:canonical: better_robot.dynamics.aba.ABAResult.joint_acceleration_local
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.aba.ABAResult.joint_acceleration_local
```

````

`````

````{py:function} aba_raw(structure: better_robot.data_model.model_structure.ModelStructure, values: better_robot.data_model.model_values.ModelValues, q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, *, fext: torch.Tensor | None = None) -> better_robot.dynamics.aba.ABAResult
:canonical: better_robot.dynamics.aba.aba_raw

```{autodoc2-docstring} better_robot.dynamics.aba.aba_raw
```
````

````{py:function} aba(model: better_robot.data_model.model.Model, q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, *, fext: torch.Tensor | None = None, data: better_robot.data_model.data.Data | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.aba.aba

```{autodoc2-docstring} better_robot.dynamics.aba.aba
```
````
