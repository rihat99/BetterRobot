# {py:mod}`better_robot.dynamics.rnea`

```{py:module} better_robot.dynamics.rnea
```

```{autodoc2-docstring} better_robot.dynamics.rnea
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RNEAResult <better_robot.dynamics.rnea.RNEAResult>`
  - ```{autodoc2-docstring} better_robot.dynamics.rnea.RNEAResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`rnea_raw <better_robot.dynamics.rnea.rnea_raw>`
  - ```{autodoc2-docstring} better_robot.dynamics.rnea.rnea_raw
    :summary:
    ```
* - {py:obj}`rnea <better_robot.dynamics.rnea.rnea>`
  - ```{autodoc2-docstring} better_robot.dynamics.rnea.rnea
    :summary:
    ```
* - {py:obj}`bias_forces <better_robot.dynamics.rnea.bias_forces>`
  - ```{autodoc2-docstring} better_robot.dynamics.rnea.bias_forces
    :summary:
    ```
* - {py:obj}`compute_generalized_gravity <better_robot.dynamics.rnea.compute_generalized_gravity>`
  - ```{autodoc2-docstring} better_robot.dynamics.rnea.compute_generalized_gravity
    :summary:
    ```
````

### API

`````{py:class} RNEAResult
:canonical: better_robot.dynamics.rnea.RNEAResult

```{autodoc2-docstring} better_robot.dynamics.rnea.RNEAResult
```

````{py:attribute} tau
:canonical: better_robot.dynamics.rnea.RNEAResult.tau
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.rnea.RNEAResult.tau
```

````

````{py:attribute} joint_pose_world
:canonical: better_robot.dynamics.rnea.RNEAResult.joint_pose_world
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.rnea.RNEAResult.joint_pose_world
```

````

````{py:attribute} joint_pose_local
:canonical: better_robot.dynamics.rnea.RNEAResult.joint_pose_local
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.rnea.RNEAResult.joint_pose_local
```

````

````{py:attribute} joint_velocity_local
:canonical: better_robot.dynamics.rnea.RNEAResult.joint_velocity_local
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.rnea.RNEAResult.joint_velocity_local
```

````

````{py:attribute} joint_acceleration_local
:canonical: better_robot.dynamics.rnea.RNEAResult.joint_acceleration_local
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.rnea.RNEAResult.joint_acceleration_local
```

````

````{py:attribute} joint_forces
:canonical: better_robot.dynamics.rnea.RNEAResult.joint_forces
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.rnea.RNEAResult.joint_forces
```

````

`````

````{py:function} rnea_raw(structure: better_robot.data_model.model_structure.ModelStructure, values: better_robot.data_model.model_values.ModelValues, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor, *, fext: torch.Tensor | None = None) -> better_robot.dynamics.rnea.RNEAResult
:canonical: better_robot.dynamics.rnea.rnea_raw

```{autodoc2-docstring} better_robot.dynamics.rnea.rnea_raw
```
````

````{py:function} rnea(model: better_robot.data_model.model.Model, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor, *, fext: torch.Tensor | None = None, data: better_robot.data_model.data.Data | None = None, use_warp: bool = False) -> torch.Tensor
:canonical: better_robot.dynamics.rnea.rnea

```{autodoc2-docstring} better_robot.dynamics.rnea.rnea
```
````

````{py:function} bias_forces(model: better_robot.data_model.model.Model, q: torch.Tensor, v: torch.Tensor, *, data: better_robot.data_model.data.Data | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.rnea.bias_forces

```{autodoc2-docstring} better_robot.dynamics.rnea.bias_forces
```
````

````{py:function} compute_generalized_gravity(model: better_robot.data_model.model.Model, q: torch.Tensor, *, data: better_robot.data_model.data.Data | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.rnea.compute_generalized_gravity

```{autodoc2-docstring} better_robot.dynamics.rnea.compute_generalized_gravity
```
````
