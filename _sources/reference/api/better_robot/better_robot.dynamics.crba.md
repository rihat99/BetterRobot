# {py:mod}`better_robot.dynamics.crba`

```{py:module} better_robot.dynamics.crba
```

```{autodoc2-docstring} better_robot.dynamics.crba
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CRBAResult <better_robot.dynamics.crba.CRBAResult>`
  - ```{autodoc2-docstring} better_robot.dynamics.crba.CRBAResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`crba_raw <better_robot.dynamics.crba.crba_raw>`
  - ```{autodoc2-docstring} better_robot.dynamics.crba.crba_raw
    :summary:
    ```
* - {py:obj}`crba <better_robot.dynamics.crba.crba>`
  - ```{autodoc2-docstring} better_robot.dynamics.crba.crba
    :summary:
    ```
````

### API

`````{py:class} CRBAResult
:canonical: better_robot.dynamics.crba.CRBAResult

```{autodoc2-docstring} better_robot.dynamics.crba.CRBAResult
```

````{py:attribute} mass_matrix
:canonical: better_robot.dynamics.crba.CRBAResult.mass_matrix
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.crba.CRBAResult.mass_matrix
```

````

````{py:attribute} joint_pose_world
:canonical: better_robot.dynamics.crba.CRBAResult.joint_pose_world
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.crba.CRBAResult.joint_pose_world
```

````

````{py:attribute} joint_pose_local
:canonical: better_robot.dynamics.crba.CRBAResult.joint_pose_local
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.crba.CRBAResult.joint_pose_local
```

````

`````

````{py:function} crba_raw(structure: better_robot.data_model.model_structure.ModelStructure, values: better_robot.data_model.model_values.ModelValues, q: torch.Tensor) -> better_robot.dynamics.crba.CRBAResult
:canonical: better_robot.dynamics.crba.crba_raw

```{autodoc2-docstring} better_robot.dynamics.crba.crba_raw
```
````

````{py:function} crba(model: better_robot.data_model.model.Model, q: torch.Tensor, *, data: better_robot.data_model.data.Data | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.crba.crba

```{autodoc2-docstring} better_robot.dynamics.crba.crba
```
````
