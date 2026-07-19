# {py:mod}`better_robot.dynamics.centroidal`

```{py:module} better_robot.dynamics.centroidal
```

```{autodoc2-docstring} better_robot.dynamics.centroidal
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CentroidalResult <better_robot.dynamics.centroidal.CentroidalResult>`
  - ```{autodoc2-docstring} better_robot.dynamics.centroidal.CentroidalResult
    :summary:
    ```
* - {py:obj}`CCRBAResult <better_robot.dynamics.centroidal.CCRBAResult>`
  - ```{autodoc2-docstring} better_robot.dynamics.centroidal.CCRBAResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ccrba_raw <better_robot.dynamics.centroidal.ccrba_raw>`
  - ```{autodoc2-docstring} better_robot.dynamics.centroidal.ccrba_raw
    :summary:
    ```
* - {py:obj}`center_of_mass <better_robot.dynamics.centroidal.center_of_mass>`
  - ```{autodoc2-docstring} better_robot.dynamics.centroidal.center_of_mass
    :summary:
    ```
* - {py:obj}`compute_centroidal_map <better_robot.dynamics.centroidal.compute_centroidal_map>`
  - ```{autodoc2-docstring} better_robot.dynamics.centroidal.compute_centroidal_map
    :summary:
    ```
* - {py:obj}`compute_centroidal_momentum <better_robot.dynamics.centroidal.compute_centroidal_momentum>`
  - ```{autodoc2-docstring} better_robot.dynamics.centroidal.compute_centroidal_momentum
    :summary:
    ```
* - {py:obj}`ccrba <better_robot.dynamics.centroidal.ccrba>`
  - ```{autodoc2-docstring} better_robot.dynamics.centroidal.ccrba
    :summary:
    ```
````

### API

`````{py:class} CentroidalResult
:canonical: better_robot.dynamics.centroidal.CentroidalResult

```{autodoc2-docstring} better_robot.dynamics.centroidal.CentroidalResult
```

````{py:attribute} centroidal_map
:canonical: better_robot.dynamics.centroidal.CentroidalResult.centroidal_map
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.centroidal.CentroidalResult.centroidal_map
```

````

````{py:attribute} momentum
:canonical: better_robot.dynamics.centroidal.CentroidalResult.momentum
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.centroidal.CentroidalResult.momentum
```

````

````{py:attribute} total_mass
:canonical: better_robot.dynamics.centroidal.CentroidalResult.total_mass
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.centroidal.CentroidalResult.total_mass
```

````

````{py:attribute} com_position
:canonical: better_robot.dynamics.centroidal.CentroidalResult.com_position
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.centroidal.CentroidalResult.com_position
```

````

````{py:attribute} joint_pose_world
:canonical: better_robot.dynamics.centroidal.CentroidalResult.joint_pose_world
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.centroidal.CentroidalResult.joint_pose_world
```

````

````{py:attribute} joint_pose_local
:canonical: better_robot.dynamics.centroidal.CentroidalResult.joint_pose_local
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.centroidal.CentroidalResult.joint_pose_local
```

````

`````

`````{py:class} CCRBAResult
:canonical: better_robot.dynamics.centroidal.CCRBAResult

```{autodoc2-docstring} better_robot.dynamics.centroidal.CCRBAResult
```

````{py:attribute} centroidal_map
:canonical: better_robot.dynamics.centroidal.CCRBAResult.centroidal_map
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.centroidal.CCRBAResult.centroidal_map
```

````

````{py:attribute} momentum
:canonical: better_robot.dynamics.centroidal.CCRBAResult.momentum
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.centroidal.CCRBAResult.momentum
```

````

`````

````{py:function} ccrba_raw(structure: better_robot.data_model.model_structure.ModelStructure, values: better_robot.data_model.model_values.ModelValues, q: torch.Tensor, v: torch.Tensor | None = None) -> better_robot.dynamics.centroidal.CentroidalResult
:canonical: better_robot.dynamics.centroidal.ccrba_raw

```{autodoc2-docstring} better_robot.dynamics.centroidal.ccrba_raw
```
````

````{py:function} center_of_mass(model: better_robot.data_model.model.Model, q: torch.Tensor, v: torch.Tensor | None = None, a: torch.Tensor | None = None, *, data: better_robot.data_model.data.Data | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.centroidal.center_of_mass

```{autodoc2-docstring} better_robot.dynamics.centroidal.center_of_mass
```
````

````{py:function} compute_centroidal_map(model: better_robot.data_model.model.Model, q: torch.Tensor, *, data: better_robot.data_model.data.Data | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.centroidal.compute_centroidal_map

```{autodoc2-docstring} better_robot.dynamics.centroidal.compute_centroidal_map
```
````

````{py:function} compute_centroidal_momentum(model: better_robot.data_model.model.Model, q: torch.Tensor, v: torch.Tensor, *, data: better_robot.data_model.data.Data | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.centroidal.compute_centroidal_momentum

```{autodoc2-docstring} better_robot.dynamics.centroidal.compute_centroidal_momentum
```
````

````{py:function} ccrba(model: better_robot.data_model.model.Model, q: torch.Tensor, v: torch.Tensor, *, data: better_robot.data_model.data.Data | None = None) -> better_robot.dynamics.centroidal.CCRBAResult
:canonical: better_robot.dynamics.centroidal.ccrba

```{autodoc2-docstring} better_robot.dynamics.centroidal.ccrba
```
````
