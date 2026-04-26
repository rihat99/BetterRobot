# {py:mod}`better_robot.dynamics.centroidal`

```{py:module} better_robot.dynamics.centroidal
```

```{autodoc2-docstring} better_robot.dynamics.centroidal
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

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

````{py:function} center_of_mass(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor | None = None, a: torch.Tensor | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.centroidal.center_of_mass

```{autodoc2-docstring} better_robot.dynamics.centroidal.center_of_mass
```
````

````{py:function} compute_centroidal_map(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.dynamics.centroidal.compute_centroidal_map

```{autodoc2-docstring} better_robot.dynamics.centroidal.compute_centroidal_map
```
````

````{py:function} compute_centroidal_momentum(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor
:canonical: better_robot.dynamics.centroidal.compute_centroidal_momentum

```{autodoc2-docstring} better_robot.dynamics.centroidal.compute_centroidal_momentum
```
````

````{py:function} ccrba(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.dynamics.centroidal.ccrba

```{autodoc2-docstring} better_robot.dynamics.centroidal.ccrba
```
````
