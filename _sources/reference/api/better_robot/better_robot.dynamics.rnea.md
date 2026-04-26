# {py:mod}`better_robot.dynamics.rnea`

```{py:module} better_robot.dynamics.rnea
```

```{autodoc2-docstring} better_robot.dynamics.rnea
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

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
* - {py:obj}`compute_coriolis_matrix <better_robot.dynamics.rnea.compute_coriolis_matrix>`
  - ```{autodoc2-docstring} better_robot.dynamics.rnea.compute_coriolis_matrix
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`nle <better_robot.dynamics.rnea.nle>`
  - ```{autodoc2-docstring} better_robot.dynamics.rnea.nle
    :summary:
    ```
````

### API

````{py:function} rnea(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor, *, fext: torch.Tensor | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.rnea.rnea

```{autodoc2-docstring} better_robot.dynamics.rnea.rnea
```
````

````{py:function} bias_forces(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor
:canonical: better_robot.dynamics.rnea.bias_forces

```{autodoc2-docstring} better_robot.dynamics.rnea.bias_forces
```
````

````{py:data} nle
:canonical: better_robot.dynamics.rnea.nle
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.rnea.nle
```

````

````{py:function} compute_generalized_gravity(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.dynamics.rnea.compute_generalized_gravity

```{autodoc2-docstring} better_robot.dynamics.rnea.compute_generalized_gravity
```
````

````{py:function} compute_coriolis_matrix(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor
:canonical: better_robot.dynamics.rnea.compute_coriolis_matrix

```{autodoc2-docstring} better_robot.dynamics.rnea.compute_coriolis_matrix
```
````
