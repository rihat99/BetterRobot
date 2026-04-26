# {py:mod}`better_robot.dynamics.derivatives`

```{py:module} better_robot.dynamics.derivatives
```

```{autodoc2-docstring} better_robot.dynamics.derivatives
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_rnea_derivatives <better_robot.dynamics.derivatives.compute_rnea_derivatives>`
  - ```{autodoc2-docstring} better_robot.dynamics.derivatives.compute_rnea_derivatives
    :summary:
    ```
* - {py:obj}`compute_aba_derivatives <better_robot.dynamics.derivatives.compute_aba_derivatives>`
  - ```{autodoc2-docstring} better_robot.dynamics.derivatives.compute_aba_derivatives
    :summary:
    ```
* - {py:obj}`compute_crba_derivatives <better_robot.dynamics.derivatives.compute_crba_derivatives>`
  - ```{autodoc2-docstring} better_robot.dynamics.derivatives.compute_crba_derivatives
    :summary:
    ```
* - {py:obj}`compute_centroidal_dynamics_derivatives <better_robot.dynamics.derivatives.compute_centroidal_dynamics_derivatives>`
  - ```{autodoc2-docstring} better_robot.dynamics.derivatives.compute_centroidal_dynamics_derivatives
    :summary:
    ```
````

### API

````{py:function} compute_rnea_derivatives(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor, fext: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: better_robot.dynamics.derivatives.compute_rnea_derivatives

```{autodoc2-docstring} better_robot.dynamics.derivatives.compute_rnea_derivatives
```
````

````{py:function} compute_aba_derivatives(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, fext: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
:canonical: better_robot.dynamics.derivatives.compute_aba_derivatives

```{autodoc2-docstring} better_robot.dynamics.derivatives.compute_aba_derivatives
```
````

````{py:function} compute_crba_derivatives(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.dynamics.derivatives.compute_crba_derivatives

```{autodoc2-docstring} better_robot.dynamics.derivatives.compute_crba_derivatives
```
````

````{py:function} compute_centroidal_dynamics_derivatives(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor)
:canonical: better_robot.dynamics.derivatives.compute_centroidal_dynamics_derivatives

```{autodoc2-docstring} better_robot.dynamics.derivatives.compute_centroidal_dynamics_derivatives
```
````
