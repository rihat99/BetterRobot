# {py:mod}`better_robot.dynamics.integrators`

```{py:module} better_robot.dynamics.integrators
```

```{autodoc2-docstring} better_robot.dynamics.integrators
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`integrate_q <better_robot.dynamics.integrators.integrate_q>`
  - ```{autodoc2-docstring} better_robot.dynamics.integrators.integrate_q
    :summary:
    ```
* - {py:obj}`semi_implicit_euler <better_robot.dynamics.integrators.semi_implicit_euler>`
  - ```{autodoc2-docstring} better_robot.dynamics.integrators.semi_implicit_euler
    :summary:
    ```
* - {py:obj}`symplectic_euler <better_robot.dynamics.integrators.symplectic_euler>`
  - ```{autodoc2-docstring} better_robot.dynamics.integrators.symplectic_euler
    :summary:
    ```
* - {py:obj}`rk4 <better_robot.dynamics.integrators.rk4>`
  - ```{autodoc2-docstring} better_robot.dynamics.integrators.rk4
    :summary:
    ```
````

### API

````{py:function} integrate_q(model: better_robot.data_model.model.Model, q: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor
:canonical: better_robot.dynamics.integrators.integrate_q

```{autodoc2-docstring} better_robot.dynamics.integrators.integrate_q
```
````

````{py:function} semi_implicit_euler(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, dt: float, *, fext: torch.Tensor | None = None)
:canonical: better_robot.dynamics.integrators.semi_implicit_euler

```{autodoc2-docstring} better_robot.dynamics.integrators.semi_implicit_euler
```
````

````{py:function} symplectic_euler(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, dt: float, *, fext: torch.Tensor | None = None)
:canonical: better_robot.dynamics.integrators.symplectic_euler

```{autodoc2-docstring} better_robot.dynamics.integrators.symplectic_euler
```
````

````{py:function} rk4(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, dt: float, *, fext: torch.Tensor | None = None)
:canonical: better_robot.dynamics.integrators.rk4

```{autodoc2-docstring} better_robot.dynamics.integrators.rk4
```
````
