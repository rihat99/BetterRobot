# {py:mod}`better_robot.dynamics.state`

```{py:module} better_robot.dynamics.state
```

```{autodoc2-docstring} better_robot.dynamics.state
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`inverse_dynamics <better_robot.dynamics.state.inverse_dynamics>`
  - ```{autodoc2-docstring} better_robot.dynamics.state.inverse_dynamics
    :summary:
    ```
* - {py:obj}`mass_matrix <better_robot.dynamics.state.mass_matrix>`
  - ```{autodoc2-docstring} better_robot.dynamics.state.mass_matrix
    :summary:
    ```
* - {py:obj}`forward_dynamics <better_robot.dynamics.state.forward_dynamics>`
  - ```{autodoc2-docstring} better_robot.dynamics.state.forward_dynamics
    :summary:
    ```
* - {py:obj}`center_of_mass <better_robot.dynamics.state.center_of_mass>`
  - ```{autodoc2-docstring} better_robot.dynamics.state.center_of_mass
    :summary:
    ```
* - {py:obj}`centroidal_map <better_robot.dynamics.state.centroidal_map>`
  - ```{autodoc2-docstring} better_robot.dynamics.state.centroidal_map
    :summary:
    ```
````

### API

````{py:function} inverse_dynamics(model: better_robot.data_model.model.Model, state: better_robot.kinematics.state.KinematicState, v: torch.Tensor, a: torch.Tensor, *, fext: torch.Tensor | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.state.inverse_dynamics

```{autodoc2-docstring} better_robot.dynamics.state.inverse_dynamics
```
````

````{py:function} mass_matrix(model: better_robot.data_model.model.Model, state: better_robot.kinematics.state.KinematicState) -> torch.Tensor
:canonical: better_robot.dynamics.state.mass_matrix

```{autodoc2-docstring} better_robot.dynamics.state.mass_matrix
```
````

````{py:function} forward_dynamics(model: better_robot.data_model.model.Model, state: better_robot.kinematics.state.KinematicState, v: torch.Tensor, tau: torch.Tensor, *, fext: torch.Tensor | None = None) -> torch.Tensor
:canonical: better_robot.dynamics.state.forward_dynamics

```{autodoc2-docstring} better_robot.dynamics.state.forward_dynamics
```
````

````{py:function} center_of_mass(model: better_robot.data_model.model.Model, state: better_robot.kinematics.state.KinematicState) -> torch.Tensor
:canonical: better_robot.dynamics.state.center_of_mass

```{autodoc2-docstring} better_robot.dynamics.state.center_of_mass
```
````

````{py:function} centroidal_map(model: better_robot.data_model.model.Model, state: better_robot.kinematics.state.KinematicState) -> torch.Tensor
:canonical: better_robot.dynamics.state.centroidal_map

```{autodoc2-docstring} better_robot.dynamics.state.centroidal_map
```
````
