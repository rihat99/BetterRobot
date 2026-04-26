# {py:mod}`better_robot.dynamics.action.differential`

```{py:module} better_robot.dynamics.action.differential
```

```{autodoc2-docstring} better_robot.dynamics.action.differential
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DifferentialActionModel <better_robot.dynamics.action.differential.DifferentialActionModel>`
  - ```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModel
    :summary:
    ```
* - {py:obj}`DifferentialActionModelFreeFwd <better_robot.dynamics.action.differential.DifferentialActionModelFreeFwd>`
  - ```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModelFreeFwd
    :summary:
    ```
````

### API

`````{py:class} DifferentialActionModel
:canonical: better_robot.dynamics.action.differential.DifferentialActionModel

```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModel
```

````{py:attribute} model
:canonical: better_robot.dynamics.action.differential.DifferentialActionModel.model
:type: better_robot.data_model.model.Model
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModel.model
```

````

````{py:attribute} state
:canonical: better_robot.dynamics.action.differential.DifferentialActionModel.state
:type: better_robot.dynamics.state_manifold.StateMultibody
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModel.state
```

````

````{py:attribute} cost
:canonical: better_robot.dynamics.action.differential.DifferentialActionModel.cost
:type: typing.Optional[typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModel.cost
```

````

````{py:attribute} nu
:canonical: better_robot.dynamics.action.differential.DifferentialActionModel.nu
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModel.nu
```

````

````{py:method} create_data() -> better_robot.dynamics.action.action.ActionData
:canonical: better_robot.dynamics.action.differential.DifferentialActionModel.create_data

```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModel.create_data
```

````

````{py:method} forward_dynamics(q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor
:canonical: better_robot.dynamics.action.differential.DifferentialActionModel.forward_dynamics
:abstractmethod:

```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModel.forward_dynamics
```

````

````{py:method} calc(data: better_robot.dynamics.action.action.ActionData, x: torch.Tensor, u: torch.Tensor) -> None
:canonical: better_robot.dynamics.action.differential.DifferentialActionModel.calc

```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModel.calc
```

````

````{py:method} calc_diff(data: better_robot.dynamics.action.action.ActionData, x: torch.Tensor, u: torch.Tensor) -> None
:canonical: better_robot.dynamics.action.differential.DifferentialActionModel.calc_diff

```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModel.calc_diff
```

````

`````

`````{py:class} DifferentialActionModelFreeFwd
:canonical: better_robot.dynamics.action.differential.DifferentialActionModelFreeFwd

Bases: {py:obj}`better_robot.dynamics.action.differential.DifferentialActionModel`

```{autodoc2-docstring} better_robot.dynamics.action.differential.DifferentialActionModelFreeFwd
```

````{py:method} forward_dynamics(q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor
:canonical: better_robot.dynamics.action.differential.DifferentialActionModelFreeFwd.forward_dynamics

````

`````
