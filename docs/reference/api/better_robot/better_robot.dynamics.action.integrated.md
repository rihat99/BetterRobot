# {py:mod}`better_robot.dynamics.action.integrated`

```{py:module} better_robot.dynamics.action.integrated
```

```{autodoc2-docstring} better_robot.dynamics.action.integrated
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IntegratedActionModelEuler <better_robot.dynamics.action.integrated.IntegratedActionModelEuler>`
  -
* - {py:obj}`IntegratedActionModelRK4 <better_robot.dynamics.action.integrated.IntegratedActionModelRK4>`
  -
````

### API

`````{py:class} IntegratedActionModelEuler
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelEuler

Bases: {py:obj}`better_robot.dynamics.action.action.ActionModel`

````{py:attribute} differential
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelEuler.differential
:type: better_robot.dynamics.action.differential.DifferentialActionModel
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelEuler.differential
```

````

````{py:attribute} dt
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelEuler.dt
:type: float
:value: >
   0.01

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelEuler.dt
```

````

````{py:attribute} with_cost_residual
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelEuler.with_cost_residual
:type: bool
:value: >
   True

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelEuler.with_cost_residual
```

````

````{py:method} create_data() -> better_robot.dynamics.action.action.ActionData
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelEuler.create_data

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelEuler.create_data
```

````

````{py:method} calc(data: better_robot.dynamics.action.action.ActionData, x: torch.Tensor, u: torch.Tensor) -> None
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelEuler.calc

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelEuler.calc
```

````

````{py:method} calc_diff(data: better_robot.dynamics.action.action.ActionData, x: torch.Tensor, u: torch.Tensor) -> None
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelEuler.calc_diff

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelEuler.calc_diff
```

````

`````

`````{py:class} IntegratedActionModelRK4
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelRK4

Bases: {py:obj}`better_robot.dynamics.action.action.ActionModel`

````{py:attribute} differential
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelRK4.differential
:type: better_robot.dynamics.action.differential.DifferentialActionModel
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelRK4.differential
```

````

````{py:attribute} dt
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelRK4.dt
:type: float
:value: >
   0.01

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelRK4.dt
```

````

````{py:attribute} with_cost_residual
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelRK4.with_cost_residual
:type: bool
:value: >
   True

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelRK4.with_cost_residual
```

````

````{py:method} create_data() -> better_robot.dynamics.action.action.ActionData
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelRK4.create_data

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelRK4.create_data
```

````

````{py:method} calc(data: better_robot.dynamics.action.action.ActionData, x: torch.Tensor, u: torch.Tensor) -> None
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelRK4.calc

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelRK4.calc
```

````

````{py:method} calc_diff(data: better_robot.dynamics.action.action.ActionData, x: torch.Tensor, u: torch.Tensor) -> None
:canonical: better_robot.dynamics.action.integrated.IntegratedActionModelRK4.calc_diff

```{autodoc2-docstring} better_robot.dynamics.action.integrated.IntegratedActionModelRK4.calc_diff
```

````

`````
