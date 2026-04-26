# {py:mod}`better_robot.residuals.smoothness`

```{py:module} better_robot.residuals.smoothness
```

```{autodoc2-docstring} better_robot.residuals.smoothness
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VelocityResidual <better_robot.residuals.smoothness.VelocityResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.smoothness.VelocityResidual
    :summary:
    ```
* - {py:obj}`AccelerationResidual <better_robot.residuals.smoothness.AccelerationResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.smoothness.AccelerationResidual
    :summary:
    ```
* - {py:obj}`JerkResidual <better_robot.residuals.smoothness.JerkResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.smoothness.JerkResidual
    :summary:
    ```
````

### API

`````{py:class} VelocityResidual(model: better_robot.data_model.model.Model, *, dt: float, weight: float = 1.0)
:canonical: better_robot.residuals.smoothness.VelocityResidual

```{autodoc2-docstring} better_robot.residuals.smoothness.VelocityResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.smoothness.VelocityResidual.name
:type: str
:value: >
   'velocity'

```{autodoc2-docstring} better_robot.residuals.smoothness.VelocityResidual.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.smoothness.VelocityResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.smoothness.VelocityResidual.jacobian
```

````

````{py:method} apply_jac_transpose(state: better_robot.residuals.base.ResidualState, r: torch.Tensor) -> torch.Tensor
:canonical: better_robot.residuals.smoothness.VelocityResidual.apply_jac_transpose

```{autodoc2-docstring} better_robot.residuals.smoothness.VelocityResidual.apply_jac_transpose
```

````

`````

`````{py:class} AccelerationResidual(model: better_robot.data_model.model.Model, *, dt: float, weight: float = 1.0)
:canonical: better_robot.residuals.smoothness.AccelerationResidual

```{autodoc2-docstring} better_robot.residuals.smoothness.AccelerationResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.smoothness.AccelerationResidual.name
:type: str
:value: >
   'acceleration'

```{autodoc2-docstring} better_robot.residuals.smoothness.AccelerationResidual.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.smoothness.AccelerationResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.smoothness.AccelerationResidual.jacobian
```

````

````{py:method} apply_jac_transpose(state: better_robot.residuals.base.ResidualState, r: torch.Tensor) -> torch.Tensor
:canonical: better_robot.residuals.smoothness.AccelerationResidual.apply_jac_transpose

```{autodoc2-docstring} better_robot.residuals.smoothness.AccelerationResidual.apply_jac_transpose
```

````

`````

`````{py:class} JerkResidual(model: better_robot.data_model.model.Model, *, dt: float, weight: float = 1.0)
:canonical: better_robot.residuals.smoothness.JerkResidual

```{autodoc2-docstring} better_robot.residuals.smoothness.JerkResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.smoothness.JerkResidual.name
:type: str
:value: >
   'jerk'

```{autodoc2-docstring} better_robot.residuals.smoothness.JerkResidual.name
```

````

````{py:attribute} dim
:canonical: better_robot.residuals.smoothness.JerkResidual.dim
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.residuals.smoothness.JerkResidual.dim
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.smoothness.JerkResidual.jacobian
:abstractmethod:

```{autodoc2-docstring} better_robot.residuals.smoothness.JerkResidual.jacobian
```

````

`````
