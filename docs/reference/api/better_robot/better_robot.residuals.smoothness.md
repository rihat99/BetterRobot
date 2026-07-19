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

`````{py:class} VelocityResidual(q: better_robot.residuals._variables.RobotVariableLike, *, dt: numbers.Real, weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'velocity')
:canonical: better_robot.residuals.smoothness.VelocityResidual

Bases: {py:obj}`better_robot.residuals.smoothness._SmoothnessResidual`

```{autodoc2-docstring} better_robot.residuals.smoothness.VelocityResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.smoothness.VelocityResidual.error

````

`````

`````{py:class} AccelerationResidual(q: better_robot.residuals._variables.RobotVariableLike, *, dt: numbers.Real, weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'acceleration')
:canonical: better_robot.residuals.smoothness.AccelerationResidual

Bases: {py:obj}`better_robot.residuals.smoothness._SmoothnessResidual`

```{autodoc2-docstring} better_robot.residuals.smoothness.AccelerationResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.smoothness.AccelerationResidual.error

````

`````

`````{py:class} JerkResidual(q: better_robot.residuals._variables.RobotVariableLike, *, dt: numbers.Real, weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, name: str = 'jerk')
:canonical: better_robot.residuals.smoothness.JerkResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.smoothness.JerkResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.smoothness.JerkResidual.error
:abstractmethod:

````

`````
