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
* - {py:obj}`SmoothnessResidual <better_robot.residuals.smoothness.SmoothnessResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.smoothness.SmoothnessResidual
    :summary:
    ```
````

### API

`````{py:class} VelocityResidual(q: better_robot.residuals.utils.RobotVariableLike, *, dt: numbers.Real, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'velocity')
:canonical: better_robot.residuals.smoothness.VelocityResidual

Bases: {py:obj}`better_robot.residuals.smoothness._SmoothnessResidual`

```{autodoc2-docstring} better_robot.residuals.smoothness.VelocityResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.smoothness.VelocityResidual.error

````

`````

`````{py:class} SmoothnessResidual(q: better_robot.residuals.utils.RobotVariableLike, *, order: int, dt: numbers.Real, coordinate_weight: torch.Tensor | None = None, weight: numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'smoothness')
:canonical: better_robot.residuals.smoothness.SmoothnessResidual

Bases: {py:obj}`better_robot.residuals.smoothness._SmoothnessResidual`

```{autodoc2-docstring} better_robot.residuals.smoothness.SmoothnessResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.smoothness.SmoothnessResidual.error

````

`````
