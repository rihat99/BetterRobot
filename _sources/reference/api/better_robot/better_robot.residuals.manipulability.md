# {py:mod}`better_robot.residuals.manipulability`

```{py:module} better_robot.residuals.manipulability
```

```{autodoc2-docstring} better_robot.residuals.manipulability
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`YoshikawaResidual <better_robot.residuals.manipulability.YoshikawaResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.manipulability.YoshikawaResidual
    :summary:
    ```
````

### API

`````{py:class} YoshikawaResidual(*, frame_id: int, weight: float = 1.0)
:canonical: better_robot.residuals.manipulability.YoshikawaResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.manipulability.YoshikawaResidual
```

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.manipulability.YoshikawaResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.manipulability.YoshikawaResidual.jacobian
```

````

`````
