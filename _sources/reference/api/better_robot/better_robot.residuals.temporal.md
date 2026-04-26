# {py:mod}`better_robot.residuals.temporal`

```{py:module} better_robot.residuals.temporal
```

```{autodoc2-docstring} better_robot.residuals.temporal
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TimeIndexedResidual <better_robot.residuals.temporal.TimeIndexedResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual
    :summary:
    ```
````

### API

`````{py:class} TimeIndexedResidual(inner, t_idx: int, *, name: str | None = None)
:canonical: better_robot.residuals.temporal.TimeIndexedResidual

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual
```

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.temporal.TimeIndexedResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual.jacobian
```

````

````{py:method} apply_jac_transpose(state: better_robot.residuals.base.ResidualState, vec: torch.Tensor) -> torch.Tensor
:canonical: better_robot.residuals.temporal.TimeIndexedResidual.apply_jac_transpose

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual.apply_jac_transpose
```

````

````{py:property} spec
:canonical: better_robot.residuals.temporal.TimeIndexedResidual.spec

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual.spec
```

````

`````
