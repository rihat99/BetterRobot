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

`````{py:class} TimeIndexedResidual(inner: better_robot.residuals.base.Residual, t_idx: int, *, name: str | None = None)
:canonical: better_robot.residuals.temporal.TimeIndexedResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.temporal.TimeIndexedResidual.error

````

````{py:method} temporal_structure(variable: better_robot.residuals.utils.RobotVariableLike | str) -> better_robot.residuals.structure.TemporalPattern | None
:canonical: better_robot.residuals.temporal.TimeIndexedResidual.temporal_structure

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual.temporal_structure
```

````

````{py:method} temporal_jacobian_blocks(variable: better_robot.residuals.utils.RobotVariableLike | str) -> collections.abc.Mapping[int, torch.Tensor]
:canonical: better_robot.residuals.temporal.TimeIndexedResidual.temporal_jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual.temporal_jacobian_blocks
```

````

````{py:method} jacobian() -> tuple[torch.Tensor, ...]
:canonical: better_robot.residuals.temporal.TimeIndexedResidual.jacobian

````

`````
