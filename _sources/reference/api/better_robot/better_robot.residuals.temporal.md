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

`````{py:class} TimeIndexedResidual(inner, t_idx: int, *, horizon: int | None = None, name: str | None = None)
:canonical: better_robot.residuals.temporal.TimeIndexedResidual

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual
```

````{py:method} temporal_structure(variable_name: str) -> better_robot.residuals.structure.TemporalPattern | None
:canonical: better_robot.residuals.temporal.TimeIndexedResidual.temporal_structure

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual.temporal_structure
```

````

````{py:method} temporal_jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any], variable_name: str) -> collections.abc.Mapping[int, torch.Tensor]
:canonical: better_robot.residuals.temporal.TimeIndexedResidual.temporal_jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual.temporal_jacobian_blocks
```

````

````{py:method} jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any]) -> dict[str, torch.Tensor]
:canonical: better_robot.residuals.temporal.TimeIndexedResidual.jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.temporal.TimeIndexedResidual.jacobian_blocks
```

````

`````
