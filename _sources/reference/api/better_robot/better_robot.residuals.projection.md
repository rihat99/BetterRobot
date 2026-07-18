# {py:mod}`better_robot.residuals.projection`

```{py:module} better_robot.residuals.projection
```

```{autodoc2-docstring} better_robot.residuals.projection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProjectionResidual <better_robot.residuals.projection.ProjectionResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.projection.ProjectionResidual
    :summary:
    ```
````

### API

`````{py:class} ProjectionResidual(model: better_robot.data_model.model.Model, point_ids: collections.abc.Sequence[int] | torch.Tensor, K: torch.Tensor, extrinsics: torch.Tensor, target_px: torch.Tensor, *, weights: torch.Tensor | None = None, valid_mask: torch.Tensor | None = None, K_name: str | None = None, extrinsics_name: str | None = None, target_name: str | None = None, weights_name: str | None = None, valid_mask_name: str | None = None, min_depth: float = 1e-06, name: str = 'projection')
:canonical: better_robot.residuals.projection.ProjectionResidual

```{autodoc2-docstring} better_robot.residuals.projection.ProjectionResidual
```

````{py:attribute} reads
:canonical: better_robot.residuals.projection.ProjectionResidual.reads
:value: >
   ('data',)

```{autodoc2-docstring} better_robot.residuals.projection.ProjectionResidual.reads
```

````

````{py:method} jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any]) -> dict[str, torch.Tensor]
:canonical: better_robot.residuals.projection.ProjectionResidual.jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.projection.ProjectionResidual.jacobian_blocks
```

````

`````
