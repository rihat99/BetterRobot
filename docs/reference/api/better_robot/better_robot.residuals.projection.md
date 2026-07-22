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
* - {py:obj}`PointProjectionResidual <better_robot.residuals.projection.PointProjectionResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.projection.PointProjectionResidual
    :summary:
    ```
````

### API

`````{py:class} ProjectionResidual(q_or_state: better_robot.residuals.utils.RobotVariableLike | better_robot.residuals.nodes.RobotState, point_ids: collections.abc.Sequence[int] | torch.Tensor, K: better_robot.residuals.utils.VariableLike | torch.Tensor, extrinsics: better_robot.residuals.utils.VariableLike | torch.Tensor, target_px: better_robot.residuals.utils.VariableLike | torch.Tensor, *, weights: better_robot.residuals.utils.VariableLike | torch.Tensor | None = None, valid_mask: better_robot.residuals.utils.VariableLike | torch.Tensor | None = None, min_depth: float = 1e-06, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'projection')
:canonical: better_robot.residuals.projection.ProjectionResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.projection.ProjectionResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.projection.ProjectionResidual.error

````

````{py:method} jacobian() -> tuple[torch.Tensor, ...] | None
:canonical: better_robot.residuals.projection.ProjectionResidual.jacobian

````

`````

`````{py:class} PointProjectionResidual(points: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor, K: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor, extrinsics: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor, target_px: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor, *, confidence: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor | None = None, visibility: better_robot.residuals.utils.VariableLike | better_robot.residuals.nodes.Node | torch.Tensor | None = None, time_axis: int | None = None, min_depth: float = 1e-06, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, reduce: typing.Literal[sum, mean, mean_active] = 'sum', kernel: object | None = None, name: str = 'point_projection')
:canonical: better_robot.residuals.projection.PointProjectionResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.projection.PointProjectionResidual
```

````{py:method} weight() -> numbers.Real | torch.Tensor
:canonical: better_robot.residuals.projection.PointProjectionResidual.weight

```{autodoc2-docstring} better_robot.residuals.projection.PointProjectionResidual.weight
```

````

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.projection.PointProjectionResidual.error

````

````{py:method} active_groups() -> torch.Tensor | None
:canonical: better_robot.residuals.projection.PointProjectionResidual.active_groups

````

`````
