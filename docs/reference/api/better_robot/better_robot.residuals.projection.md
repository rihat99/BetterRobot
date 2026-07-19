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

`````{py:class} ProjectionResidual(q_or_state: better_robot.residuals._variables.RobotVariableLike | better_robot.residuals.nodes.RobotState, point_ids: collections.abc.Sequence[int] | torch.Tensor, K: better_robot.residuals._variables.VariableLike | torch.Tensor, extrinsics: better_robot.residuals._variables.VariableLike | torch.Tensor, target_px: better_robot.residuals._variables.VariableLike | torch.Tensor, *, weights: better_robot.residuals._variables.VariableLike | torch.Tensor | None = None, valid_mask: better_robot.residuals._variables.VariableLike | torch.Tensor | None = None, min_depth: float = 1e-06, weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'projection')
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
