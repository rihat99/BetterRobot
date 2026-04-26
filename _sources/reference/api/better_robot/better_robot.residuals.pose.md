# {py:mod}`better_robot.residuals.pose`

```{py:module} better_robot.residuals.pose
```

```{autodoc2-docstring} better_robot.residuals.pose
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PoseResidual <better_robot.residuals.pose.PoseResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.pose.PoseResidual
    :summary:
    ```
* - {py:obj}`PositionResidual <better_robot.residuals.pose.PositionResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.pose.PositionResidual
    :summary:
    ```
* - {py:obj}`OrientationResidual <better_robot.residuals.pose.OrientationResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.pose.OrientationResidual
    :summary:
    ```
````

### API

`````{py:class} PoseResidual(*, frame_id: int, target: torch.Tensor, pos_weight: float = 1.0, ori_weight: float = 1.0)
:canonical: better_robot.residuals.pose.PoseResidual

```{autodoc2-docstring} better_robot.residuals.pose.PoseResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.pose.PoseResidual.name
:type: str
:value: >
   'pose'

```{autodoc2-docstring} better_robot.residuals.pose.PoseResidual.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.pose.PoseResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.pose.PoseResidual.jacobian
```

````

`````

`````{py:class} PositionResidual(*, frame_id: int, target: torch.Tensor, weight: float = 1.0)
:canonical: better_robot.residuals.pose.PositionResidual

```{autodoc2-docstring} better_robot.residuals.pose.PositionResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.pose.PositionResidual.name
:type: str
:value: >
   'position'

```{autodoc2-docstring} better_robot.residuals.pose.PositionResidual.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.pose.PositionResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.pose.PositionResidual.jacobian
```

````

`````

`````{py:class} OrientationResidual(*, frame_id: int, target: torch.Tensor, weight: float = 1.0)
:canonical: better_robot.residuals.pose.OrientationResidual

```{autodoc2-docstring} better_robot.residuals.pose.OrientationResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.pose.OrientationResidual.name
:type: str
:value: >
   'orientation'

```{autodoc2-docstring} better_robot.residuals.pose.OrientationResidual.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.pose.OrientationResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.pose.OrientationResidual.jacobian
```

````

`````
