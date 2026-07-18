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

`````{py:class} PoseResidual(*, frame_id: int, target: torch.Tensor, pos_weight: float = 1.0, ori_weight: float = 1.0, model: better_robot.data_model.model.Model | None = None, name: str = 'pose', target_name: str | None = None)
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

````{py:attribute} reads
:canonical: better_robot.residuals.pose.PoseResidual.reads
:value: >
   ('q', 'data')

```{autodoc2-docstring} better_robot.residuals.pose.PoseResidual.reads
```

````

````{py:method} jacobian(value: better_robot.residuals.base.ResidualState | collections.abc.Mapping[str, typing.Any]) -> torch.Tensor | None
:canonical: better_robot.residuals.pose.PoseResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.pose.PoseResidual.jacobian
```

````

````{py:method} jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any]) -> dict[str, torch.Tensor]
:canonical: better_robot.residuals.pose.PoseResidual.jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.pose.PoseResidual.jacobian_blocks
```

````

`````

`````{py:class} PositionResidual(*, frame_id: int, target: torch.Tensor, weight: float = 1.0, model: better_robot.data_model.model.Model | None = None, name: str = 'position', target_name: str | None = None)
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

````{py:attribute} reads
:canonical: better_robot.residuals.pose.PositionResidual.reads
:value: >
   ('q', 'data')

```{autodoc2-docstring} better_robot.residuals.pose.PositionResidual.reads
```

````

````{py:method} jacobian(value: better_robot.residuals.base.ResidualState | collections.abc.Mapping[str, typing.Any]) -> torch.Tensor | None
:canonical: better_robot.residuals.pose.PositionResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.pose.PositionResidual.jacobian
```

````

````{py:method} jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any]) -> dict[str, torch.Tensor]
:canonical: better_robot.residuals.pose.PositionResidual.jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.pose.PositionResidual.jacobian_blocks
```

````

`````

`````{py:class} OrientationResidual(*, frame_id: int, target: torch.Tensor, weight: float = 1.0, model: better_robot.data_model.model.Model | None = None, name: str = 'orientation', target_name: str | None = None)
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

````{py:attribute} reads
:canonical: better_robot.residuals.pose.OrientationResidual.reads
:value: >
   ('q', 'data')

```{autodoc2-docstring} better_robot.residuals.pose.OrientationResidual.reads
```

````

````{py:method} jacobian(value: better_robot.residuals.base.ResidualState | collections.abc.Mapping[str, typing.Any]) -> torch.Tensor | None
:canonical: better_robot.residuals.pose.OrientationResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.pose.OrientationResidual.jacobian
```

````

````{py:method} jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any]) -> dict[str, torch.Tensor]
:canonical: better_robot.residuals.pose.OrientationResidual.jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.pose.OrientationResidual.jacobian_blocks
```

````

`````
