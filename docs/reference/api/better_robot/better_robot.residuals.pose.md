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

````{py:class} PoseResidual(q_or_state: better_robot.residuals.utils.RobotVariableLike | better_robot.residuals.nodes.RobotState, *, frame: str | None = None, frame_id: int | None = None, target: torch.Tensor | better_robot.residuals.utils.VariableLike, knot: int | None = None, pos_weight: numbers.Real = 1.0, ori_weight: numbers.Real = 1.0, weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'pose')
:canonical: better_robot.residuals.pose.PoseResidual

Bases: {py:obj}`better_robot.residuals.pose._KinematicResidual`

```{autodoc2-docstring} better_robot.residuals.pose.PoseResidual
```

````

````{py:class} PositionResidual(q_or_state: better_robot.residuals.utils.RobotVariableLike | better_robot.residuals.nodes.RobotState, *, frame: str | None = None, frame_id: int | None = None, target: torch.Tensor | better_robot.residuals.utils.VariableLike, knot: int | None = None, weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'position')
:canonical: better_robot.residuals.pose.PositionResidual

Bases: {py:obj}`better_robot.residuals.pose._KinematicResidual`

```{autodoc2-docstring} better_robot.residuals.pose.PositionResidual
```

````

````{py:class} OrientationResidual(q_or_state: better_robot.residuals.utils.RobotVariableLike | better_robot.residuals.nodes.RobotState, *, frame: str | None = None, frame_id: int | None = None, target: torch.Tensor | better_robot.residuals.utils.VariableLike, knot: int | None = None, weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'orientation')
:canonical: better_robot.residuals.pose.OrientationResidual

Bases: {py:obj}`better_robot.residuals.pose._KinematicResidual`

```{autodoc2-docstring} better_robot.residuals.pose.OrientationResidual
```

````
