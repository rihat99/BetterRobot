# {py:mod}`better_robot.residuals.limits`

```{py:module} better_robot.residuals.limits
```

```{autodoc2-docstring} better_robot.residuals.limits
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointPositionLimit <better_robot.residuals.limits.JointPositionLimit>`
  - ```{autodoc2-docstring} better_robot.residuals.limits.JointPositionLimit
    :summary:
    ```
* - {py:obj}`JointVelocityLimit <better_robot.residuals.limits.JointVelocityLimit>`
  - ```{autodoc2-docstring} better_robot.residuals.limits.JointVelocityLimit
    :summary:
    ```
````

### API

`````{py:class} JointPositionLimit(q: better_robot.residuals.utils.RobotVariableLike, *, knot: int | None = None, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'joint_position_limit')
:canonical: better_robot.residuals.limits.JointPositionLimit

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.limits.JointPositionLimit
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.limits.JointPositionLimit.error

````

````{py:method} temporal_structure(variable: better_robot.residuals.utils.RobotVariableLike | str) -> better_robot.residuals.structure.TemporalPattern | None
:canonical: better_robot.residuals.limits.JointPositionLimit.temporal_structure

```{autodoc2-docstring} better_robot.residuals.limits.JointPositionLimit.temporal_structure
```

````

````{py:method} temporal_jacobian_blocks(variable: better_robot.residuals.utils.RobotVariableLike | str) -> collections.abc.Mapping[int, torch.Tensor]
:canonical: better_robot.residuals.limits.JointPositionLimit.temporal_jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.limits.JointPositionLimit.temporal_jacobian_blocks
```

````

````{py:method} jacobian() -> tuple[torch.Tensor, ...]
:canonical: better_robot.residuals.limits.JointPositionLimit.jacobian

````

`````

`````{py:class} JointVelocityLimit(velocity: better_robot.residuals.utils.VariableLike, limit: torch.Tensor | better_robot.residuals.utils.VariableLike, *, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'joint_velocity_limit')
:canonical: better_robot.residuals.limits.JointVelocityLimit

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.limits.JointVelocityLimit
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.limits.JointVelocityLimit.error

````

`````
