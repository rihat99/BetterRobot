# {py:mod}`better_robot.residuals.human`

```{py:module} better_robot.residuals.human
```

```{autodoc2-docstring} better_robot.residuals.human
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SwingTwistLimitResidual <better_robot.residuals.human.SwingTwistLimitResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.human.SwingTwistLimitResidual
    :summary:
    ```
````

### API

`````{py:class} SwingTwistLimitResidual(q: better_robot.residuals._variables.RobotValueLike, joint_ids: collections.abc.Sequence[int], twist_axis: torch.Tensor, swing_max: numbers.Real | torch.Tensor, twist_range: tuple[numbers.Real, numbers.Real] | torch.Tensor, *, weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'swing_twist_limit')
:canonical: better_robot.residuals.human.SwingTwistLimitResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.human.SwingTwistLimitResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.human.SwingTwistLimitResidual.error

````

`````
