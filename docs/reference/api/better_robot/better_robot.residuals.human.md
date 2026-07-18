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

`````{py:class} SwingTwistLimitResidual(model: better_robot.data_model.model.Model, joint_ids: collections.abc.Sequence[int], twist_axis: torch.Tensor, swing_max: numbers.Real | torch.Tensor, twist_range: tuple[numbers.Real, numbers.Real] | torch.Tensor, *, name: str = 'swing_twist_limit')
:canonical: better_robot.residuals.human.SwingTwistLimitResidual

```{autodoc2-docstring} better_robot.residuals.human.SwingTwistLimitResidual
```

````{py:attribute} reads
:canonical: better_robot.residuals.human.SwingTwistLimitResidual.reads
:value: >
   ('q',)

```{autodoc2-docstring} better_robot.residuals.human.SwingTwistLimitResidual.reads
```

````

`````
