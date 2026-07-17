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
* - {py:obj}`JointAccelLimit <better_robot.residuals.limits.JointAccelLimit>`
  - ```{autodoc2-docstring} better_robot.residuals.limits.JointAccelLimit
    :summary:
    ```
````

### API

`````{py:class} JointPositionLimit(model: better_robot.data_model.model.Model, *, weight: float = 1.0, name: str = 'joint_position_limit')
:canonical: better_robot.residuals.limits.JointPositionLimit

```{autodoc2-docstring} better_robot.residuals.limits.JointPositionLimit
```

````{py:attribute} name
:canonical: better_robot.residuals.limits.JointPositionLimit.name
:type: str
:value: >
   'joint_position_limit'

```{autodoc2-docstring} better_robot.residuals.limits.JointPositionLimit.name
```

````

````{py:attribute} reads
:canonical: better_robot.residuals.limits.JointPositionLimit.reads
:value: >
   ('q',)

```{autodoc2-docstring} better_robot.residuals.limits.JointPositionLimit.reads
```

````

````{py:method} jacobian(value: better_robot.residuals.base.ResidualState | collections.abc.Mapping[str, typing.Any]) -> torch.Tensor | None
:canonical: better_robot.residuals.limits.JointPositionLimit.jacobian

```{autodoc2-docstring} better_robot.residuals.limits.JointPositionLimit.jacobian
```

````

````{py:method} jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any]) -> dict[str, torch.Tensor]
:canonical: better_robot.residuals.limits.JointPositionLimit.jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.limits.JointPositionLimit.jacobian_blocks
```

````

`````

`````{py:class} JointVelocityLimit(model: better_robot.data_model.model.Model, *, weight: float = 1.0)
:canonical: better_robot.residuals.limits.JointVelocityLimit

```{autodoc2-docstring} better_robot.residuals.limits.JointVelocityLimit
```

````{py:attribute} name
:canonical: better_robot.residuals.limits.JointVelocityLimit.name
:type: str
:value: >
   'joint_velocity_limit'

```{autodoc2-docstring} better_robot.residuals.limits.JointVelocityLimit.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.limits.JointVelocityLimit.jacobian
:abstractmethod:

```{autodoc2-docstring} better_robot.residuals.limits.JointVelocityLimit.jacobian
```

````

`````

`````{py:class} JointAccelLimit(model: better_robot.data_model.model.Model, *, weight: float = 1.0)
:canonical: better_robot.residuals.limits.JointAccelLimit

```{autodoc2-docstring} better_robot.residuals.limits.JointAccelLimit
```

````{py:attribute} name
:canonical: better_robot.residuals.limits.JointAccelLimit.name
:type: str
:value: >
   'joint_accel_limit'

```{autodoc2-docstring} better_robot.residuals.limits.JointAccelLimit.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.limits.JointAccelLimit.jacobian
:abstractmethod:

```{autodoc2-docstring} better_robot.residuals.limits.JointAccelLimit.jacobian
```

````

`````
