# {py:mod}`better_robot.residuals.collision`

```{py:module} better_robot.residuals.collision
```

```{autodoc2-docstring} better_robot.residuals.collision
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SelfCollisionResidual <better_robot.residuals.collision.SelfCollisionResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.collision.SelfCollisionResidual
    :summary:
    ```
* - {py:obj}`WorldCollisionResidual <better_robot.residuals.collision.WorldCollisionResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.collision.WorldCollisionResidual
    :summary:
    ```
````

### API

`````{py:class} SelfCollisionResidual(model: better_robot.data_model.model.Model, robot_collision: better_robot.collision.robot_collision.RobotCollision, *, margin: float = 0.02, weight: float = 1.0)
:canonical: better_robot.residuals.collision.SelfCollisionResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.collision.SelfCollisionResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.collision.SelfCollisionResidual.name
:type: str
:value: >
   'self_collision'

```{autodoc2-docstring} better_robot.residuals.collision.SelfCollisionResidual.name
```

````

````{py:attribute} reads
:canonical: better_robot.residuals.collision.SelfCollisionResidual.reads
:value: >
   ('q', 'data')

```{autodoc2-docstring} better_robot.residuals.collision.SelfCollisionResidual.reads
```

````

`````

`````{py:class} WorldCollisionResidual(model: better_robot.data_model.model.Model, robot_collision: better_robot.collision.robot_collision.RobotCollision, world: typing.Sequence[better_robot.collision.geometry.Sphere | better_robot.collision.geometry.Capsule | better_robot.collision.geometry.Box], *, margin: float = 0.02, weight: float = 1.0)
:canonical: better_robot.residuals.collision.WorldCollisionResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.collision.WorldCollisionResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.collision.WorldCollisionResidual.name
:type: str
:value: >
   'world_collision'

```{autodoc2-docstring} better_robot.residuals.collision.WorldCollisionResidual.name
```

````

````{py:attribute} reads
:canonical: better_robot.residuals.collision.WorldCollisionResidual.reads
:value: >
   ('q', 'data')

```{autodoc2-docstring} better_robot.residuals.collision.WorldCollisionResidual.reads
```

````

`````
