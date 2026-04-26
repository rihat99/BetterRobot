# {py:mod}`better_robot.viewer.render_modes.collision`

```{py:module} better_robot.viewer.render_modes.collision
```

```{autodoc2-docstring} better_robot.viewer.render_modes.collision
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CollisionMode <better_robot.viewer.render_modes.collision.CollisionMode>`
  - ```{autodoc2-docstring} better_robot.viewer.render_modes.collision.CollisionMode
    :summary:
    ```
````

### API

`````{py:class} CollisionMode(robot_collision: object, *, alpha: float = 0.35)
:canonical: better_robot.viewer.render_modes.collision.CollisionMode

```{autodoc2-docstring} better_robot.viewer.render_modes.collision.CollisionMode
```

````{py:attribute} name
:canonical: better_robot.viewer.render_modes.collision.CollisionMode.name
:value: >
   'Collision'

```{autodoc2-docstring} better_robot.viewer.render_modes.collision.CollisionMode.name
```

````

````{py:attribute} description
:canonical: better_robot.viewer.render_modes.collision.CollisionMode.description
:value: >
   'Capsule-based collision decomposition'

```{autodoc2-docstring} better_robot.viewer.render_modes.collision.CollisionMode.description
```

````

````{py:method} is_available(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data, robot_collision: object = None) -> bool
:canonical: better_robot.viewer.render_modes.collision.CollisionMode.is_available
:classmethod:

```{autodoc2-docstring} better_robot.viewer.render_modes.collision.CollisionMode.is_available
```

````

````{py:method} attach(context: better_robot.viewer.render_modes.base.RenderContext, model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.render_modes.collision.CollisionMode.attach
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.render_modes.collision.CollisionMode.attach
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.render_modes.collision.CollisionMode.update
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.render_modes.collision.CollisionMode.update
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.render_modes.collision.CollisionMode.set_visible
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.render_modes.collision.CollisionMode.set_visible
```

````

````{py:method} detach() -> None
:canonical: better_robot.viewer.render_modes.collision.CollisionMode.detach
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.render_modes.collision.CollisionMode.detach
```

````

`````
