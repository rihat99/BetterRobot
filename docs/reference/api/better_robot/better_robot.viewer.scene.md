# {py:mod}`better_robot.viewer.scene`

```{py:module} better_robot.viewer.scene
```

```{autodoc2-docstring} better_robot.viewer.scene
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Scene <better_robot.viewer.scene.Scene>`
  - ```{autodoc2-docstring} better_robot.viewer.scene.Scene
    :summary:
    ```
````

### API

`````{py:class} Scene(model: better_robot.data_model.model.Model, *, backend: typing.Any, namespace: str = '/robot', theme: better_robot.viewer.themes.Theme | None = None)
:canonical: better_robot.viewer.scene.Scene

```{autodoc2-docstring} better_robot.viewer.scene.Scene
```

````{py:method} default(model: better_robot.data_model.model.Model, *, backend: typing.Any, theme: better_robot.viewer.themes.Theme | None = None) -> better_robot.viewer.scene.Scene
:canonical: better_robot.viewer.scene.Scene.default
:classmethod:

```{autodoc2-docstring} better_robot.viewer.scene.Scene.default
```

````

````{py:method} add_mode(mode: typing.Any) -> None
:canonical: better_robot.viewer.scene.Scene.add_mode

```{autodoc2-docstring} better_robot.viewer.scene.Scene.add_mode
```

````

````{py:method} remove_mode(mode_name: str) -> None
:canonical: better_robot.viewer.scene.Scene.remove_mode

```{autodoc2-docstring} better_robot.viewer.scene.Scene.remove_mode
```

````

````{py:method} available_modes() -> list[str]
:canonical: better_robot.viewer.scene.Scene.available_modes

```{autodoc2-docstring} better_robot.viewer.scene.Scene.available_modes
```

````

````{py:method} set_mode_visible(mode_name: str, visible: bool) -> None
:canonical: better_robot.viewer.scene.Scene.set_mode_visible

```{autodoc2-docstring} better_robot.viewer.scene.Scene.set_mode_visible
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.scene.Scene.update

```{autodoc2-docstring} better_robot.viewer.scene.Scene.update
```

````

````{py:method} update_from_q(q: torch.Tensor) -> None
:canonical: better_robot.viewer.scene.Scene.update_from_q

```{autodoc2-docstring} better_robot.viewer.scene.Scene.update_from_q
```

````

`````
