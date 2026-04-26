# {py:mod}`better_robot.viewer.overlays.com`

```{py:module} better_robot.viewer.overlays.com
```

```{autodoc2-docstring} better_robot.viewer.overlays.com
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ComOverlay <better_robot.viewer.overlays.com.ComOverlay>`
  - ```{autodoc2-docstring} better_robot.viewer.overlays.com.ComOverlay
    :summary:
    ```
````

### API

`````{py:class} ComOverlay
:canonical: better_robot.viewer.overlays.com.ComOverlay

```{autodoc2-docstring} better_robot.viewer.overlays.com.ComOverlay
```

````{py:attribute} name
:canonical: better_robot.viewer.overlays.com.ComOverlay.name
:value: >
   'com'

```{autodoc2-docstring} better_robot.viewer.overlays.com.ComOverlay.name
```

````

````{py:attribute} description
:canonical: better_robot.viewer.overlays.com.ComOverlay.description
:value: >
   'See docs/design/12_VIEWER.md §5'

```{autodoc2-docstring} better_robot.viewer.overlays.com.ComOverlay.description
```

````

````{py:method} is_available(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> bool
:canonical: better_robot.viewer.overlays.com.ComOverlay.is_available
:classmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.com.ComOverlay.is_available
```

````

````{py:method} attach(context: better_robot.viewer.render_modes.base.RenderContext, model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.com.ComOverlay.attach
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.com.ComOverlay.attach
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.com.ComOverlay.update
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.com.ComOverlay.update
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.overlays.com.ComOverlay.set_visible
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.com.ComOverlay.set_visible
```

````

````{py:method} detach() -> None
:canonical: better_robot.viewer.overlays.com.ComOverlay.detach
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.com.ComOverlay.detach
```

````

`````
