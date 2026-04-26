# {py:mod}`better_robot.viewer.render_modes.base`

```{py:module} better_robot.viewer.render_modes.base
```

```{autodoc2-docstring} better_robot.viewer.render_modes.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RenderContext <better_robot.viewer.render_modes.base.RenderContext>`
  - ```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderContext
    :summary:
    ```
* - {py:obj}`RenderMode <better_robot.viewer.render_modes.base.RenderMode>`
  - ```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderMode
    :summary:
    ```
````

### API

`````{py:class} RenderContext
:canonical: better_robot.viewer.render_modes.base.RenderContext

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderContext
```

````{py:attribute} backend
:canonical: better_robot.viewer.render_modes.base.RenderContext.backend
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderContext.backend
```

````

````{py:attribute} namespace
:canonical: better_robot.viewer.render_modes.base.RenderContext.namespace
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderContext.namespace
```

````

````{py:attribute} batch_index
:canonical: better_robot.viewer.render_modes.base.RenderContext.batch_index
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderContext.batch_index
```

````

````{py:attribute} theme
:canonical: better_robot.viewer.render_modes.base.RenderContext.theme
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderContext.theme
```

````

`````

`````{py:class} RenderMode
:canonical: better_robot.viewer.render_modes.base.RenderMode

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderMode
```

````{py:attribute} name
:canonical: better_robot.viewer.render_modes.base.RenderMode.name
:type: typing.ClassVar[str]
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderMode.name
```

````

````{py:attribute} description
:canonical: better_robot.viewer.render_modes.base.RenderMode.description
:type: typing.ClassVar[str]
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderMode.description
```

````

````{py:method} is_available(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> bool
:canonical: better_robot.viewer.render_modes.base.RenderMode.is_available
:classmethod:

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderMode.is_available
```

````

````{py:method} attach(context: better_robot.viewer.render_modes.base.RenderContext, model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.render_modes.base.RenderMode.attach

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderMode.attach
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.render_modes.base.RenderMode.update

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderMode.update
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.render_modes.base.RenderMode.set_visible

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderMode.set_visible
```

````

````{py:method} detach() -> None
:canonical: better_robot.viewer.render_modes.base.RenderMode.detach

```{autodoc2-docstring} better_robot.viewer.render_modes.base.RenderMode.detach
```

````

`````
