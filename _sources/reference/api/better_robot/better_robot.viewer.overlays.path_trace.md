# {py:mod}`better_robot.viewer.overlays.path_trace`

```{py:module} better_robot.viewer.overlays.path_trace
```

```{autodoc2-docstring} better_robot.viewer.overlays.path_trace
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PathTraceOverlay <better_robot.viewer.overlays.path_trace.PathTraceOverlay>`
  - ```{autodoc2-docstring} better_robot.viewer.overlays.path_trace.PathTraceOverlay
    :summary:
    ```
````

### API

`````{py:class} PathTraceOverlay(*args: typing.Any, **kwargs: typing.Any)
:canonical: better_robot.viewer.overlays.path_trace.PathTraceOverlay

```{autodoc2-docstring} better_robot.viewer.overlays.path_trace.PathTraceOverlay
```

````{py:attribute} name
:canonical: better_robot.viewer.overlays.path_trace.PathTraceOverlay.name
:value: >
   'path_trace'

```{autodoc2-docstring} better_robot.viewer.overlays.path_trace.PathTraceOverlay.name
```

````

````{py:attribute} description
:canonical: better_robot.viewer.overlays.path_trace.PathTraceOverlay.description
:value: >
   'World-frame path of a named frame (future work §10.6)'

```{autodoc2-docstring} better_robot.viewer.overlays.path_trace.PathTraceOverlay.description
```

````

````{py:method} is_available(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> bool
:canonical: better_robot.viewer.overlays.path_trace.PathTraceOverlay.is_available
:classmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.path_trace.PathTraceOverlay.is_available
```

````

````{py:method} attach(context: better_robot.viewer.render_modes.base.RenderContext, model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.path_trace.PathTraceOverlay.attach
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.path_trace.PathTraceOverlay.attach
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.path_trace.PathTraceOverlay.update
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.path_trace.PathTraceOverlay.update
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.overlays.path_trace.PathTraceOverlay.set_visible
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.path_trace.PathTraceOverlay.set_visible
```

````

````{py:method} detach() -> None
:canonical: better_robot.viewer.overlays.path_trace.PathTraceOverlay.detach
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.path_trace.PathTraceOverlay.detach
```

````

`````
