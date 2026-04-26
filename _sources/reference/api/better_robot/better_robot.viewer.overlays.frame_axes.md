# {py:mod}`better_robot.viewer.overlays.frame_axes`

```{py:module} better_robot.viewer.overlays.frame_axes
```

```{autodoc2-docstring} better_robot.viewer.overlays.frame_axes
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FrameAxesOverlay <better_robot.viewer.overlays.frame_axes.FrameAxesOverlay>`
  - ```{autodoc2-docstring} better_robot.viewer.overlays.frame_axes.FrameAxesOverlay
    :summary:
    ```
````

### API

`````{py:class} FrameAxesOverlay(*, frame_names: typing.Sequence[str] | None = None, axes_length: float = 0.04, visible: bool = False)
:canonical: better_robot.viewer.overlays.frame_axes.FrameAxesOverlay

```{autodoc2-docstring} better_robot.viewer.overlays.frame_axes.FrameAxesOverlay
```

````{py:attribute} name
:canonical: better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.name
:value: >
   'Frame axes'

```{autodoc2-docstring} better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.name
```

````

````{py:attribute} description
:canonical: better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.description
:value: >
   'Coordinate triads on named frames'

```{autodoc2-docstring} better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.description
```

````

````{py:method} is_available(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> bool
:canonical: better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.is_available
:classmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.is_available
```

````

````{py:method} attach(context: better_robot.viewer.render_modes.base.RenderContext, model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.attach

```{autodoc2-docstring} better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.attach
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.update

```{autodoc2-docstring} better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.update
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.set_visible

```{autodoc2-docstring} better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.set_visible
```

````

````{py:method} detach() -> None
:canonical: better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.detach

```{autodoc2-docstring} better_robot.viewer.overlays.frame_axes.FrameAxesOverlay.detach
```

````

`````
