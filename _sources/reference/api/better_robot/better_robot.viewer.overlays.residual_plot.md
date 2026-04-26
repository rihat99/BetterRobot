# {py:mod}`better_robot.viewer.overlays.residual_plot`

```{py:module} better_robot.viewer.overlays.residual_plot
```

```{autodoc2-docstring} better_robot.viewer.overlays.residual_plot
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ResidualPlotOverlay <better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay>`
  - ```{autodoc2-docstring} better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay
    :summary:
    ```
````

### API

`````{py:class} ResidualPlotOverlay
:canonical: better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay

```{autodoc2-docstring} better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay
```

````{py:attribute} name
:canonical: better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.name
:value: >
   'residual_plot'

```{autodoc2-docstring} better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.name
```

````

````{py:attribute} description
:canonical: better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.description
:value: >
   'See docs/concepts/viewer.md §5'

```{autodoc2-docstring} better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.description
```

````

````{py:method} is_available(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> bool
:canonical: better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.is_available
:classmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.is_available
```

````

````{py:method} attach(context: better_robot.viewer.render_modes.base.RenderContext, model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.attach
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.attach
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.update
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.update
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.set_visible
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.set_visible
```

````

````{py:method} detach() -> None
:canonical: better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.detach
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.residual_plot.ResidualPlotOverlay.detach
```

````

`````
