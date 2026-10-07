# {py:mod}`better_robot.viewer.layers.plot`

```{py:module} better_robot.viewer.layers.plot
```

```{autodoc2-docstring} better_robot.viewer.layers.plot
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PlotPanel <better_robot.viewer.layers.plot.PlotPanel>`
  - ```{autodoc2-docstring} better_robot.viewer.layers.plot.PlotPanel
    :summary:
    ```
````

### API

`````{py:class} PlotPanel(name: str, series: numpy.ndarray, *, labels: list[str] | None = None, fps: float = 30.0)
:canonical: better_robot.viewer.layers.plot.PlotPanel

Bases: {py:obj}`better_robot.viewer.layer.Layer`

```{autodoc2-docstring} better_robot.viewer.layers.plot.PlotPanel
```

````{py:method} build(scene: typing.Any, root: str) -> None
:canonical: better_robot.viewer.layers.plot.PlotPanel.build

````

````{py:method} build_gui(gui: typing.Any) -> None
:canonical: better_robot.viewer.layers.plot.PlotPanel.build_gui

````

````{py:method} set_frame(frame: int) -> None
:canonical: better_robot.viewer.layers.plot.PlotPanel.set_frame

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.layers.plot.PlotPanel.set_visible

````

````{py:method} dispose() -> None
:canonical: better_robot.viewer.layers.plot.PlotPanel.dispose

````

`````
