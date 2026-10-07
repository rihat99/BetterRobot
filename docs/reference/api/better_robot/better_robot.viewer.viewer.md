# {py:mod}`better_robot.viewer.viewer`

```{py:module} better_robot.viewer.viewer
```

```{autodoc2-docstring} better_robot.viewer.viewer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Viewer <better_robot.viewer.viewer.Viewer>`
  - ```{autodoc2-docstring} better_robot.viewer.viewer.Viewer
    :summary:
    ```
````

### API

`````{py:class} Viewer(*, host: str = '127.0.0.1', port: int = 8080, fps: float = 30.0, theme: better_robot.viewer.theme.Theme | None = None, server: typing.Any | None = None)
:canonical: better_robot.viewer.viewer.Viewer

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer
```

````{py:property} url
:canonical: better_robot.viewer.viewer.Viewer.url
:type: str

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer.url
```

````

````{py:property} layers
:canonical: better_robot.viewer.viewer.Viewer.layers
:type: list[better_robot.viewer.layer.Layer]

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer.layers
```

````

````{py:method} add(layer: better_robot.viewer.layer.Layer, group: str = 'scene', *, offset: tuple[float, float, float] | None = None) -> better_robot.viewer.layer.Layer
:canonical: better_robot.viewer.viewer.Viewer.add

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer.add
```

````

````{py:method} add_plot(name: str, series: typing.Any, *, labels: list[str] | None = None, group: str = 'plots') -> better_robot.viewer.layers.plot.PlotPanel
:canonical: better_robot.viewer.viewer.Viewer.add_plot

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer.add_plot
```

````

````{py:method} remove_group(group: str) -> None
:canonical: better_robot.viewer.viewer.Viewer.remove_group

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer.remove_group
```

````

````{py:method} set_visible(layer: better_robot.viewer.layer.Layer, visible: bool) -> None
:canonical: better_robot.viewer.viewer.Viewer.set_visible

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer.set_visible
```

````

````{py:method} show(frame: int | None = None) -> None
:canonical: better_robot.viewer.viewer.Viewer.show

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer.show
```

````

````{py:method} save(path: str | pathlib.Path, *, fps: float | None = None) -> pathlib.Path
:canonical: better_robot.viewer.viewer.Viewer.save

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer.save
```

````

````{py:method} record(path: str | pathlib.Path, *, fps: float | None = None, **options: typing.Any) -> pathlib.Path
:canonical: better_robot.viewer.viewer.Viewer.record

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer.record
```

````

````{py:method} play(block: bool = True) -> None
:canonical: better_robot.viewer.viewer.Viewer.play

```{autodoc2-docstring} better_robot.viewer.viewer.Viewer.play
```

````

`````
