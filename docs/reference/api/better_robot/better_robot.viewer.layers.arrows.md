# {py:mod}`better_robot.viewer.layers.arrows`

```{py:module} better_robot.viewer.layers.arrows
```

```{autodoc2-docstring} better_robot.viewer.layers.arrows
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ArrowsLayer <better_robot.viewer.layers.arrows.ArrowsLayer>`
  - ```{autodoc2-docstring} better_robot.viewer.layers.arrows.ArrowsLayer
    :summary:
    ```
````

### API

`````{py:class} ArrowsLayer(name: str, origins: numpy.ndarray, vectors: numpy.ndarray, *, scale: float = 1.0, colors: numpy.ndarray | better_robot.viewer._math.RGB = (0.9, 0.25, 0.2), head_size: float = 0.03, line_width: float = 3.0)
:canonical: better_robot.viewer.layers.arrows.ArrowsLayer

Bases: {py:obj}`better_robot.viewer.layer.Layer`

```{autodoc2-docstring} better_robot.viewer.layers.arrows.ArrowsLayer
```

````{py:method} build(scene: typing.Any, root: str) -> None
:canonical: better_robot.viewer.layers.arrows.ArrowsLayer.build

````

````{py:method} set_frame(frame: int) -> None
:canonical: better_robot.viewer.layers.arrows.ArrowsLayer.set_frame

````

````{py:method} geometry(frame: int) -> list[better_robot.viewer.layer.Geometry]
:canonical: better_robot.viewer.layers.arrows.ArrowsLayer.geometry

````

`````
