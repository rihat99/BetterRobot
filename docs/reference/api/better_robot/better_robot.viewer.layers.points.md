# {py:mod}`better_robot.viewer.layers.points`

```{py:module} better_robot.viewer.layers.points
```

```{autodoc2-docstring} better_robot.viewer.layers.points
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PointsLayer <better_robot.viewer.layers.points.PointsLayer>`
  - ```{autodoc2-docstring} better_robot.viewer.layers.points.PointsLayer
    :summary:
    ```
````

### API

`````{py:class} PointsLayer(name: str, points: numpy.ndarray, *, colors: numpy.ndarray | better_robot.viewer._math.RGB = (0.25, 0.25, 0.25), point_size: float = 0.01, point_shape: str = 'circle')
:canonical: better_robot.viewer.layers.points.PointsLayer

Bases: {py:obj}`better_robot.viewer.layer.Layer`

```{autodoc2-docstring} better_robot.viewer.layers.points.PointsLayer
```

````{py:method} build(scene: typing.Any, root: str) -> None
:canonical: better_robot.viewer.layers.points.PointsLayer.build

````

````{py:method} geometry(frame: int) -> list[better_robot.viewer.layer.Geometry]
:canonical: better_robot.viewer.layers.points.PointsLayer.geometry

````

````{py:method} set_frame(frame: int) -> None
:canonical: better_robot.viewer.layers.points.PointsLayer.set_frame

````

`````
