# {py:mod}`better_robot.viewer.layers.ground`

```{py:module} better_robot.viewer.layers.ground
```

```{autodoc2-docstring} better_robot.viewer.layers.ground
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GroundLayer <better_robot.viewer.layers.ground.GroundLayer>`
  - ```{autodoc2-docstring} better_robot.viewer.layers.ground.GroundLayer
    :summary:
    ```
````

### API

`````{py:class} GroundLayer(name: str = 'ground', *, size: float = 10.0, height: float = 0.0, plane: str = 'xy', cell_color: better_robot.viewer._math.RGB = (0.78, 0.78, 0.78), section_color: better_robot.viewer._math.RGB = (0.55, 0.55, 0.55))
:canonical: better_robot.viewer.layers.ground.GroundLayer

Bases: {py:obj}`better_robot.viewer.layer.Layer`

```{autodoc2-docstring} better_robot.viewer.layers.ground.GroundLayer
```

````{py:method} build(scene: typing.Any, root: str) -> None
:canonical: better_robot.viewer.layers.ground.GroundLayer.build

````

````{py:method} geometry(frame: int) -> list[better_robot.viewer.layer.Geometry]
:canonical: better_robot.viewer.layers.ground.GroundLayer.geometry

```{autodoc2-docstring} better_robot.viewer.layers.ground.GroundLayer.geometry
```

````

````{py:method} set_frame(frame: int) -> None
:canonical: better_robot.viewer.layers.ground.GroundLayer.set_frame

````

`````
