# {py:mod}`better_robot.viewer.layers.frames`

```{py:module} better_robot.viewer.layers.frames
```

```{autodoc2-docstring} better_robot.viewer.layers.frames
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FramesLayer <better_robot.viewer.layers.frames.FramesLayer>`
  - ```{autodoc2-docstring} better_robot.viewer.layers.frames.FramesLayer
    :summary:
    ```
````

### API

`````{py:class} FramesLayer(name: str, poses: numpy.ndarray, *, axes_length: float = 0.1, axes_radius: float = 0.004)
:canonical: better_robot.viewer.layers.frames.FramesLayer

Bases: {py:obj}`better_robot.viewer.layer.Layer`

```{autodoc2-docstring} better_robot.viewer.layers.frames.FramesLayer
```

````{py:method} build(scene: typing.Any, root: str) -> None
:canonical: better_robot.viewer.layers.frames.FramesLayer.build

````

````{py:method} geometry(frame: int) -> list[better_robot.viewer.layer.Geometry]
:canonical: better_robot.viewer.layers.frames.FramesLayer.geometry

````

````{py:method} set_frame(frame: int) -> None
:canonical: better_robot.viewer.layers.frames.FramesLayer.set_frame

````

`````
