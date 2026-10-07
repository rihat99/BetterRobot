# {py:mod}`better_robot.viewer.layers.skeleton`

```{py:module} better_robot.viewer.layers.skeleton
```

```{autodoc2-docstring} better_robot.viewer.layers.skeleton
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SkeletonLayer <better_robot.viewer.layers.skeleton.SkeletonLayer>`
  - ```{autodoc2-docstring} better_robot.viewer.layers.skeleton.SkeletonLayer
    :summary:
    ```
````

### API

`````{py:class} SkeletonLayer(name: str, joints: numpy.ndarray, parents: numpy.ndarray | list[int], *, radius: float | numpy.ndarray = 0.02, bone_radius: float | None = None, colors: numpy.ndarray | better_robot.viewer._math.RGB = (0.2, 0.6, 1.0), bone_color: better_robot.viewer._math.RGB = (0.72, 0.72, 0.72))
:canonical: better_robot.viewer.layers.skeleton.SkeletonLayer

Bases: {py:obj}`better_robot.viewer.layer.Layer`

```{autodoc2-docstring} better_robot.viewer.layers.skeleton.SkeletonLayer
```

````{py:method} build(scene: typing.Any, root: str) -> None
:canonical: better_robot.viewer.layers.skeleton.SkeletonLayer.build

````

````{py:method} set_frame(frame: int) -> None
:canonical: better_robot.viewer.layers.skeleton.SkeletonLayer.set_frame

````

````{py:method} geometry(frame: int) -> list[better_robot.viewer.layer.Geometry]
:canonical: better_robot.viewer.layers.skeleton.SkeletonLayer.geometry

````

`````
