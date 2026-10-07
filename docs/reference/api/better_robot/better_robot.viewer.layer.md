# {py:mod}`better_robot.viewer.layer`

```{py:module} better_robot.viewer.layer
```

```{autodoc2-docstring} better_robot.viewer.layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Geometry <better_robot.viewer.layer.Geometry>`
  - ```{autodoc2-docstring} better_robot.viewer.layer.Geometry
    :summary:
    ```
* - {py:obj}`Layer <better_robot.viewer.layer.Layer>`
  - ```{autodoc2-docstring} better_robot.viewer.layer.Layer
    :summary:
    ```
````

### API

`````{py:class} Geometry
:canonical: better_robot.viewer.layer.Geometry

```{autodoc2-docstring} better_robot.viewer.layer.Geometry
```

````{py:attribute} vertices
:canonical: better_robot.viewer.layer.Geometry.vertices
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.layer.Geometry.vertices
```

````

````{py:attribute} faces
:canonical: better_robot.viewer.layer.Geometry.faces
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.layer.Geometry.faces
```

````

````{py:attribute} colors
:canonical: better_robot.viewer.layer.Geometry.colors
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.layer.Geometry.colors
```

````

`````

`````{py:class} Layer(name: str, num_frames: int = 1)
:canonical: better_robot.viewer.layer.Layer

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} better_robot.viewer.layer.Layer
```

````{py:method} build(scene: typing.Any, root: str) -> None
:canonical: better_robot.viewer.layer.Layer.build
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.layer.Layer.build
```

````

````{py:method} set_frame(frame: int) -> None
:canonical: better_robot.viewer.layer.Layer.set_frame
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.layer.Layer.set_frame
```

````

````{py:method} build_gui(gui: typing.Any) -> None
:canonical: better_robot.viewer.layer.Layer.build_gui

```{autodoc2-docstring} better_robot.viewer.layer.Layer.build_gui
```

````

````{py:method} geometry(frame: int) -> list[better_robot.viewer.layer.Geometry]
:canonical: better_robot.viewer.layer.Layer.geometry

```{autodoc2-docstring} better_robot.viewer.layer.Layer.geometry
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.layer.Layer.set_visible

```{autodoc2-docstring} better_robot.viewer.layer.Layer.set_visible
```

````

````{py:method} dispose() -> None
:canonical: better_robot.viewer.layer.Layer.dispose

```{autodoc2-docstring} better_robot.viewer.layer.Layer.dispose
```

````

`````
