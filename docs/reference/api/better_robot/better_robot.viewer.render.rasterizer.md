# {py:mod}`better_robot.viewer.render.rasterizer`

```{py:module} better_robot.viewer.render.rasterizer
```

```{autodoc2-docstring} better_robot.viewer.render.rasterizer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Camera <better_robot.viewer.render.rasterizer.Camera>`
  - ```{autodoc2-docstring} better_robot.viewer.render.rasterizer.Camera
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`rasterize <better_robot.viewer.render.rasterizer.rasterize>`
  - ```{autodoc2-docstring} better_robot.viewer.render.rasterizer.rasterize
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TILE <better_robot.viewer.render.rasterizer.TILE>`
  - ```{autodoc2-docstring} better_robot.viewer.render.rasterizer.TILE
    :summary:
    ```
````

### API

````{py:data} TILE
:canonical: better_robot.viewer.render.rasterizer.TILE
:value: >
   16

```{autodoc2-docstring} better_robot.viewer.render.rasterizer.TILE
```

````

`````{py:class} Camera
:canonical: better_robot.viewer.render.rasterizer.Camera

```{autodoc2-docstring} better_robot.viewer.render.rasterizer.Camera
```

````{py:attribute} position
:canonical: better_robot.viewer.render.rasterizer.Camera.position
:type: tuple[float, float, float]
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.render.rasterizer.Camera.position
```

````

````{py:attribute} target
:canonical: better_robot.viewer.render.rasterizer.Camera.target
:type: tuple[float, float, float]
:value: >
   (0.0, 0.0, 0.0)

```{autodoc2-docstring} better_robot.viewer.render.rasterizer.Camera.target
```

````

````{py:attribute} up
:canonical: better_robot.viewer.render.rasterizer.Camera.up
:type: tuple[float, float, float]
:value: >
   (0.0, 0.0, 1.0)

```{autodoc2-docstring} better_robot.viewer.render.rasterizer.Camera.up
```

````

````{py:attribute} fov
:canonical: better_robot.viewer.render.rasterizer.Camera.fov
:type: float
:value: >
   'radians(...)'

```{autodoc2-docstring} better_robot.viewer.render.rasterizer.Camera.fov
```

````

````{py:method} view_matrix() -> numpy.ndarray
:canonical: better_robot.viewer.render.rasterizer.Camera.view_matrix

```{autodoc2-docstring} better_robot.viewer.render.rasterizer.Camera.view_matrix
```

````

````{py:method} framing(points: numpy.ndarray, *, azimuth: float = math.radians(35.0), elevation: float = math.radians(25.0), fov: float = math.radians(45.0), margin: float = 1.3) -> better_robot.viewer.render.rasterizer.Camera
:canonical: better_robot.viewer.render.rasterizer.Camera.framing
:classmethod:

```{autodoc2-docstring} better_robot.viewer.render.rasterizer.Camera.framing
```

````

`````

````{py:function} rasterize(vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor, camera: better_robot.viewer.render.rasterizer.Camera, size: tuple[int, int], *, background: tuple[float, float, float] = (1.0, 1.0, 1.0), light_direction: tuple[float, float, float] = (-0.4, -0.6, -1.0)) -> torch.Tensor
:canonical: better_robot.viewer.render.rasterizer.rasterize

```{autodoc2-docstring} better_robot.viewer.render.rasterizer.rasterize
```
````
