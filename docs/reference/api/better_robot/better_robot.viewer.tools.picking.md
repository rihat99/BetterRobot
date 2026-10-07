# {py:mod}`better_robot.viewer.tools.picking`

```{py:module} better_robot.viewer.tools.picking
```

```{autodoc2-docstring} better_robot.viewer.tools.picking
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CameraFrustum <better_robot.viewer.tools.picking.CameraFrustum>`
  - ```{autodoc2-docstring} better_robot.viewer.tools.picking.CameraFrustum
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ray_hit <better_robot.viewer.tools.picking.ray_hit>`
  - ```{autodoc2-docstring} better_robot.viewer.tools.picking.ray_hit
    :summary:
    ```
* - {py:obj}`vertices_in_sphere <better_robot.viewer.tools.picking.vertices_in_sphere>`
  - ```{autodoc2-docstring} better_robot.viewer.tools.picking.vertices_in_sphere
    :summary:
    ```
* - {py:obj}`vertices_in_rect <better_robot.viewer.tools.picking.vertices_in_rect>`
  - ```{autodoc2-docstring} better_robot.viewer.tools.picking.vertices_in_rect
    :summary:
    ```
* - {py:obj}`project <better_robot.viewer.tools.picking.project>`
  - ```{autodoc2-docstring} better_robot.viewer.tools.picking.project
    :summary:
    ```
````

### API

`````{py:class} CameraFrustum
:canonical: better_robot.viewer.tools.picking.CameraFrustum

```{autodoc2-docstring} better_robot.viewer.tools.picking.CameraFrustum
```

````{py:attribute} position
:canonical: better_robot.viewer.tools.picking.CameraFrustum.position
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.tools.picking.CameraFrustum.position
```

````

````{py:attribute} wxyz
:canonical: better_robot.viewer.tools.picking.CameraFrustum.wxyz
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.tools.picking.CameraFrustum.wxyz
```

````

````{py:attribute} fov
:canonical: better_robot.viewer.tools.picking.CameraFrustum.fov
:type: float
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.tools.picking.CameraFrustum.fov
```

````

````{py:attribute} aspect
:canonical: better_robot.viewer.tools.picking.CameraFrustum.aspect
:type: float
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.tools.picking.CameraFrustum.aspect
```

````

````{py:method} from_camera(camera: typing.Any) -> better_robot.viewer.tools.picking.CameraFrustum
:canonical: better_robot.viewer.tools.picking.CameraFrustum.from_camera
:classmethod:

```{autodoc2-docstring} better_robot.viewer.tools.picking.CameraFrustum.from_camera
```

````

`````

````{py:function} ray_hit(origin: numpy.ndarray, direction: numpy.ndarray, vertices: numpy.ndarray, faces: numpy.ndarray) -> tuple[numpy.ndarray, int] | None
:canonical: better_robot.viewer.tools.picking.ray_hit

```{autodoc2-docstring} better_robot.viewer.tools.picking.ray_hit
```
````

````{py:function} vertices_in_sphere(vertices: numpy.ndarray, center: numpy.ndarray, radius: float) -> numpy.ndarray
:canonical: better_robot.viewer.tools.picking.vertices_in_sphere

```{autodoc2-docstring} better_robot.viewer.tools.picking.vertices_in_sphere
```
````

````{py:function} vertices_in_rect(vertices: numpy.ndarray, camera: better_robot.viewer.tools.picking.CameraFrustum, rect_min: numpy.ndarray, rect_max: numpy.ndarray) -> numpy.ndarray
:canonical: better_robot.viewer.tools.picking.vertices_in_rect

```{autodoc2-docstring} better_robot.viewer.tools.picking.vertices_in_rect
```
````

````{py:function} project(vertices: numpy.ndarray, camera: better_robot.viewer.tools.picking.CameraFrustum) -> tuple[numpy.ndarray, numpy.ndarray]
:canonical: better_robot.viewer.tools.picking.project

```{autodoc2-docstring} better_robot.viewer.tools.picking.project
```
````
