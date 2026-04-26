# {py:mod}`better_robot.viewer.camera`

```{py:module} better_robot.viewer.camera
```

```{autodoc2-docstring} better_robot.viewer.camera
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Camera <better_robot.viewer.camera.Camera>`
  - ```{autodoc2-docstring} better_robot.viewer.camera.Camera
    :summary:
    ```
* - {py:obj}`CameraPath <better_robot.viewer.camera.CameraPath>`
  - ```{autodoc2-docstring} better_robot.viewer.camera.CameraPath
    :summary:
    ```
````

### API

`````{py:class} Camera
:canonical: better_robot.viewer.camera.Camera

```{autodoc2-docstring} better_robot.viewer.camera.Camera
```

````{py:attribute} position
:canonical: better_robot.viewer.camera.Camera.position
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.camera.Camera.position
```

````

````{py:attribute} look_at
:canonical: better_robot.viewer.camera.Camera.look_at
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.camera.Camera.look_at
```

````

````{py:attribute} up
:canonical: better_robot.viewer.camera.Camera.up
:type: tuple[float, float, float]
:value: >
   (0.0, 0.0, 1.0)

```{autodoc2-docstring} better_robot.viewer.camera.Camera.up
```

````

````{py:attribute} fov_deg
:canonical: better_robot.viewer.camera.Camera.fov_deg
:type: float
:value: >
   50.0

```{autodoc2-docstring} better_robot.viewer.camera.Camera.fov_deg
```

````

````{py:attribute} near
:canonical: better_robot.viewer.camera.Camera.near
:type: float
:value: >
   0.01

```{autodoc2-docstring} better_robot.viewer.camera.Camera.near
```

````

````{py:attribute} far
:canonical: better_robot.viewer.camera.Camera.far
:type: float
:value: >
   100.0

```{autodoc2-docstring} better_robot.viewer.camera.Camera.far
```

````

`````

`````{py:class} CameraPath(*args: typing.Any, **kwargs: typing.Any)
:canonical: better_robot.viewer.camera.CameraPath

```{autodoc2-docstring} better_robot.viewer.camera.CameraPath
```

````{py:method} at(k: int) -> better_robot.viewer.camera.Camera
:canonical: better_robot.viewer.camera.CameraPath.at
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.camera.CameraPath.at
```

````

````{py:method} orbit(*, center: torch.Tensor, radius: float, n_frames: int, axis: tuple[float, float, float] = (0.0, 0.0, 1.0), elevation_deg: float = 20.0) -> better_robot.viewer.camera.CameraPath
:canonical: better_robot.viewer.camera.CameraPath.orbit
:abstractmethod:
:classmethod:

```{autodoc2-docstring} better_robot.viewer.camera.CameraPath.orbit
```

````

````{py:method} follow_frame(model: better_robot.data_model.model.Model, trajectory: better_robot.tasks.trajectory.Trajectory, *, frame: str, offset: torch.Tensor) -> better_robot.viewer.camera.CameraPath
:canonical: better_robot.viewer.camera.CameraPath.follow_frame
:abstractmethod:
:classmethod:

```{autodoc2-docstring} better_robot.viewer.camera.CameraPath.follow_frame
```

````

````{py:method} static(camera: better_robot.viewer.camera.Camera, n_frames: int) -> better_robot.viewer.camera.CameraPath
:canonical: better_robot.viewer.camera.CameraPath.static
:abstractmethod:
:classmethod:

```{autodoc2-docstring} better_robot.viewer.camera.CameraPath.static
```

````

`````
