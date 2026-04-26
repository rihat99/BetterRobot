# {py:mod}`better_robot.viewer.renderers.offscreen_backend`

```{py:module} better_robot.viewer.renderers.offscreen_backend
```

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OffscreenBackend <better_robot.viewer.renderers.offscreen_backend.OffscreenBackend>`
  - ```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend
    :summary:
    ```
````

### API

`````{py:class} OffscreenBackend(*, width: int = 1280, height: int = 720)
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend
```

````{py:attribute} is_interactive
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.is_interactive
:type: bool
:value: >
   False

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.is_interactive
```

````

````{py:attribute} supports_gui
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.supports_gui
:type: bool
:value: >
   False

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.supports_gui
```

````

````{py:method} add_mesh(name: str, vertices: typing.Any, faces: typing.Any, *, rgba: typing.Any = None, parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.add_mesh
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.add_mesh
```

````

````{py:method} add_sphere(name: str, *, radius: float, rgba: typing.Any, parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.add_sphere
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.add_sphere
```

````

````{py:method} add_cylinder(name: str, *, radius: float, length: float, rgba: typing.Any, parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.add_cylinder
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.add_cylinder
```

````

````{py:method} add_capsule(name: str, *, radius: float, length: float, rgba: typing.Any, parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.add_capsule
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.add_capsule
```

````

````{py:method} add_frame(name: str, *, axes_length: float = 0.1) -> None
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.add_frame
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.add_frame
```

````

````{py:method} remove(name: str) -> None
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.remove
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.remove
```

````

````{py:method} set_transform(name: str, pose: typing.Any) -> None
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.set_transform
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.set_transform
```

````

````{py:method} set_visible(name: str, visible: bool) -> None
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.set_visible
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.set_visible
```

````

````{py:method} set_camera(camera: typing.Any) -> None
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.set_camera
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.set_camera
```

````

````{py:method} capture_frame() -> typing.Any
:canonical: better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.capture_frame
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.offscreen_backend.OffscreenBackend.capture_frame
```

````

`````
