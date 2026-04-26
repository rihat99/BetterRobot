# {py:mod}`better_robot.viewer.renderers.testing`

```{py:module} better_robot.viewer.renderers.testing
```

```{autodoc2-docstring} better_robot.viewer.renderers.testing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Call <better_robot.viewer.renderers.testing.Call>`
  - ```{autodoc2-docstring} better_robot.viewer.renderers.testing.Call
    :summary:
    ```
* - {py:obj}`MockBackend <better_robot.viewer.renderers.testing.MockBackend>`
  - ```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend
    :summary:
    ```
````

### API

`````{py:class} Call
:canonical: better_robot.viewer.renderers.testing.Call

```{autodoc2-docstring} better_robot.viewer.renderers.testing.Call
```

````{py:attribute} method
:canonical: better_robot.viewer.renderers.testing.Call.method
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.renderers.testing.Call.method
```

````

````{py:attribute} args
:canonical: better_robot.viewer.renderers.testing.Call.args
:type: tuple
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.renderers.testing.Call.args
```

````

````{py:attribute} kwargs
:canonical: better_robot.viewer.renderers.testing.Call.kwargs
:type: dict
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.renderers.testing.Call.kwargs
```

````

`````

`````{py:class} MockBackend()
:canonical: better_robot.viewer.renderers.testing.MockBackend

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend
```

````{py:attribute} is_interactive
:canonical: better_robot.viewer.renderers.testing.MockBackend.is_interactive
:type: bool
:value: >
   False

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.is_interactive
```

````

````{py:attribute} supports_gui
:canonical: better_robot.viewer.renderers.testing.MockBackend.supports_gui
:type: bool
:value: >
   False

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.supports_gui
```

````

````{py:method} calls_for(method: str) -> list[better_robot.viewer.renderers.testing.Call]
:canonical: better_robot.viewer.renderers.testing.MockBackend.calls_for

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.calls_for
```

````

````{py:method} last_transform(name: str) -> torch.Tensor | None
:canonical: better_robot.viewer.renderers.testing.MockBackend.last_transform

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.last_transform
```

````

````{py:method} reset() -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.reset

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.reset
```

````

````{py:method} add_mesh(name: str, vertices: torch.Tensor, faces: torch.Tensor, *, rgba=(0.8, 0.8, 0.8, 1.0), parent=None) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_mesh

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_mesh
```

````

````{py:method} add_sphere(name: str, *, radius: float, rgba, parent=None) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_sphere

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_sphere
```

````

````{py:method} add_cylinder(name: str, *, radius: float, length: float, rgba, parent=None) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_cylinder

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_cylinder
```

````

````{py:method} add_capsule(name: str, *, radius: float, length: float, rgba, parent=None) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_capsule

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_capsule
```

````

````{py:method} add_frame(name: str, *, axes_length: float = 0.1) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_frame

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_frame
```

````

````{py:method} add_grid(name: str, **kwargs: typing.Any) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_grid

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_grid
```

````

````{py:method} add_mesh_trimesh(name: str, mesh: typing.Any, *, scale: typing.Any = 1.0) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_mesh_trimesh

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_mesh_trimesh
```

````

````{py:method} remove(name: str) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.remove

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.remove
```

````

````{py:method} set_transform(name: str, pose: torch.Tensor) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.set_transform

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.set_transform
```

````

````{py:method} set_visible(name: str, visible: bool) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.set_visible

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.set_visible
```

````

````{py:method} set_camera(camera: typing.Any) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.set_camera

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.set_camera
```

````

````{py:method} capture_frame() -> typing.Any
:canonical: better_robot.viewer.renderers.testing.MockBackend.capture_frame

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.capture_frame
```

````

````{py:method} add_transform_control(name: str, pose: typing.Any, *, scale: float = 0.15, on_update: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_transform_control

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_transform_control
```

````

````{py:method} add_gui_button(label: str, callback: typing.Any) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_gui_button

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_gui_button
```

````

````{py:method} add_gui_slider(label: str, *, min: float, max: float, step: float, value: float, callback: typing.Any) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_gui_slider

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_gui_slider
```

````

````{py:method} add_gui_checkbox(label: str, *, value: bool, callback: typing.Any) -> None
:canonical: better_robot.viewer.renderers.testing.MockBackend.add_gui_checkbox

```{autodoc2-docstring} better_robot.viewer.renderers.testing.MockBackend.add_gui_checkbox
```

````

`````
