# {py:mod}`better_robot.viewer.renderers.viser_backend`

```{py:module} better_robot.viewer.renderers.viser_backend
```

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ViserBackend <better_robot.viewer.renderers.viser_backend.ViserBackend>`
  - ```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend
    :summary:
    ```
````

### API

`````{py:class} ViserBackend(*, port: int = 8080)
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend
```

````{py:attribute} is_interactive
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.is_interactive
:type: bool
:value: >
   True

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.is_interactive
```

````

````{py:attribute} supports_gui
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.supports_gui
:type: bool
:value: >
   True

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.supports_gui
```

````

````{py:method} add_mesh(name: str, vertices: torch.Tensor, faces: torch.Tensor, *, rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0), parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_mesh

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_mesh
```

````

````{py:method} add_mesh_trimesh(name: str, mesh: typing.Any, *, scale: typing.Any = 1.0) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_mesh_trimesh

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_mesh_trimesh
```

````

````{py:method} add_sphere(name: str, *, radius: float, rgba: tuple[float, float, float, float], parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_sphere

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_sphere
```

````

````{py:method} add_cylinder(name: str, *, radius: float, length: float, rgba: tuple[float, float, float, float], parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_cylinder

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_cylinder
```

````

````{py:method} add_capsule(name: str, *, radius: float, length: float, rgba: tuple[float, float, float, float], parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_capsule

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_capsule
```

````

````{py:method} add_frame(name: str, *, axes_length: float = 0.1) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_frame

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_frame
```

````

````{py:method} add_arrow(name: str, *, length: float, shaft_radius: float, head_length: float, head_radius: float, rgba: tuple[float, float, float, float], parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_arrow

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_arrow
```

````

````{py:method} add_grid(name: str, **kwargs: typing.Any) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_grid

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_grid
```

````

````{py:method} remove(name: str) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.remove

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.remove
```

````

````{py:method} set_transform(name: str, pose: torch.Tensor) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.set_transform

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.set_transform
```

````

````{py:method} set_visible(name: str, visible: bool) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.set_visible

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.set_visible
```

````

````{py:method} set_camera(camera: typing.Any) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.set_camera
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.set_camera
```

````

````{py:method} capture_frame() -> np.ndarray
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.capture_frame
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.capture_frame
```

````

````{py:method} add_gui_button(label: str, callback: typing.Any) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_gui_button

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_gui_button
```

````

````{py:method} add_gui_slider(label: str, *, min: float, max: float, step: float, value: float, callback: typing.Any) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_gui_slider

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_gui_slider
```

````

````{py:method} add_gui_checkbox(label: str, *, value: bool, callback: typing.Any) -> None
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_gui_checkbox

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_gui_checkbox
```

````

````{py:method} add_transform_control(name: str, pose: torch.Tensor, *, scale: float = 0.15, on_update: Callable[[torch.Tensor], None] | None = None) -> typing.Any
:canonical: better_robot.viewer.renderers.viser_backend.ViserBackend.add_transform_control

```{autodoc2-docstring} better_robot.viewer.renderers.viser_backend.ViserBackend.add_transform_control
```

````

`````
