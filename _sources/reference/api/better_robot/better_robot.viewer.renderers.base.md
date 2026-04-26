# {py:mod}`better_robot.viewer.renderers.base`

```{py:module} better_robot.viewer.renderers.base
```

```{autodoc2-docstring} better_robot.viewer.renderers.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RendererBackend <better_robot.viewer.renderers.base.RendererBackend>`
  - ```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend
    :summary:
    ```
````

### API

`````{py:class} RendererBackend
:canonical: better_robot.viewer.renderers.base.RendererBackend

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend
```

````{py:attribute} is_interactive
:canonical: better_robot.viewer.renderers.base.RendererBackend.is_interactive
:type: bool
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.is_interactive
```

````

````{py:attribute} supports_gui
:canonical: better_robot.viewer.renderers.base.RendererBackend.supports_gui
:type: bool
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.supports_gui
```

````

````{py:method} add_mesh(name: str, vertices: torch.Tensor, faces: torch.Tensor, *, rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0), parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.add_mesh

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.add_mesh
```

````

````{py:method} add_sphere(name: str, *, radius: float, rgba: tuple[float, float, float, float], parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.add_sphere

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.add_sphere
```

````

````{py:method} add_cylinder(name: str, *, radius: float, length: float, rgba: tuple[float, float, float, float], parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.add_cylinder

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.add_cylinder
```

````

````{py:method} add_capsule(name: str, *, radius: float, length: float, rgba: tuple[float, float, float, float], parent: typing.Any = None) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.add_capsule

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.add_capsule
```

````

````{py:method} add_frame(name: str, *, axes_length: float = 0.1) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.add_frame

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.add_frame
```

````

````{py:method} remove(name: str) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.remove

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.remove
```

````

````{py:method} set_transform(name: str, pose: torch.Tensor) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.set_transform

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.set_transform
```

````

````{py:method} set_visible(name: str, visible: bool) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.set_visible

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.set_visible
```

````

````{py:method} set_camera(camera: typing.Any) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.set_camera

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.set_camera
```

````

````{py:method} capture_frame() -> np.ndarray
:canonical: better_robot.viewer.renderers.base.RendererBackend.capture_frame

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.capture_frame
```

````

````{py:method} add_gui_button(label: str, callback: typing.Any) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.add_gui_button

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.add_gui_button
```

````

````{py:method} add_gui_slider(label: str, *, min: float, max: float, step: float, value: float, callback: typing.Any) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.add_gui_slider

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.add_gui_slider
```

````

````{py:method} add_gui_checkbox(label: str, *, value: bool, callback: typing.Any) -> None
:canonical: better_robot.viewer.renderers.base.RendererBackend.add_gui_checkbox

```{autodoc2-docstring} better_robot.viewer.renderers.base.RendererBackend.add_gui_checkbox
```

````

`````
