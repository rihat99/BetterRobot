# {py:mod}`better_robot.viewer.render_modes.urdf_mesh`

```{py:module} better_robot.viewer.render_modes.urdf_mesh
```

```{autodoc2-docstring} better_robot.viewer.render_modes.urdf_mesh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`URDFMeshMode <better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode>`
  - ```{autodoc2-docstring} better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode
    :summary:
    ```
````

### API

`````{py:class} URDFMeshMode(*, alpha: float = 1.0, resolver: AssetResolver | None = None)
:canonical: better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode

```{autodoc2-docstring} better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode
```

````{py:attribute} name
:canonical: better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.name
:value: >
   'URDF mesh'

```{autodoc2-docstring} better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.name
```

````

````{py:attribute} description
:canonical: better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.description
:value: >
   'Visual meshes from the URDF / MJCF'

```{autodoc2-docstring} better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.description
```

````

````{py:method} is_available(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> bool
:canonical: better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.is_available
:classmethod:

```{autodoc2-docstring} better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.is_available
```

````

````{py:method} attach(context: better_robot.viewer.render_modes.base.RenderContext, model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.attach

```{autodoc2-docstring} better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.attach
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.update

```{autodoc2-docstring} better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.update
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.set_visible

```{autodoc2-docstring} better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.set_visible
```

````

````{py:method} detach() -> None
:canonical: better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.detach

```{autodoc2-docstring} better_robot.viewer.render_modes.urdf_mesh.URDFMeshMode.detach
```

````

`````
