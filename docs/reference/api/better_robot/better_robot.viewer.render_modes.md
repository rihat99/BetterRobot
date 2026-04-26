# {py:mod}`better_robot.viewer.render_modes`

```{py:module} better_robot.viewer.render_modes
```

```{autodoc2-docstring} better_robot.viewer.render_modes
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

better_robot.viewer.render_modes.base
better_robot.viewer.render_modes.urdf_mesh
better_robot.viewer.render_modes.collision
better_robot.viewer.render_modes.skeleton
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`register_mode <better_robot.viewer.render_modes.register_mode>`
  - ```{autodoc2-docstring} better_robot.viewer.render_modes.register_mode
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MODE_REGISTRY <better_robot.viewer.render_modes.MODE_REGISTRY>`
  - ```{autodoc2-docstring} better_robot.viewer.render_modes.MODE_REGISTRY
    :summary:
    ```
````

### API

````{py:data} MODE_REGISTRY
:canonical: better_robot.viewer.render_modes.MODE_REGISTRY
:type: dict[str, type]
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.render_modes.MODE_REGISTRY
```

````

````{py:function} register_mode(cls: type) -> type
:canonical: better_robot.viewer.render_modes.register_mode

```{autodoc2-docstring} better_robot.viewer.render_modes.register_mode
```
````
