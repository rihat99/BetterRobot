# {py:mod}`better_robot.viewer.render_modes.skeleton`

```{py:module} better_robot.viewer.render_modes.skeleton
```

```{autodoc2-docstring} better_robot.viewer.render_modes.skeleton
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SkeletonMode <better_robot.viewer.render_modes.skeleton.SkeletonMode>`
  - ```{autodoc2-docstring} better_robot.viewer.render_modes.skeleton.SkeletonMode
    :summary:
    ```
````

### API

`````{py:class} SkeletonMode(*, joint_radius: float = 0.03, link_radius: float = 0.015, show_root: bool = True, colour_by: typing.Literal[uniform, subtree, depth] = 'uniform')
:canonical: better_robot.viewer.render_modes.skeleton.SkeletonMode

```{autodoc2-docstring} better_robot.viewer.render_modes.skeleton.SkeletonMode
```

````{py:attribute} name
:canonical: better_robot.viewer.render_modes.skeleton.SkeletonMode.name
:value: >
   'Skeleton'

```{autodoc2-docstring} better_robot.viewer.render_modes.skeleton.SkeletonMode.name
```

````

````{py:attribute} description
:canonical: better_robot.viewer.render_modes.skeleton.SkeletonMode.description
:value: >
   'Spheres for joints, cylinders for links'

```{autodoc2-docstring} better_robot.viewer.render_modes.skeleton.SkeletonMode.description
```

````

````{py:method} is_available(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> bool
:canonical: better_robot.viewer.render_modes.skeleton.SkeletonMode.is_available
:classmethod:

```{autodoc2-docstring} better_robot.viewer.render_modes.skeleton.SkeletonMode.is_available
```

````

````{py:method} attach(context: better_robot.viewer.render_modes.base.RenderContext, model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.render_modes.skeleton.SkeletonMode.attach

```{autodoc2-docstring} better_robot.viewer.render_modes.skeleton.SkeletonMode.attach
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.render_modes.skeleton.SkeletonMode.update

```{autodoc2-docstring} better_robot.viewer.render_modes.skeleton.SkeletonMode.update
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.render_modes.skeleton.SkeletonMode.set_visible

```{autodoc2-docstring} better_robot.viewer.render_modes.skeleton.SkeletonMode.set_visible
```

````

````{py:method} detach() -> None
:canonical: better_robot.viewer.render_modes.skeleton.SkeletonMode.detach

```{autodoc2-docstring} better_robot.viewer.render_modes.skeleton.SkeletonMode.detach
```

````

`````
