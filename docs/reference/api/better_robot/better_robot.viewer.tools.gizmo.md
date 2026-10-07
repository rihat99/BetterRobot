# {py:mod}`better_robot.viewer.tools.gizmo`

```{py:module} better_robot.viewer.tools.gizmo
```

```{autodoc2-docstring} better_robot.viewer.tools.gizmo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PoseGizmo <better_robot.viewer.tools.gizmo.PoseGizmo>`
  - ```{autodoc2-docstring} better_robot.viewer.tools.gizmo.PoseGizmo
    :summary:
    ```
````

### API

`````{py:class} PoseGizmo(viewer: typing.Any, name: str, pose: numpy.ndarray, *, on_change: collections.abc.Callable[[numpy.ndarray], None] | None = None, scale: float = 0.15)
:canonical: better_robot.viewer.tools.gizmo.PoseGizmo

```{autodoc2-docstring} better_robot.viewer.tools.gizmo.PoseGizmo
```

````{py:property} pose
:canonical: better_robot.viewer.tools.gizmo.PoseGizmo.pose
:type: numpy.ndarray

```{autodoc2-docstring} better_robot.viewer.tools.gizmo.PoseGizmo.pose
```

````

````{py:method} on_change(callback: collections.abc.Callable[[numpy.ndarray], None]) -> None
:canonical: better_robot.viewer.tools.gizmo.PoseGizmo.on_change

```{autodoc2-docstring} better_robot.viewer.tools.gizmo.PoseGizmo.on_change
```

````

````{py:method} remove() -> None
:canonical: better_robot.viewer.tools.gizmo.PoseGizmo.remove

```{autodoc2-docstring} better_robot.viewer.tools.gizmo.PoseGizmo.remove
```

````

`````
