# {py:mod}`better_robot.viewer.overlays.targets`

```{py:module} better_robot.viewer.overlays.targets
```

```{autodoc2-docstring} better_robot.viewer.overlays.targets
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TargetsOverlay <better_robot.viewer.overlays.targets.TargetsOverlay>`
  - ```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay
    :summary:
    ```
````

### API

`````{py:class} TargetsOverlay(targets: dict[str, torch.Tensor], *, on_change: typing.Callable[[dict[str, torch.Tensor]], None] | None = None, scale: float = 0.15)
:canonical: better_robot.viewer.overlays.targets.TargetsOverlay

```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay
```

````{py:attribute} name
:canonical: better_robot.viewer.overlays.targets.TargetsOverlay.name
:value: >
   'IK targets'

```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay.name
```

````

````{py:attribute} description
:canonical: better_robot.viewer.overlays.targets.TargetsOverlay.description
:value: >
   'Draggable SE(3) IK target gizmos'

```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay.description
```

````

````{py:property} targets
:canonical: better_robot.viewer.overlays.targets.TargetsOverlay.targets
:type: dict[str, torch.Tensor]

```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay.targets
```

````

````{py:method} live_targets() -> dict[str, torch.Tensor]
:canonical: better_robot.viewer.overlays.targets.TargetsOverlay.live_targets

```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay.live_targets
```

````

````{py:method} is_available(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> bool
:canonical: better_robot.viewer.overlays.targets.TargetsOverlay.is_available
:classmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay.is_available
```

````

````{py:method} attach(context: better_robot.viewer.render_modes.base.RenderContext, model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.targets.TargetsOverlay.attach

```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay.attach
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.targets.TargetsOverlay.update

```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay.update
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.overlays.targets.TargetsOverlay.set_visible

```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay.set_visible
```

````

````{py:method} detach() -> None
:canonical: better_robot.viewer.overlays.targets.TargetsOverlay.detach

```{autodoc2-docstring} better_robot.viewer.overlays.targets.TargetsOverlay.detach
```

````

`````
