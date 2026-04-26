# {py:mod}`better_robot.viewer.overlays.force_vectors`

```{py:module} better_robot.viewer.overlays.force_vectors
```

```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ForceVectorsOverlay <better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay>`
  - ```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay
    :summary:
    ```
````

### API

`````{py:class} ForceVectorsOverlay(contact_joint_names: typing.Sequence[str], *, scale: float = 0.0005, shaft_radius: float = 0.006, head_length: float = 0.03, head_radius: float = 0.015, rgba: tuple[float, float, float, float] = _ARROW_RGBA, visible: bool = True)
:canonical: better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay

```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay
```

````{py:attribute} name
:canonical: better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.name
:value: >
   'Force vectors'

```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.name
```

````

````{py:attribute} description
:canonical: better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.description
:value: >
   'Linear force arrows at contact joints'

```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.description
```

````

````{py:method} is_available(model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> bool
:canonical: better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.is_available
:classmethod:

```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.is_available
```

````

````{py:method} attach(context: better_robot.viewer.render_modes.base.RenderContext, model: better_robot.data_model.model.Model, data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.attach

```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.attach
```

````

````{py:method} update(data: better_robot.data_model.data.Data) -> None
:canonical: better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.update

```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.update
```

````

````{py:method} update_frame(anchor_positions: torch.Tensor, forces: torch.Tensor) -> None
:canonical: better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.update_frame

```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.update_frame
```

````

````{py:method} set_visible(visible: bool) -> None
:canonical: better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.set_visible

```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.set_visible
```

````

````{py:method} detach() -> None
:canonical: better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.detach

```{autodoc2-docstring} better_robot.viewer.overlays.force_vectors.ForceVectorsOverlay.detach
```

````

`````
