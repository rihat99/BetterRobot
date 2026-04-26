# {py:mod}`better_robot.viewer.visualizer`

```{py:module} better_robot.viewer.visualizer
```

```{autodoc2-docstring} better_robot.viewer.visualizer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Visualizer <better_robot.viewer.visualizer.Visualizer>`
  - ```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer
    :summary:
    ```
````

### API

`````{py:class} Visualizer(model: better_robot.data_model.model.Model, *, port: int = 8080, theme: Theme | None = None)
:canonical: better_robot.viewer.visualizer.Visualizer

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer
```

````{py:method} show(*, block: bool = True) -> None
:canonical: better_robot.viewer.visualizer.Visualizer.show

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.show
```

````

````{py:method} close() -> None
:canonical: better_robot.viewer.visualizer.Visualizer.close

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.close
```

````

````{py:property} last_q
:canonical: better_robot.viewer.visualizer.Visualizer.last_q
:type: torch.Tensor

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.last_q
```

````

````{py:method} update(q_or_data: torch.Tensor | Data) -> None
:canonical: better_robot.viewer.visualizer.Visualizer.update

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.update
```

````

````{py:method} add_trajectory(trajectory: better_robot.tasks.trajectory.Trajectory) -> better_robot.viewer.trajectory_player.TrajectoryPlayer
:canonical: better_robot.viewer.visualizer.Visualizer.add_trajectory

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.add_trajectory
```

````

````{py:method} current_player() -> TrajectoryPlayer | None
:canonical: better_robot.viewer.visualizer.Visualizer.current_player

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.current_player
```

````

````{py:method} add_ik_result(result: better_robot.tasks.ik.IKResult) -> None
:canonical: better_robot.viewer.visualizer.Visualizer.add_ik_result

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.add_ik_result
```

````

````{py:method} add_ik_targets(targets: dict[str, torch.Tensor], *, on_change: Callable[[dict[str, torch.Tensor]], None] | None = None, scale: float = 0.15) -> better_robot.viewer.overlays.targets.TargetsOverlay
:canonical: better_robot.viewer.visualizer.Visualizer.add_ik_targets

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.add_ik_targets
```

````

````{py:method} record(*args: object, **kwargs: object) -> None
:canonical: better_robot.viewer.visualizer.Visualizer.record
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.record
```

````

````{py:method} add_robot(*args: object, **kwargs: object) -> None
:canonical: better_robot.viewer.visualizer.Visualizer.add_robot
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.add_robot
```

````

````{py:method} scene(name: str | None = None) -> better_robot.viewer.scene.Scene
:canonical: better_robot.viewer.visualizer.Visualizer.scene

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.scene
```

````

````{py:method} set_batch_index(idx: int) -> None
:canonical: better_robot.viewer.visualizer.Visualizer.set_batch_index
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.visualizer.Visualizer.set_batch_index
```

````

`````
