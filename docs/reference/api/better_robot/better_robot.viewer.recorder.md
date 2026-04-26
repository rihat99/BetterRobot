# {py:mod}`better_robot.viewer.recorder`

```{py:module} better_robot.viewer.recorder
```

```{autodoc2-docstring} better_robot.viewer.recorder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VideoRecorder <better_robot.viewer.recorder.VideoRecorder>`
  - ```{autodoc2-docstring} better_robot.viewer.recorder.VideoRecorder
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`render_trajectory <better_robot.viewer.recorder.render_trajectory>`
  - ```{autodoc2-docstring} better_robot.viewer.recorder.render_trajectory
    :summary:
    ```
````

### API

`````{py:class} VideoRecorder(scene: better_robot.viewer.scene.Scene, *, fps: int = 30, resolution: tuple[int, int] = (1280, 720))
:canonical: better_robot.viewer.recorder.VideoRecorder

```{autodoc2-docstring} better_robot.viewer.recorder.VideoRecorder
```

````{py:method} record_trajectory(trajectory: better_robot.tasks.trajectory.Trajectory, path: str, *, camera: Camera | None = None, loop: int = 1, speed: float = 1.0, on_frame: typing.Any = None) -> None
:canonical: better_robot.viewer.recorder.VideoRecorder.record_trajectory
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.recorder.VideoRecorder.record_trajectory
```

````

````{py:method} write_frame() -> None
:canonical: better_robot.viewer.recorder.VideoRecorder.write_frame
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.recorder.VideoRecorder.write_frame
```

````

````{py:method} save(path: str) -> None
:canonical: better_robot.viewer.recorder.VideoRecorder.save
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.recorder.VideoRecorder.save
```

````

````{py:method} start_live(path: str, *, duration: float | None = None) -> None
:canonical: better_robot.viewer.recorder.VideoRecorder.start_live
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.recorder.VideoRecorder.start_live
```

````

````{py:method} stop_live() -> None
:canonical: better_robot.viewer.recorder.VideoRecorder.stop_live
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.recorder.VideoRecorder.stop_live
```

````

`````

````{py:function} render_trajectory(model: better_robot.data_model.model.Model, trajectory: better_robot.tasks.trajectory.Trajectory, path: str, *, modes: typing.Sequence[str] = ('auto', ), overlays: typing.Sequence[str] = ('grid', 'frame_axes'), camera: Camera | None = None, fps: int = 30, resolution: tuple[int, int] = (1280, 720), loop: int = 1, speed: float = 1.0, headless: bool = True, theme: typing.Any = None) -> None
:canonical: better_robot.viewer.recorder.render_trajectory

```{autodoc2-docstring} better_robot.viewer.recorder.render_trajectory
```
````
