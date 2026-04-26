# {py:mod}`better_robot.viewer.trajectory_player`

```{py:module} better_robot.viewer.trajectory_player
```

```{autodoc2-docstring} better_robot.viewer.trajectory_player
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrajectoryPlayer <better_robot.viewer.trajectory_player.TrajectoryPlayer>`
  - ```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer
    :summary:
    ```
````

### API

`````{py:class} TrajectoryPlayer(scene: better_robot.viewer.scene.Scene, trajectory: better_robot.tasks.trajectory.Trajectory)
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer
```

````{py:property} horizon
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.horizon
:type: int

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.horizon
```

````

````{py:method} show_frame(k: int) -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.show_frame

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.show_frame
```

````

````{py:method} play(*, fps: float = 30.0) -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.play

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.play
```

````

````{py:method} seek(t: float) -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.seek
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.seek
```

````

````{py:method} seek_frame(k: int) -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.seek_frame

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.seek_frame
```

````

````{py:method} step(dt: float) -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.step
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.step
```

````

````{py:method} pause() -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.pause
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.pause
```

````

````{py:method} set_speed(speed: float) -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.set_speed
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.set_speed
```

````

````{py:method} set_loop(loop: bool) -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.set_loop
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.set_loop
```

````

````{py:method} set_ghost(every: int | None) -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.set_ghost
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.set_ghost
```

````

````{py:method} set_trace(frame_name: str | None) -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.set_trace
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.set_trace
```

````

````{py:method} set_batch_index(idx: int) -> None
:canonical: better_robot.viewer.trajectory_player.TrajectoryPlayer.set_batch_index
:abstractmethod:

```{autodoc2-docstring} better_robot.viewer.trajectory_player.TrajectoryPlayer.set_batch_index
```

````

`````
