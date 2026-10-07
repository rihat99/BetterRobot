# {py:mod}`better_robot.viewer.timeline`

```{py:module} better_robot.viewer.timeline
```

```{autodoc2-docstring} better_robot.viewer.timeline
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Timeline <better_robot.viewer.timeline.Timeline>`
  - ```{autodoc2-docstring} better_robot.viewer.timeline.Timeline
    :summary:
    ```
````

### API

`````{py:class} Timeline(num_frames: int = 1, fps: float = 30.0, loop: bool = True)
:canonical: better_robot.viewer.timeline.Timeline

```{autodoc2-docstring} better_robot.viewer.timeline.Timeline
```

````{py:property} frame
:canonical: better_robot.viewer.timeline.Timeline.frame
:type: int

```{autodoc2-docstring} better_robot.viewer.timeline.Timeline.frame
```

````

````{py:property} playing
:canonical: better_robot.viewer.timeline.Timeline.playing
:type: bool

```{autodoc2-docstring} better_robot.viewer.timeline.Timeline.playing
```

````

````{py:property} time
:canonical: better_robot.viewer.timeline.Timeline.time
:type: float

```{autodoc2-docstring} better_robot.viewer.timeline.Timeline.time
```

````

````{py:method} on_frame(callback: collections.abc.Callable[[int], None]) -> None
:canonical: better_robot.viewer.timeline.Timeline.on_frame

```{autodoc2-docstring} better_robot.viewer.timeline.Timeline.on_frame
```

````

````{py:method} extend(num_frames: int) -> None
:canonical: better_robot.viewer.timeline.Timeline.extend

```{autodoc2-docstring} better_robot.viewer.timeline.Timeline.extend
```

````

````{py:method} play() -> None
:canonical: better_robot.viewer.timeline.Timeline.play

```{autodoc2-docstring} better_robot.viewer.timeline.Timeline.play
```

````

````{py:method} pause() -> None
:canonical: better_robot.viewer.timeline.Timeline.pause

```{autodoc2-docstring} better_robot.viewer.timeline.Timeline.pause
```

````

````{py:method} toggle() -> None
:canonical: better_robot.viewer.timeline.Timeline.toggle

```{autodoc2-docstring} better_robot.viewer.timeline.Timeline.toggle
```

````

````{py:method} build_gui(gui: typing.Any) -> None
:canonical: better_robot.viewer.timeline.Timeline.build_gui

```{autodoc2-docstring} better_robot.viewer.timeline.Timeline.build_gui
```

````

`````
