# {py:mod}`better_robot.viewer.render`

```{py:module} better_robot.viewer.render
```

```{autodoc2-docstring} better_robot.viewer.render
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

better_robot.viewer.render.rasterizer
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`render_frame <better_robot.viewer.render.render_frame>`
  - ```{autodoc2-docstring} better_robot.viewer.render.render_frame
    :summary:
    ```
* - {py:obj}`frame_camera <better_robot.viewer.render.frame_camera>`
  - ```{autodoc2-docstring} better_robot.viewer.render.frame_camera
    :summary:
    ```
* - {py:obj}`render_frames <better_robot.viewer.render.render_frames>`
  - ```{autodoc2-docstring} better_robot.viewer.render.render_frames
    :summary:
    ```
* - {py:obj}`record <better_robot.viewer.render.record>`
  - ```{autodoc2-docstring} better_robot.viewer.render.record
    :summary:
    ```
````

### API

````{py:function} render_frame(layers: collections.abc.Sequence[better_robot.viewer.layer.Layer], frame: int, camera: better_robot.viewer.render.rasterizer.Camera, *, size: tuple[int, int] = (960, 540), background: tuple[float, float, float] = (1.0, 1.0, 1.0), device: torch.device | str | None = None) -> numpy.ndarray
:canonical: better_robot.viewer.render.render_frame

```{autodoc2-docstring} better_robot.viewer.render.render_frame
```
````

````{py:function} frame_camera(layers: collections.abc.Sequence[better_robot.viewer.layer.Layer], frames: collections.abc.Iterable[int] = (0, ), **options: float) -> better_robot.viewer.render.rasterizer.Camera
:canonical: better_robot.viewer.render.frame_camera

```{autodoc2-docstring} better_robot.viewer.render.frame_camera
```
````

````{py:function} render_frames(layers: collections.abc.Sequence[better_robot.viewer.layer.Layer], frames: collections.abc.Iterable[int] | None = None, *, camera: better_robot.viewer.render.rasterizer.Camera | None = None, size: tuple[int, int] = (960, 540), background: tuple[float, float, float] = (1.0, 1.0, 1.0), device: torch.device | str | None = None) -> numpy.ndarray
:canonical: better_robot.viewer.render.render_frames

```{autodoc2-docstring} better_robot.viewer.render.render_frames
```
````

````{py:function} record(layers: collections.abc.Sequence[better_robot.viewer.layer.Layer], path: str | pathlib.Path, *, fps: float = 30.0, frames: collections.abc.Iterable[int] | None = None, camera: better_robot.viewer.render.rasterizer.Camera | None = None, size: tuple[int, int] = (960, 540), background: tuple[float, float, float] = (1.0, 1.0, 1.0), device: torch.device | str | None = None) -> pathlib.Path
:canonical: better_robot.viewer.render.record

```{autodoc2-docstring} better_robot.viewer.render.record
```
````
