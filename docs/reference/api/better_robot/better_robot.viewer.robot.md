# {py:mod}`better_robot.viewer.robot`

```{py:module} better_robot.viewer.robot
```

```{autodoc2-docstring} better_robot.viewer.robot
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RobotView <better_robot.viewer.robot.RobotView>`
  - ```{autodoc2-docstring} better_robot.viewer.robot.RobotView
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`robot_layers <better_robot.viewer.robot.robot_layers>`
  - ```{autodoc2-docstring} better_robot.viewer.robot.robot_layers
    :summary:
    ```
* - {py:obj}`robot_skeleton <better_robot.viewer.robot.robot_skeleton>`
  - ```{autodoc2-docstring} better_robot.viewer.robot.robot_skeleton
    :summary:
    ```
* - {py:obj}`robot_meshes <better_robot.viewer.robot.robot_meshes>`
  - ```{autodoc2-docstring} better_robot.viewer.robot.robot_meshes
    :summary:
    ```
````

### API

`````{py:class} RobotView(model: better_robot.data_model.model.Model, q: torch.Tensor, *, name: str = 'robot', visual: bool = True, collision: bool = False, skeleton: bool = False, frames: collections.abc.Sequence[str | int] = (), tau: torch.Tensor | None = None, spheres: better_robot.collision.robot_spheres.RobotSpheres | None = None, theme: better_robot.viewer.theme.Theme = LIGHT, opacity: float = 1.0)
:canonical: better_robot.viewer.robot.RobotView

```{autodoc2-docstring} better_robot.viewer.robot.RobotView
```

````{py:method} update(q: torch.Tensor) -> None
:canonical: better_robot.viewer.robot.RobotView.update

```{autodoc2-docstring} better_robot.viewer.robot.RobotView.update
```

````

`````

````{py:function} robot_layers(model: better_robot.data_model.model.Model, q: torch.Tensor, **options: typing.Any) -> list[better_robot.viewer.layer.Layer]
:canonical: better_robot.viewer.robot.robot_layers

```{autodoc2-docstring} better_robot.viewer.robot.robot_layers
```
````

````{py:function} robot_skeleton(model: better_robot.data_model.model.Model, joint_poses: numpy.ndarray, *, name: str = 'skeleton', tau: torch.Tensor | None = None, theme: better_robot.viewer.theme.Theme = LIGHT, radius: float = 0.02) -> better_robot.viewer.layers.skeleton.SkeletonLayer
:canonical: better_robot.viewer.robot.robot_skeleton

```{autodoc2-docstring} better_robot.viewer.robot.robot_skeleton
```
````

````{py:function} robot_meshes(model: better_robot.data_model.model.Model, *, collision: bool = False, default_color: typing.Any = (0.8, 0.8, 0.8)) -> list[better_robot.viewer.layers.mesh.MeshSpec]
:canonical: better_robot.viewer.robot.robot_meshes

```{autodoc2-docstring} better_robot.viewer.robot.robot_meshes
```
````
