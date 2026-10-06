# {py:mod}`better_robot.collision.robot_spheres`

```{py:module} better_robot.collision.robot_spheres
```

```{autodoc2-docstring} better_robot.collision.robot_spheres
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RobotSpheres <better_robot.collision.robot_spheres.RobotSpheres>`
  - ```{autodoc2-docstring} better_robot.collision.robot_spheres.RobotSpheres
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`self_collision_pairs <better_robot.collision.robot_spheres.self_collision_pairs>`
  - ```{autodoc2-docstring} better_robot.collision.robot_spheres.self_collision_pairs
    :summary:
    ```
* - {py:obj}`self_distances <better_robot.collision.robot_spheres.self_distances>`
  - ```{autodoc2-docstring} better_robot.collision.robot_spheres.self_distances
    :summary:
    ```
````

### API

`````{py:class} RobotSpheres
:canonical: better_robot.collision.robot_spheres.RobotSpheres

```{autodoc2-docstring} better_robot.collision.robot_spheres.RobotSpheres
```

````{py:attribute} frame_ids
:canonical: better_robot.collision.robot_spheres.RobotSpheres.frame_ids
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.robot_spheres.RobotSpheres.frame_ids
```

````

````{py:attribute} offsets
:canonical: better_robot.collision.robot_spheres.RobotSpheres.offsets
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.robot_spheres.RobotSpheres.offsets
```

````

````{py:attribute} radii
:canonical: better_robot.collision.robot_spheres.RobotSpheres.radii
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.robot_spheres.RobotSpheres.radii
```

````

````{py:method} from_frames(model: better_robot.data_model.model.Model, spheres: collections.abc.Mapping[str | int, collections.abc.Sequence[tuple[collections.abc.Sequence[float], float]]]) -> better_robot.collision.robot_spheres.RobotSpheres
:canonical: better_robot.collision.robot_spheres.RobotSpheres.from_frames
:classmethod:

```{autodoc2-docstring} better_robot.collision.robot_spheres.RobotSpheres.from_frames
```

````

````{py:method} centers(state: better_robot.kinematics.state.KinematicState) -> torch.Tensor
:canonical: better_robot.collision.robot_spheres.RobotSpheres.centers

```{autodoc2-docstring} better_robot.collision.robot_spheres.RobotSpheres.centers
```

````

`````

````{py:function} self_collision_pairs(model: better_robot.data_model.model.Model, spheres: better_robot.collision.robot_spheres.RobotSpheres) -> torch.Tensor
:canonical: better_robot.collision.robot_spheres.self_collision_pairs

```{autodoc2-docstring} better_robot.collision.robot_spheres.self_collision_pairs
```
````

````{py:function} self_distances(spheres: better_robot.collision.robot_spheres.RobotSpheres, state: better_robot.kinematics.state.KinematicState, pairs: torch.Tensor) -> torch.Tensor
:canonical: better_robot.collision.robot_spheres.self_distances

```{autodoc2-docstring} better_robot.collision.robot_spheres.self_distances
```
````
