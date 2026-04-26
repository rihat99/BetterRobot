# {py:mod}`better_robot.collision.robot_collision`

```{py:module} better_robot.collision.robot_collision
```

```{autodoc2-docstring} better_robot.collision.robot_collision
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RobotCollision <better_robot.collision.robot_collision.RobotCollision>`
  - ```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision
    :summary:
    ```
````

### API

`````{py:class} RobotCollision
:canonical: better_robot.collision.robot_collision.RobotCollision

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision
```

````{py:attribute} frame_ids
:canonical: better_robot.collision.robot_collision.RobotCollision.frame_ids
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision.frame_ids
```

````

````{py:attribute} local_a
:canonical: better_robot.collision.robot_collision.RobotCollision.local_a
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision.local_a
```

````

````{py:attribute} local_b
:canonical: better_robot.collision.robot_collision.RobotCollision.local_b
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision.local_b
```

````

````{py:attribute} radii
:canonical: better_robot.collision.robot_collision.RobotCollision.radii
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision.radii
```

````

````{py:attribute} self_pairs
:canonical: better_robot.collision.robot_collision.RobotCollision.self_pairs
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision.self_pairs
```

````

````{py:attribute} allowed_pairs_mask
:canonical: better_robot.collision.robot_collision.RobotCollision.allowed_pairs_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision.allowed_pairs_mask
```

````

````{py:method} from_model(model: better_robot.data_model.model.Model, *, mode: typing.Literal[capsule, sphere] = 'capsule', allow_adjacent: bool = False) -> better_robot.collision.robot_collision.RobotCollision
:canonical: better_robot.collision.robot_collision.RobotCollision.from_model
:abstractmethod:
:classmethod:

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision.from_model
```

````

````{py:method} world_capsules(data: better_robot.data_model.data.Data) -> better_robot.collision.geometry.Capsule
:canonical: better_robot.collision.robot_collision.RobotCollision.world_capsules
:abstractmethod:

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision.world_capsules
```

````

````{py:method} self_distances(data: better_robot.data_model.data.Data) -> torch.Tensor
:canonical: better_robot.collision.robot_collision.RobotCollision.self_distances
:abstractmethod:

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision.self_distances
```

````

````{py:method} world_distances(data: better_robot.data_model.data.Data, world: typing.Sequence[better_robot.collision.geometry.Sphere | better_robot.collision.geometry.Capsule | better_robot.collision.geometry.Box]) -> torch.Tensor
:canonical: better_robot.collision.robot_collision.RobotCollision.world_distances
:abstractmethod:

```{autodoc2-docstring} better_robot.collision.robot_collision.RobotCollision.world_distances
```

````

`````
