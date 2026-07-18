# {py:mod}`better_robot.tasks.smoothing`

```{py:module} better_robot.tasks.smoothing
```

```{autodoc2-docstring} better_robot.tasks.smoothing
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`smooth_trajectory <better_robot.tasks.smoothing.smooth_trajectory>`
  - ```{autodoc2-docstring} better_robot.tasks.smoothing.smooth_trajectory
    :summary:
    ```
````

### API

````{py:function} smooth_trajectory(trajectory: better_robot.tasks.trajectory.Trajectory, kernel: torch.Tensor, *, kind: typing.Literal[auto, better_robot.lie.so3, better_robot.lie.se3] = 'auto') -> better_robot.tasks.trajectory.Trajectory
:canonical: better_robot.tasks.smoothing.smooth_trajectory

```{autodoc2-docstring} better_robot.tasks.smoothing.smooth_trajectory
```
````
