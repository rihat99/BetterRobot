# {py:mod}`better_robot.spatial.motion`

```{py:module} better_robot.spatial.motion
```

```{autodoc2-docstring} better_robot.spatial.motion
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Motion <better_robot.spatial.motion.Motion>`
  - ```{autodoc2-docstring} better_robot.spatial.motion.Motion
    :summary:
    ```
````

### API

`````{py:class} Motion
:canonical: better_robot.spatial.motion.Motion

```{autodoc2-docstring} better_robot.spatial.motion.Motion
```

````{py:attribute} data
:canonical: better_robot.spatial.motion.Motion.data
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.spatial.motion.Motion.data
```

````

````{py:method} zero(*, batch_shape: tuple[int, ...] = (), device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> better_robot.spatial.motion.Motion
:canonical: better_robot.spatial.motion.Motion.zero
:classmethod:

```{autodoc2-docstring} better_robot.spatial.motion.Motion.zero
```

````

````{py:property} linear
:canonical: better_robot.spatial.motion.Motion.linear
:type: torch.Tensor

```{autodoc2-docstring} better_robot.spatial.motion.Motion.linear
```

````

````{py:property} angular
:canonical: better_robot.spatial.motion.Motion.angular
:type: torch.Tensor

```{autodoc2-docstring} better_robot.spatial.motion.Motion.angular
```

````

````{py:method} cross_motion(other: better_robot.spatial.motion.Motion) -> better_robot.spatial.motion.Motion
:canonical: better_robot.spatial.motion.Motion.cross_motion

```{autodoc2-docstring} better_robot.spatial.motion.Motion.cross_motion
```

````

````{py:method} cross_force(other) -> better_robot.spatial.force.Force
:canonical: better_robot.spatial.motion.Motion.cross_force

```{autodoc2-docstring} better_robot.spatial.motion.Motion.cross_force
```

````

````{py:method} se3_action(T) -> better_robot.spatial.motion.Motion
:canonical: better_robot.spatial.motion.Motion.se3_action

```{autodoc2-docstring} better_robot.spatial.motion.Motion.se3_action
```

````

````{py:method} compose(other: better_robot.spatial.motion.Motion) -> better_robot.spatial.motion.Motion
:canonical: better_robot.spatial.motion.Motion.compose

```{autodoc2-docstring} better_robot.spatial.motion.Motion.compose
```

````

`````
