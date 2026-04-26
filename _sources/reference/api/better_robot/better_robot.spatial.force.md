# {py:mod}`better_robot.spatial.force`

```{py:module} better_robot.spatial.force
```

```{autodoc2-docstring} better_robot.spatial.force
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Force <better_robot.spatial.force.Force>`
  - ```{autodoc2-docstring} better_robot.spatial.force.Force
    :summary:
    ```
````

### API

`````{py:class} Force
:canonical: better_robot.spatial.force.Force

```{autodoc2-docstring} better_robot.spatial.force.Force
```

````{py:attribute} data
:canonical: better_robot.spatial.force.Force.data
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.spatial.force.Force.data
```

````

````{py:method} zero(*, batch_shape: tuple[int, ...] = (), device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> better_robot.spatial.force.Force
:canonical: better_robot.spatial.force.Force.zero
:classmethod:

```{autodoc2-docstring} better_robot.spatial.force.Force.zero
```

````

````{py:property} linear
:canonical: better_robot.spatial.force.Force.linear
:type: torch.Tensor

```{autodoc2-docstring} better_robot.spatial.force.Force.linear
```

````

````{py:property} angular
:canonical: better_robot.spatial.force.Force.angular
:type: torch.Tensor

```{autodoc2-docstring} better_robot.spatial.force.Force.angular
```

````

````{py:method} cross_motion(other) -> Motion
:canonical: better_robot.spatial.force.Force.cross_motion
:abstractmethod:

```{autodoc2-docstring} better_robot.spatial.force.Force.cross_motion
```

````

````{py:method} se3_action(T) -> better_robot.spatial.force.Force
:canonical: better_robot.spatial.force.Force.se3_action

```{autodoc2-docstring} better_robot.spatial.force.Force.se3_action
```

````

`````
