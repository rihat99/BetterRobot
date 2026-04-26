# {py:mod}`better_robot.kinematics`

```{py:module} better_robot.kinematics
```

```{autodoc2-docstring} better_robot.kinematics
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

better_robot.kinematics.jacobian
better_robot.kinematics.jacobian_strategy
better_robot.kinematics.chain
better_robot.kinematics.forward
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ReferenceFrame <better_robot.kinematics.ReferenceFrame>`
  - ```{autodoc2-docstring} better_robot.kinematics.ReferenceFrame
    :summary:
    ```
````

### API

`````{py:class} ReferenceFrame()
:canonical: better_robot.kinematics.ReferenceFrame

Bases: {py:obj}`str`, {py:obj}`enum.Enum`

```{autodoc2-docstring} better_robot.kinematics.ReferenceFrame
```

````{py:attribute} WORLD
:canonical: better_robot.kinematics.ReferenceFrame.WORLD
:value: >
   'world'

```{autodoc2-docstring} better_robot.kinematics.ReferenceFrame.WORLD
```

````

````{py:attribute} LOCAL
:canonical: better_robot.kinematics.ReferenceFrame.LOCAL
:value: >
   'local'

```{autodoc2-docstring} better_robot.kinematics.ReferenceFrame.LOCAL
```

````

````{py:attribute} LOCAL_WORLD_ALIGNED
:canonical: better_robot.kinematics.ReferenceFrame.LOCAL_WORLD_ALIGNED
:value: >
   'local_world_aligned'

```{autodoc2-docstring} better_robot.kinematics.ReferenceFrame.LOCAL_WORLD_ALIGNED
```

````

`````
