# {py:mod}`better_robot.data_model.frame`

```{py:module} better_robot.data_model.frame
```

```{autodoc2-docstring} better_robot.data_model.frame
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Frame <better_robot.data_model.frame.Frame>`
  - ```{autodoc2-docstring} better_robot.data_model.frame.Frame
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FrameType <better_robot.data_model.frame.FrameType>`
  - ```{autodoc2-docstring} better_robot.data_model.frame.FrameType
    :summary:
    ```
````

### API

````{py:data} FrameType
:canonical: better_robot.data_model.frame.FrameType
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.frame.FrameType
```

````

`````{py:class} Frame
:canonical: better_robot.data_model.frame.Frame

```{autodoc2-docstring} better_robot.data_model.frame.Frame
```

````{py:attribute} name
:canonical: better_robot.data_model.frame.Frame.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.frame.Frame.name
```

````

````{py:attribute} parent_joint
:canonical: better_robot.data_model.frame.Frame.parent_joint
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.frame.Frame.parent_joint
```

````

````{py:attribute} joint_placement
:canonical: better_robot.data_model.frame.Frame.joint_placement
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.frame.Frame.joint_placement
```

````

````{py:attribute} frame_type
:canonical: better_robot.data_model.frame.Frame.frame_type
:type: better_robot.data_model.frame.FrameType
:value: >
   'op'

```{autodoc2-docstring} better_robot.data_model.frame.Frame.frame_type
```

````

`````
