# {py:mod}`better_robot.dynamics.action.action`

```{py:module} better_robot.dynamics.action.action
```

```{autodoc2-docstring} better_robot.dynamics.action.action
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ActionData <better_robot.dynamics.action.action.ActionData>`
  - ```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData
    :summary:
    ```
* - {py:obj}`ActionModel <better_robot.dynamics.action.action.ActionModel>`
  - ```{autodoc2-docstring} better_robot.dynamics.action.action.ActionModel
    :summary:
    ```
````

### API

`````{py:class} ActionData
:canonical: better_robot.dynamics.action.action.ActionData

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData
```

````{py:attribute} xnext
:canonical: better_robot.dynamics.action.action.ActionData.xnext
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData.xnext
```

````

````{py:attribute} cost
:canonical: better_robot.dynamics.action.action.ActionData.cost
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData.cost
```

````

````{py:attribute} fx
:canonical: better_robot.dynamics.action.action.ActionData.fx
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData.fx
```

````

````{py:attribute} fu
:canonical: better_robot.dynamics.action.action.ActionData.fu
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData.fu
```

````

````{py:attribute} lx
:canonical: better_robot.dynamics.action.action.ActionData.lx
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData.lx
```

````

````{py:attribute} lu
:canonical: better_robot.dynamics.action.action.ActionData.lu
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData.lu
```

````

````{py:attribute} lxx
:canonical: better_robot.dynamics.action.action.ActionData.lxx
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData.lxx
```

````

````{py:attribute} lxu
:canonical: better_robot.dynamics.action.action.ActionData.lxu
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData.lxu
```

````

````{py:attribute} luu
:canonical: better_robot.dynamics.action.action.ActionData.luu
:type: typing.Optional[torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData.luu
```

````

````{py:attribute} extras
:canonical: better_robot.dynamics.action.action.ActionData.extras
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionData.extras
```

````

`````

`````{py:class} ActionModel
:canonical: better_robot.dynamics.action.action.ActionModel

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionModel
```

````{py:attribute} state
:canonical: better_robot.dynamics.action.action.ActionModel.state
:type: better_robot.dynamics.state_manifold.StateMultibody
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionModel.state
```

````

````{py:attribute} nu
:canonical: better_robot.dynamics.action.action.ActionModel.nu
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionModel.nu
```

````

````{py:method} calc(data: better_robot.dynamics.action.action.ActionData, x: torch.Tensor, u: torch.Tensor) -> None
:canonical: better_robot.dynamics.action.action.ActionModel.calc

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionModel.calc
```

````

````{py:method} calc_diff(data: better_robot.dynamics.action.action.ActionData, x: torch.Tensor, u: torch.Tensor) -> None
:canonical: better_robot.dynamics.action.action.ActionModel.calc_diff

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionModel.calc_diff
```

````

````{py:method} create_data() -> better_robot.dynamics.action.action.ActionData
:canonical: better_robot.dynamics.action.action.ActionModel.create_data

```{autodoc2-docstring} better_robot.dynamics.action.action.ActionModel.create_data
```

````

`````
