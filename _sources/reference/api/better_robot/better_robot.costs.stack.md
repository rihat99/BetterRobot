# {py:mod}`better_robot.costs.stack`

```{py:module} better_robot.costs.stack
```

```{autodoc2-docstring} better_robot.costs.stack
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CostItem <better_robot.costs.stack.CostItem>`
  - ```{autodoc2-docstring} better_robot.costs.stack.CostItem
    :summary:
    ```
* - {py:obj}`CostStack <better_robot.costs.stack.CostStack>`
  - ```{autodoc2-docstring} better_robot.costs.stack.CostStack
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CostKind <better_robot.costs.stack.CostKind>`
  - ```{autodoc2-docstring} better_robot.costs.stack.CostKind
    :summary:
    ```
````

### API

````{py:data} CostKind
:canonical: better_robot.costs.stack.CostKind
:value: >
   None

```{autodoc2-docstring} better_robot.costs.stack.CostKind
```

````

`````{py:class} CostItem
:canonical: better_robot.costs.stack.CostItem

```{autodoc2-docstring} better_robot.costs.stack.CostItem
```

````{py:attribute} name
:canonical: better_robot.costs.stack.CostItem.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.costs.stack.CostItem.name
```

````

````{py:attribute} residual
:canonical: better_robot.costs.stack.CostItem.residual
:type: better_robot.residuals.base.Residual
:value: >
   None

```{autodoc2-docstring} better_robot.costs.stack.CostItem.residual
```

````

````{py:attribute} weight
:canonical: better_robot.costs.stack.CostItem.weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} better_robot.costs.stack.CostItem.weight
```

````

````{py:attribute} active
:canonical: better_robot.costs.stack.CostItem.active
:type: bool
:value: >
   True

```{autodoc2-docstring} better_robot.costs.stack.CostItem.active
```

````

````{py:attribute} kind
:canonical: better_robot.costs.stack.CostItem.kind
:type: better_robot.costs.stack.CostKind
:value: >
   'soft'

```{autodoc2-docstring} better_robot.costs.stack.CostItem.kind
```

````

````{py:attribute} slice
:canonical: better_robot.costs.stack.CostItem.slice
:type: better_robot.costs.stack.CostItem.slice | None
:value: >
   None

```{autodoc2-docstring} better_robot.costs.stack.CostItem.slice
```

````

`````

`````{py:class} CostStack()
:canonical: better_robot.costs.stack.CostStack

```{autodoc2-docstring} better_robot.costs.stack.CostStack
```

````{py:attribute} items
:canonical: better_robot.costs.stack.CostStack.items
:type: dict[str, better_robot.costs.stack.CostItem]
:value: >
   None

```{autodoc2-docstring} better_robot.costs.stack.CostStack.items
```

````

````{py:method} add(name: str, residual: better_robot.residuals.base.Residual, *, weight: float = 1.0, kind: better_robot.costs.stack.CostKind = 'soft') -> None
:canonical: better_robot.costs.stack.CostStack.add

```{autodoc2-docstring} better_robot.costs.stack.CostStack.add
```

````

````{py:method} remove(name: str) -> None
:canonical: better_robot.costs.stack.CostStack.remove

```{autodoc2-docstring} better_robot.costs.stack.CostStack.remove
```

````

````{py:method} set_active(name: str, active: bool) -> None
:canonical: better_robot.costs.stack.CostStack.set_active

```{autodoc2-docstring} better_robot.costs.stack.CostStack.set_active
```

````

````{py:method} set_weight(name: str, weight: float) -> None
:canonical: better_robot.costs.stack.CostStack.set_weight

```{autodoc2-docstring} better_robot.costs.stack.CostStack.set_weight
```

````

````{py:method} total_dim() -> int
:canonical: better_robot.costs.stack.CostStack.total_dim

```{autodoc2-docstring} better_robot.costs.stack.CostStack.total_dim
```

````

````{py:method} slice_map() -> dict[str, slice]
:canonical: better_robot.costs.stack.CostStack.slice_map

```{autodoc2-docstring} better_robot.costs.stack.CostStack.slice_map
```

````

````{py:method} residual(state: better_robot.residuals.base.ResidualState) -> torch.Tensor
:canonical: better_robot.costs.stack.CostStack.residual

```{autodoc2-docstring} better_robot.costs.stack.CostStack.residual
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState, *, strategy: better_robot.kinematics.jacobian_strategy.JacobianStrategy = JacobianStrategy.AUTO) -> torch.Tensor
:canonical: better_robot.costs.stack.CostStack.jacobian

```{autodoc2-docstring} better_robot.costs.stack.CostStack.jacobian
```

````

````{py:method} gradient(state: better_robot.residuals.base.ResidualState) -> torch.Tensor
:canonical: better_robot.costs.stack.CostStack.gradient

```{autodoc2-docstring} better_robot.costs.stack.CostStack.gradient
```

````

`````
