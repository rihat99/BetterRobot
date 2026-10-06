# {py:mod}`better_robot.optim2.linear`

```{py:module} better_robot.optim2.linear
```

```{autodoc2-docstring} better_robot.optim2.linear
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Normal <better_robot.optim2.linear.Normal>`
  - ```{autodoc2-docstring} better_robot.optim2.linear.Normal
    :summary:
    ```
* - {py:obj}`BandLayout <better_robot.optim2.linear.BandLayout>`
  - ```{autodoc2-docstring} better_robot.optim2.linear.BandLayout
    :summary:
    ```
* - {py:obj}`BandedNormal <better_robot.optim2.linear.BandedNormal>`
  - ```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal
    :summary:
    ```
* - {py:obj}`ArrowNormal <better_robot.optim2.linear.ArrowNormal>`
  - ```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal
    :summary:
    ```
* - {py:obj}`DenseSink <better_robot.optim2.linear.DenseSink>`
  - ```{autodoc2-docstring} better_robot.optim2.linear.DenseSink
    :summary:
    ```
* - {py:obj}`BandSink <better_robot.optim2.linear.BandSink>`
  - ```{autodoc2-docstring} better_robot.optim2.linear.BandSink
    :summary:
    ```
* - {py:obj}`ArrowSink <better_robot.optim2.linear.ArrowSink>`
  - ```{autodoc2-docstring} better_robot.optim2.linear.ArrowSink
    :summary:
    ```
````

### API

`````{py:class} Normal
:canonical: better_robot.optim2.linear.Normal

```{autodoc2-docstring} better_robot.optim2.linear.Normal
```

````{py:attribute} H
:canonical: better_robot.optim2.linear.Normal.H
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.Normal.H
```

````

````{py:attribute} g
:canonical: better_robot.optim2.linear.Normal.g
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.Normal.g
```

````

````{py:attribute} lower
:canonical: better_robot.optim2.linear.Normal.lower
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.Normal.lower
```

````

````{py:attribute} upper
:canonical: better_robot.optim2.linear.Normal.upper
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.Normal.upper
```

````

````{py:method} restrict(lower: torch.Tensor, upper: torch.Tensor, *, tolerance: float = 1e-09) -> better_robot.optim2.linear.Normal
:canonical: better_robot.optim2.linear.Normal.restrict

```{autodoc2-docstring} better_robot.optim2.linear.Normal.restrict
```

````

````{py:method} masked(keep: torch.Tensor) -> better_robot.optim2.linear.Normal
:canonical: better_robot.optim2.linear.Normal.masked

```{autodoc2-docstring} better_robot.optim2.linear.Normal.masked
```

````

````{py:method} solve(damping: torch.Tensor | float = 0.0, *, floor: float = 1e-09, g: torch.Tensor | None = None) -> torch.Tensor
:canonical: better_robot.optim2.linear.Normal.solve

```{autodoc2-docstring} better_robot.optim2.linear.Normal.solve
```

````

````{py:method} predicted_decrease(delta: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim2.linear.Normal.predicted_decrease

```{autodoc2-docstring} better_robot.optim2.linear.Normal.predicted_decrease
```

````

`````

`````{py:class} BandLayout
:canonical: better_robot.optim2.linear.BandLayout

```{autodoc2-docstring} better_robot.optim2.linear.BandLayout
```

````{py:attribute} block_of_column
:canonical: better_robot.optim2.linear.BandLayout.block_of_column
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandLayout.block_of_column
```

````

````{py:attribute} row_of_column
:canonical: better_robot.optim2.linear.BandLayout.row_of_column
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandLayout.row_of_column
```

````

````{py:attribute} blocks
:canonical: better_robot.optim2.linear.BandLayout.blocks
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandLayout.blocks
```

````

````{py:attribute} block_size
:canonical: better_robot.optim2.linear.BandLayout.block_size
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandLayout.block_size
```

````

````{py:attribute} time_columns
:canonical: better_robot.optim2.linear.BandLayout.time_columns
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandLayout.time_columns
```

````

````{py:attribute} shared_columns
:canonical: better_robot.optim2.linear.BandLayout.shared_columns
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandLayout.shared_columns
```

````

````{py:property} position
:canonical: better_robot.optim2.linear.BandLayout.position
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim2.linear.BandLayout.position
```

````

`````

`````{py:class} BandedNormal
:canonical: better_robot.optim2.linear.BandedNormal

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal
```

````{py:attribute} diagonal
:canonical: better_robot.optim2.linear.BandedNormal.diagonal
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.diagonal
```

````

````{py:attribute} upper
:canonical: better_robot.optim2.linear.BandedNormal.upper
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.upper
```

````

````{py:attribute} g
:canonical: better_robot.optim2.linear.BandedNormal.g
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.g
```

````

````{py:attribute} layout
:canonical: better_robot.optim2.linear.BandedNormal.layout
:type: better_robot.optim2.linear.BandLayout
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.layout
```

````

````{py:attribute} lower
:canonical: better_robot.optim2.linear.BandedNormal.lower
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.lower
```

````

````{py:attribute} upper_bound
:canonical: better_robot.optim2.linear.BandedNormal.upper_bound
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.upper_bound
```

````

````{py:method} masked(keep: torch.Tensor) -> better_robot.optim2.linear.BandedNormal
:canonical: better_robot.optim2.linear.BandedNormal.masked

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.masked
```

````

````{py:method} restrict(lower: torch.Tensor, upper: torch.Tensor, *, tolerance: float = 1e-09) -> better_robot.optim2.linear.BandedNormal
:canonical: better_robot.optim2.linear.BandedNormal.restrict

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.restrict
```

````

````{py:method} solve(damping: torch.Tensor | float = 0.0, *, floor: float = 1e-09, g: torch.Tensor | None = None) -> torch.Tensor
:canonical: better_robot.optim2.linear.BandedNormal.solve

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.solve
```

````

````{py:method} predicted_decrease(delta: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim2.linear.BandedNormal.predicted_decrease

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.predicted_decrease
```

````

````{py:method} dense() -> torch.Tensor
:canonical: better_robot.optim2.linear.BandedNormal.dense

```{autodoc2-docstring} better_robot.optim2.linear.BandedNormal.dense
```

````

`````

`````{py:class} ArrowNormal
:canonical: better_robot.optim2.linear.ArrowNormal

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal
```

````{py:attribute} band
:canonical: better_robot.optim2.linear.ArrowNormal.band
:type: better_robot.optim2.linear.BandedNormal
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.band
```

````

````{py:attribute} border
:canonical: better_robot.optim2.linear.ArrowNormal.border
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.border
```

````

````{py:attribute} corner
:canonical: better_robot.optim2.linear.ArrowNormal.corner
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.corner
```

````

````{py:attribute} g
:canonical: better_robot.optim2.linear.ArrowNormal.g
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.g
```

````

````{py:attribute} lower
:canonical: better_robot.optim2.linear.ArrowNormal.lower
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.lower
```

````

````{py:attribute} upper
:canonical: better_robot.optim2.linear.ArrowNormal.upper
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.upper
```

````

````{py:method} masked(keep: torch.Tensor) -> better_robot.optim2.linear.ArrowNormal
:canonical: better_robot.optim2.linear.ArrowNormal.masked

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.masked
```

````

````{py:method} restrict(lower: torch.Tensor, upper: torch.Tensor, *, tolerance: float = 1e-09) -> better_robot.optim2.linear.ArrowNormal
:canonical: better_robot.optim2.linear.ArrowNormal.restrict

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.restrict
```

````

````{py:method} solve(damping: torch.Tensor | float = 0.0, *, floor: float = 1e-09, g: torch.Tensor | None = None) -> torch.Tensor
:canonical: better_robot.optim2.linear.ArrowNormal.solve

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.solve
```

````

````{py:method} predicted_decrease(delta: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim2.linear.ArrowNormal.predicted_decrease

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.predicted_decrease
```

````

````{py:method} dense() -> torch.Tensor
:canonical: better_robot.optim2.linear.ArrowNormal.dense

```{autodoc2-docstring} better_robot.optim2.linear.ArrowNormal.dense
```

````

`````

`````{py:class} DenseSink(reference: torch.Tensor, batch: int, size: int)
:canonical: better_robot.optim2.linear.DenseSink

```{autodoc2-docstring} better_robot.optim2.linear.DenseSink
```

````{py:method} add(columns_a: torch.Tensor, columns_b: torch.Tensor, values: torch.Tensor) -> None
:canonical: better_robot.optim2.linear.DenseSink.add

```{autodoc2-docstring} better_robot.optim2.linear.DenseSink.add
```

````

````{py:method} finish(g: torch.Tensor) -> better_robot.optim2.linear.Normal
:canonical: better_robot.optim2.linear.DenseSink.finish

```{autodoc2-docstring} better_robot.optim2.linear.DenseSink.finish
```

````

`````

`````{py:class} BandSink(reference: torch.Tensor, batch: int, layout: better_robot.optim2.linear.BandLayout)
:canonical: better_robot.optim2.linear.BandSink

```{autodoc2-docstring} better_robot.optim2.linear.BandSink
```

````{py:method} add(columns_a: torch.Tensor, columns_b: torch.Tensor, values: torch.Tensor) -> None
:canonical: better_robot.optim2.linear.BandSink.add

```{autodoc2-docstring} better_robot.optim2.linear.BandSink.add
```

````

````{py:method} finish(g: torch.Tensor) -> better_robot.optim2.linear.BandedNormal
:canonical: better_robot.optim2.linear.BandSink.finish

```{autodoc2-docstring} better_robot.optim2.linear.BandSink.finish
```

````

`````

`````{py:class} ArrowSink(reference: torch.Tensor, batch: int, layout: better_robot.optim2.linear.BandLayout)
:canonical: better_robot.optim2.linear.ArrowSink

```{autodoc2-docstring} better_robot.optim2.linear.ArrowSink
```

````{py:method} add(columns_a: torch.Tensor, columns_b: torch.Tensor, values: torch.Tensor) -> None
:canonical: better_robot.optim2.linear.ArrowSink.add

```{autodoc2-docstring} better_robot.optim2.linear.ArrowSink.add
```

````

````{py:method} finish(g: torch.Tensor) -> better_robot.optim2.linear.ArrowNormal
:canonical: better_robot.optim2.linear.ArrowSink.finish

```{autodoc2-docstring} better_robot.optim2.linear.ArrowSink.finish
```

````

`````
