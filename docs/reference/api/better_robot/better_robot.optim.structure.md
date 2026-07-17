# {py:mod}`better_robot.optim.structure`

```{py:module} better_robot.optim.structure
```

```{autodoc2-docstring} better_robot.optim.structure
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearizationReason <better_robot.optim.structure.LinearizationReason>`
  - ```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason
    :summary:
    ```
* - {py:obj}`LinearizationDecision <better_robot.optim.structure.LinearizationDecision>`
  - ```{autodoc2-docstring} better_robot.optim.structure.LinearizationDecision
    :summary:
    ```
* - {py:obj}`TemporalAnalysis <better_robot.optim.structure.TemporalAnalysis>`
  - ```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis
    :summary:
    ```
* - {py:obj}`BlockBandedMatrix <better_robot.optim.structure.BlockBandedMatrix>`
  - ```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix
    :summary:
    ```
* - {py:obj}`NormalOperator <better_robot.optim.structure.NormalOperator>`
  - ```{autodoc2-docstring} better_robot.optim.structure.NormalOperator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearizationMode <better_robot.optim.structure.LinearizationMode>`
  - ```{autodoc2-docstring} better_robot.optim.structure.LinearizationMode
    :summary:
    ```
* - {py:obj}`LinearSystemKind <better_robot.optim.structure.LinearSystemKind>`
  - ```{autodoc2-docstring} better_robot.optim.structure.LinearSystemKind
    :summary:
    ```
* - {py:obj}`MatVec <better_robot.optim.structure.MatVec>`
  - ```{autodoc2-docstring} better_robot.optim.structure.MatVec
    :summary:
    ```
````

### API

````{py:data} LinearizationMode
:canonical: better_robot.optim.structure.LinearizationMode
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.LinearizationMode
```

````

````{py:data} LinearSystemKind
:canonical: better_robot.optim.structure.LinearSystemKind
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.LinearSystemKind
```

````

````{py:data} MatVec
:canonical: better_robot.optim.structure.MatVec
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.MatVec
```

````

`````{py:class} LinearizationReason()
:canonical: better_robot.optim.structure.LinearizationReason

Bases: {py:obj}`str`, {py:obj}`enum.Enum`

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason
```

````{py:attribute} FORCED_DENSE
:canonical: better_robot.optim.structure.LinearizationReason.FORCED_DENSE
:value: >
   'forced_dense'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.FORCED_DENSE
```

````

````{py:attribute} ELIGIBLE_BANDED
:canonical: better_robot.optim.structure.LinearizationReason.ELIGIBLE_BANDED
:value: >
   'eligible_banded'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.ELIGIBLE_BANDED
```

````

````{py:attribute} EXPLICIT_MATRIX_FREE
:canonical: better_robot.optim.structure.LinearizationReason.EXPLICIT_MATRIX_FREE
:value: >
   'explicit_matrix_free'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.EXPLICIT_MATRIX_FREE
```

````

````{py:attribute} NO_TIME_VARIABLE
:canonical: better_robot.optim.structure.LinearizationReason.NO_TIME_VARIABLE
:value: >
   'no_time_variable'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.NO_TIME_VARIABLE
```

````

````{py:attribute} MULTIPLE_OPTIMIZED_VARIABLES
:canonical: better_robot.optim.structure.LinearizationReason.MULTIPLE_OPTIMIZED_VARIABLES
:value: >
   'multiple_optimized_variables'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.MULTIPLE_OPTIMIZED_VARIABLES
```

````

````{py:attribute} NONSEPARABLE_MASK
:canonical: better_robot.optim.structure.LinearizationReason.NONSEPARABLE_MASK
:value: >
   'nonseparable_mask'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.NONSEPARABLE_MASK
```

````

````{py:attribute} UNDECLARED_TEMPORAL_RESIDUAL
:canonical: better_robot.optim.structure.LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL
:value: >
   'undeclared_temporal_residual'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL
```

````

````{py:attribute} MISSING_TEMPORAL_BLOCKS
:canonical: better_robot.optim.structure.LinearizationReason.MISSING_TEMPORAL_BLOCKS
:value: >
   'missing_temporal_blocks'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.MISSING_TEMPORAL_BLOCKS
```

````

````{py:attribute} MIXED_OPTIMIZED_DEPENDENCY
:canonical: better_robot.optim.structure.LinearizationReason.MIXED_OPTIMIZED_DEPENDENCY
:value: >
   'mixed_optimized_dependency'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.MIXED_OPTIMIZED_DEPENDENCY
```

````

````{py:attribute} EXPLICIT_DENSE_SOLVER
:canonical: better_robot.optim.structure.LinearizationReason.EXPLICIT_DENSE_SOLVER
:value: >
   'explicit_dense_solver'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.EXPLICIT_DENSE_SOLVER
```

````

````{py:attribute} INCOMPATIBLE_SOLVER
:canonical: better_robot.optim.structure.LinearizationReason.INCOMPATIBLE_SOLVER
:value: >
   'incompatible_solver'

```{autodoc2-docstring} better_robot.optim.structure.LinearizationReason.INCOMPATIBLE_SOLVER
```

````

`````

`````{py:class} LinearizationDecision
:canonical: better_robot.optim.structure.LinearizationDecision

```{autodoc2-docstring} better_robot.optim.structure.LinearizationDecision
```

````{py:attribute} requested
:canonical: better_robot.optim.structure.LinearizationDecision.requested
:type: better_robot.optim.structure.LinearizationMode
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.LinearizationDecision.requested
```

````

````{py:attribute} used
:canonical: better_robot.optim.structure.LinearizationDecision.used
:type: typing.Literal[dense, banded, matrix_free]
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.LinearizationDecision.used
```

````

````{py:attribute} reason
:canonical: better_robot.optim.structure.LinearizationDecision.reason
:type: better_robot.optim.structure.LinearizationReason
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.LinearizationDecision.reason
```

````

````{py:attribute} detail
:canonical: better_robot.optim.structure.LinearizationDecision.detail
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.LinearizationDecision.detail
```

````

`````

`````{py:class} TemporalAnalysis
:canonical: better_robot.optim.structure.TemporalAnalysis

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis
```

````{py:attribute} operator_eligible
:canonical: better_robot.optim.structure.TemporalAnalysis.operator_eligible
:type: bool
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis.operator_eligible
```

````

````{py:attribute} direct_eligible
:canonical: better_robot.optim.structure.TemporalAnalysis.direct_eligible
:type: bool
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis.direct_eligible
```

````

````{py:attribute} reason
:canonical: better_robot.optim.structure.TemporalAnalysis.reason
:type: better_robot.optim.structure.LinearizationReason
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis.reason
```

````

````{py:attribute} detail
:canonical: better_robot.optim.structure.TemporalAnalysis.detail
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis.detail
```

````

````{py:attribute} variable_name
:canonical: better_robot.optim.structure.TemporalAnalysis.variable_name
:type: str | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis.variable_name
```

````

````{py:attribute} time_length
:canonical: better_robot.optim.structure.TemporalAnalysis.time_length
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis.time_length
```

````

````{py:attribute} tangent_width
:canonical: better_robot.optim.structure.TemporalAnalysis.tangent_width
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis.tangent_width
```

````

````{py:attribute} reduced_width
:canonical: better_robot.optim.structure.TemporalAnalysis.reduced_width
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis.reduced_width
```

````

````{py:attribute} bandwidth
:canonical: better_robot.optim.structure.TemporalAnalysis.bandwidth
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis.bandwidth
```

````

````{py:attribute} patterns
:canonical: better_robot.optim.structure.TemporalAnalysis.patterns
:type: tuple[tuple[str, better_robot.residuals.structure.TemporalPattern], ...]
:value: >
   ()

```{autodoc2-docstring} better_robot.optim.structure.TemporalAnalysis.patterns
```

````

`````

`````{py:class} BlockBandedMatrix
:canonical: better_robot.optim.structure.BlockBandedMatrix

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix
```

````{py:attribute} bands
:canonical: better_robot.optim.structure.BlockBandedMatrix.bands
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.bands
```

````

````{py:attribute} bandwidth
:canonical: better_robot.optim.structure.BlockBandedMatrix.bandwidth
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.bandwidth
```

````

````{py:property} batch_shape
:canonical: better_robot.optim.structure.BlockBandedMatrix.batch_shape
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.batch_shape
```

````

````{py:property} time_length
:canonical: better_robot.optim.structure.BlockBandedMatrix.time_length
:type: int

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.time_length
```

````

````{py:property} block_size
:canonical: better_robot.optim.structure.BlockBandedMatrix.block_size
:type: int

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.block_size
```

````

````{py:property} size
:canonical: better_robot.optim.structure.BlockBandedMatrix.size
:type: int

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.size
```

````

````{py:property} finite
:canonical: better_robot.optim.structure.BlockBandedMatrix.finite
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.finite
```

````

````{py:method} diagonal() -> torch.Tensor
:canonical: better_robot.optim.structure.BlockBandedMatrix.diagonal

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.diagonal
```

````

````{py:method} densify() -> torch.Tensor
:canonical: better_robot.optim.structure.BlockBandedMatrix.densify

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.densify
```

````

````{py:method} matvec(vector: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.structure.BlockBandedMatrix.matvec

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.matvec
```

````

````{py:method} add_diagonal(diagonal: torch.Tensor) -> better_robot.optim.structure.BlockBandedMatrix
:canonical: better_robot.optim.structure.BlockBandedMatrix.add_diagonal

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.add_diagonal
```

````

````{py:method} scaled_restricted(scale: torch.Tensor, movable: torch.Tensor, diagonal: torch.Tensor) -> better_robot.optim.structure.BlockBandedMatrix
:canonical: better_robot.optim.structure.BlockBandedMatrix.scaled_restricted

```{autodoc2-docstring} better_robot.optim.structure.BlockBandedMatrix.scaled_restricted
```

````

`````

`````{py:class} NormalOperator
:canonical: better_robot.optim.structure.NormalOperator

```{autodoc2-docstring} better_robot.optim.structure.NormalOperator
```

````{py:attribute} size
:canonical: better_robot.optim.structure.NormalOperator.size
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.NormalOperator.size
```

````

````{py:attribute} matvec
:canonical: better_robot.optim.structure.NormalOperator.matvec
:type: better_robot.optim.structure.MatVec
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.NormalOperator.matvec
```

````

````{py:attribute} preconditioner
:canonical: better_robot.optim.structure.NormalOperator.preconditioner
:type: better_robot.optim.structure.MatVec | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.NormalOperator.preconditioner
```

````

````{py:attribute} block_shape
:canonical: better_robot.optim.structure.NormalOperator.block_shape
:type: tuple[int, int] | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.structure.NormalOperator.block_shape
```

````

`````
