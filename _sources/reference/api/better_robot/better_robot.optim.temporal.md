# {py:mod}`better_robot.optim.temporal`

```{py:module} better_robot.optim.temporal
```

```{autodoc2-docstring} better_robot.optim.temporal
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearizationReason <better_robot.optim.temporal.LinearizationReason>`
  -
* - {py:obj}`TemporalAnalysis <better_robot.optim.temporal.TemporalAnalysis>`
  - ```{autodoc2-docstring} better_robot.optim.temporal.TemporalAnalysis
    :summary:
    ```
* - {py:obj}`BlockBandedMatrix <better_robot.optim.temporal.BlockBandedMatrix>`
  - ```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix
    :summary:
    ```
* - {py:obj}`StructuredNormal <better_robot.optim.temporal.StructuredNormal>`
  - ```{autodoc2-docstring} better_robot.optim.temporal.StructuredNormal
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`analyze_temporal_problem <better_robot.optim.temporal.analyze_temporal_problem>`
  - ```{autodoc2-docstring} better_robot.optim.temporal.analyze_temporal_problem
    :summary:
    ```
* - {py:obj}`assemble_structured_normal <better_robot.optim.temporal.assemble_structured_normal>`
  - ```{autodoc2-docstring} better_robot.optim.temporal.assemble_structured_normal
    :summary:
    ```
````

### API

`````{py:class} LinearizationReason()
:canonical: better_robot.optim.temporal.LinearizationReason

Bases: {py:obj}`str`, {py:obj}`enum.Enum`

````{py:attribute} FORCED_DENSE
:canonical: better_robot.optim.temporal.LinearizationReason.FORCED_DENSE
:value: >
   'forced_dense'

```{autodoc2-docstring} better_robot.optim.temporal.LinearizationReason.FORCED_DENSE
```

````

````{py:attribute} ELIGIBLE_BANDED
:canonical: better_robot.optim.temporal.LinearizationReason.ELIGIBLE_BANDED
:value: >
   'eligible_banded'

```{autodoc2-docstring} better_robot.optim.temporal.LinearizationReason.ELIGIBLE_BANDED
```

````

````{py:attribute} NO_TIME_VARIABLE
:canonical: better_robot.optim.temporal.LinearizationReason.NO_TIME_VARIABLE
:value: >
   'no_time_variable'

```{autodoc2-docstring} better_robot.optim.temporal.LinearizationReason.NO_TIME_VARIABLE
```

````

````{py:attribute} MULTIPLE_OPTIMIZED_VARIABLES
:canonical: better_robot.optim.temporal.LinearizationReason.MULTIPLE_OPTIMIZED_VARIABLES
:value: >
   'multiple_optimized_variables'

```{autodoc2-docstring} better_robot.optim.temporal.LinearizationReason.MULTIPLE_OPTIMIZED_VARIABLES
```

````

````{py:attribute} UNDECLARED_TEMPORAL_RESIDUAL
:canonical: better_robot.optim.temporal.LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL
:value: >
   'undeclared_temporal_residual'

```{autodoc2-docstring} better_robot.optim.temporal.LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL
```

````

````{py:attribute} MISSING_TEMPORAL_BLOCKS
:canonical: better_robot.optim.temporal.LinearizationReason.MISSING_TEMPORAL_BLOCKS
:value: >
   'missing_temporal_blocks'

```{autodoc2-docstring} better_robot.optim.temporal.LinearizationReason.MISSING_TEMPORAL_BLOCKS
```

````

````{py:attribute} MIXED_OPTIMIZED_DEPENDENCY
:canonical: better_robot.optim.temporal.LinearizationReason.MIXED_OPTIMIZED_DEPENDENCY
:value: >
   'mixed_optimized_dependency'

```{autodoc2-docstring} better_robot.optim.temporal.LinearizationReason.MIXED_OPTIMIZED_DEPENDENCY
```

````

````{py:attribute} EXPLICIT_DENSE_SOLVER
:canonical: better_robot.optim.temporal.LinearizationReason.EXPLICIT_DENSE_SOLVER
:value: >
   'explicit_dense_solver'

```{autodoc2-docstring} better_robot.optim.temporal.LinearizationReason.EXPLICIT_DENSE_SOLVER
```

````

`````

`````{py:class} TemporalAnalysis
:canonical: better_robot.optim.temporal.TemporalAnalysis

```{autodoc2-docstring} better_robot.optim.temporal.TemporalAnalysis
```

````{py:attribute} direct_eligible
:canonical: better_robot.optim.temporal.TemporalAnalysis.direct_eligible
:type: bool
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.TemporalAnalysis.direct_eligible
```

````

````{py:attribute} reason
:canonical: better_robot.optim.temporal.TemporalAnalysis.reason
:type: better_robot.optim.temporal.LinearizationReason
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.TemporalAnalysis.reason
```

````

````{py:attribute} detail
:canonical: better_robot.optim.temporal.TemporalAnalysis.detail
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.TemporalAnalysis.detail
```

````

````{py:attribute} variable_name
:canonical: better_robot.optim.temporal.TemporalAnalysis.variable_name
:type: str | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.TemporalAnalysis.variable_name
```

````

````{py:attribute} time_length
:canonical: better_robot.optim.temporal.TemporalAnalysis.time_length
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.TemporalAnalysis.time_length
```

````

````{py:attribute} tangent_width
:canonical: better_robot.optim.temporal.TemporalAnalysis.tangent_width
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.TemporalAnalysis.tangent_width
```

````

````{py:attribute} bandwidth
:canonical: better_robot.optim.temporal.TemporalAnalysis.bandwidth
:type: int
:value: >
   0

```{autodoc2-docstring} better_robot.optim.temporal.TemporalAnalysis.bandwidth
```

````

````{py:attribute} patterns
:canonical: better_robot.optim.temporal.TemporalAnalysis.patterns
:type: tuple[tuple[str, better_robot.residuals.structure.TemporalPattern], ...]
:value: >
   ()

```{autodoc2-docstring} better_robot.optim.temporal.TemporalAnalysis.patterns
```

````

`````

`````{py:class} BlockBandedMatrix
:canonical: better_robot.optim.temporal.BlockBandedMatrix

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix
```

````{py:attribute} bands
:canonical: better_robot.optim.temporal.BlockBandedMatrix.bands
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.bands
```

````

````{py:attribute} bandwidth
:canonical: better_robot.optim.temporal.BlockBandedMatrix.bandwidth
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.bandwidth
```

````

````{py:property} batch_shape
:canonical: better_robot.optim.temporal.BlockBandedMatrix.batch_shape
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.batch_shape
```

````

````{py:property} time_length
:canonical: better_robot.optim.temporal.BlockBandedMatrix.time_length
:type: int

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.time_length
```

````

````{py:property} block_size
:canonical: better_robot.optim.temporal.BlockBandedMatrix.block_size
:type: int

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.block_size
```

````

````{py:property} size
:canonical: better_robot.optim.temporal.BlockBandedMatrix.size
:type: int

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.size
```

````

````{py:property} finite
:canonical: better_robot.optim.temporal.BlockBandedMatrix.finite
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.finite
```

````

````{py:method} diagonal() -> torch.Tensor
:canonical: better_robot.optim.temporal.BlockBandedMatrix.diagonal

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.diagonal
```

````

````{py:method} densify() -> torch.Tensor
:canonical: better_robot.optim.temporal.BlockBandedMatrix.densify

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.densify
```

````

````{py:method} matvec(vector: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.temporal.BlockBandedMatrix.matvec

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.matvec
```

````

````{py:method} add_diagonal(diagonal: torch.Tensor) -> better_robot.optim.temporal.BlockBandedMatrix
:canonical: better_robot.optim.temporal.BlockBandedMatrix.add_diagonal

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.add_diagonal
```

````

````{py:method} restricted(movable: torch.Tensor, diagonal: torch.Tensor) -> better_robot.optim.temporal.BlockBandedMatrix
:canonical: better_robot.optim.temporal.BlockBandedMatrix.restricted

```{autodoc2-docstring} better_robot.optim.temporal.BlockBandedMatrix.restricted
```

````

`````

````{py:function} analyze_temporal_problem(problem: better_robot.optim.problem.Problem) -> better_robot.optim.temporal.TemporalAnalysis
:canonical: better_robot.optim.temporal.analyze_temporal_problem

```{autodoc2-docstring} better_robot.optim.temporal.analyze_temporal_problem
```
````

`````{py:class} StructuredNormal
:canonical: better_robot.optim.temporal.StructuredNormal

```{autodoc2-docstring} better_robot.optim.temporal.StructuredNormal
```

````{py:attribute} gradient
:canonical: better_robot.optim.temporal.StructuredNormal.gradient
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.StructuredNormal.gradient
```

````

````{py:attribute} normal_diagonal
:canonical: better_robot.optim.temporal.StructuredNormal.normal_diagonal
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.StructuredNormal.normal_diagonal
```

````

````{py:attribute} normal
:canonical: better_robot.optim.temporal.StructuredNormal.normal
:type: better_robot.optim.temporal.BlockBandedMatrix
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.StructuredNormal.normal
```

````

````{py:attribute} finite
:canonical: better_robot.optim.temporal.StructuredNormal.finite
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.temporal.StructuredNormal.finite
```

````

````{py:method} jvp(vector: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.temporal.StructuredNormal.jvp

```{autodoc2-docstring} better_robot.optim.temporal.StructuredNormal.jvp
```

````

````{py:method} vjp(cotangent: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.temporal.StructuredNormal.vjp

```{autodoc2-docstring} better_robot.optim.temporal.StructuredNormal.vjp
```

````

````{py:method} normal_matvec(vector: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.temporal.StructuredNormal.normal_matvec

```{autodoc2-docstring} better_robot.optim.temporal.StructuredNormal.normal_matvec
```

````

`````

````{py:function} assemble_structured_normal(problem: better_robot.optim.problem.Problem, *, batch_shape: tuple[int, ...], row_scale: torch.Tensor | None = None, residual: torch.Tensor | None = None, create_graph: bool = False, validate_runtime: bool = True) -> better_robot.optim.temporal.StructuredNormal
:canonical: better_robot.optim.temporal.assemble_structured_normal

```{autodoc2-docstring} better_robot.optim.temporal.assemble_structured_normal
```
````
