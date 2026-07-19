# {py:mod}`better_robot.optim.implicit`

```{py:module} better_robot.optim.implicit
```

```{autodoc2-docstring} better_robot.optim.implicit
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImplicitDiffConfig <better_robot.optim.implicit.ImplicitDiffConfig>`
  - ```{autodoc2-docstring} better_robot.optim.implicit.ImplicitDiffConfig
    :summary:
    ```
````

### API

````{py:exception} ImplicitDifferentiationError(message: str, *, invalid_indices: tuple[tuple[int, ...], ...] = (), statuses: tuple[str, ...] = ())
:canonical: better_robot.optim.implicit.ImplicitDifferentiationError

Bases: {py:obj}`RuntimeError`

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitDifferentiationError
```

````

`````{py:class} ImplicitDiffConfig
:canonical: better_robot.optim.implicit.ImplicitDiffConfig

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitDiffConfig
```

````{py:attribute} max_dense_tangent_dim
:canonical: better_robot.optim.implicit.ImplicitDiffConfig.max_dense_tangent_dim
:type: int
:value: >
   512

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitDiffConfig.max_dense_tangent_dim
```

````

````{py:attribute} allow_banded_dense_backward
:canonical: better_robot.optim.implicit.ImplicitDiffConfig.allow_banded_dense_backward
:type: bool
:value: >
   False

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitDiffConfig.allow_banded_dense_backward
```

````

`````
