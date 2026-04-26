# {py:mod}`better_robot.optim.kernels.tukey`

```{py:module} better_robot.optim.kernels.tukey
```

```{autodoc2-docstring} better_robot.optim.kernels.tukey
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Tukey <better_robot.optim.kernels.tukey.Tukey>`
  - ```{autodoc2-docstring} better_robot.optim.kernels.tukey.Tukey
    :summary:
    ```
````

### API

`````{py:class} Tukey(*, c: float = 4.685)
:canonical: better_robot.optim.kernels.tukey.Tukey

```{autodoc2-docstring} better_robot.optim.kernels.tukey.Tukey
```

````{py:method} rho(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.tukey.Tukey.rho

```{autodoc2-docstring} better_robot.optim.kernels.tukey.Tukey.rho
```

````

````{py:method} weight(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.tukey.Tukey.weight

```{autodoc2-docstring} better_robot.optim.kernels.tukey.Tukey.weight
```

````

`````
