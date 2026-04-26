# {py:mod}`better_robot.optim.kernels.huber`

```{py:module} better_robot.optim.kernels.huber
```

```{autodoc2-docstring} better_robot.optim.kernels.huber
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Huber <better_robot.optim.kernels.huber.Huber>`
  - ```{autodoc2-docstring} better_robot.optim.kernels.huber.Huber
    :summary:
    ```
````

### API

`````{py:class} Huber(*, delta: float = 1.0)
:canonical: better_robot.optim.kernels.huber.Huber

```{autodoc2-docstring} better_robot.optim.kernels.huber.Huber
```

````{py:method} rho(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.huber.Huber.rho

```{autodoc2-docstring} better_robot.optim.kernels.huber.Huber.rho
```

````

````{py:method} weight(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.huber.Huber.weight

```{autodoc2-docstring} better_robot.optim.kernels.huber.Huber.weight
```

````

`````
