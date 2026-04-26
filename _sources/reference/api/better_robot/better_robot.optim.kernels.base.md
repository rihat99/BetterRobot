# {py:mod}`better_robot.optim.kernels.base`

```{py:module} better_robot.optim.kernels.base
```

```{autodoc2-docstring} better_robot.optim.kernels.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RobustKernel <better_robot.optim.kernels.base.RobustKernel>`
  - ```{autodoc2-docstring} better_robot.optim.kernels.base.RobustKernel
    :summary:
    ```
````

### API

`````{py:class} RobustKernel
:canonical: better_robot.optim.kernels.base.RobustKernel

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.optim.kernels.base.RobustKernel
```

````{py:method} weight(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.base.RobustKernel.weight

```{autodoc2-docstring} better_robot.optim.kernels.base.RobustKernel.weight
```

````

`````
