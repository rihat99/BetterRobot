# {py:mod}`better_robot.optim.kernels`

```{py:module} better_robot.optim.kernels
```

```{autodoc2-docstring} better_robot.optim.kernels
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RobustKernel <better_robot.optim.kernels.RobustKernel>`
  - ```{autodoc2-docstring} better_robot.optim.kernels.RobustKernel
    :summary:
    ```
* - {py:obj}`L2 <better_robot.optim.kernels.L2>`
  - ```{autodoc2-docstring} better_robot.optim.kernels.L2
    :summary:
    ```
* - {py:obj}`Huber <better_robot.optim.kernels.Huber>`
  - ```{autodoc2-docstring} better_robot.optim.kernels.Huber
    :summary:
    ```
* - {py:obj}`Cauchy <better_robot.optim.kernels.Cauchy>`
  - ```{autodoc2-docstring} better_robot.optim.kernels.Cauchy
    :summary:
    ```
* - {py:obj}`Tukey <better_robot.optim.kernels.Tukey>`
  - ```{autodoc2-docstring} better_robot.optim.kernels.Tukey
    :summary:
    ```
* - {py:obj}`GemanMcClure <better_robot.optim.kernels.GemanMcClure>`
  - ```{autodoc2-docstring} better_robot.optim.kernels.GemanMcClure
    :summary:
    ```
````

### API

`````{py:class} RobustKernel
:canonical: better_robot.optim.kernels.RobustKernel

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.optim.kernels.RobustKernel
```

````{py:method} rho(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.RobustKernel.rho

```{autodoc2-docstring} better_robot.optim.kernels.RobustKernel.rho
```

````

````{py:method} weight(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.RobustKernel.weight

```{autodoc2-docstring} better_robot.optim.kernels.RobustKernel.weight
```

````

`````

`````{py:class} L2
:canonical: better_robot.optim.kernels.L2

```{autodoc2-docstring} better_robot.optim.kernels.L2
```

````{py:method} rho(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.L2.rho

```{autodoc2-docstring} better_robot.optim.kernels.L2.rho
```

````

````{py:method} weight(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.L2.weight

```{autodoc2-docstring} better_robot.optim.kernels.L2.weight
```

````

`````

`````{py:class} Huber(*, delta: float = 1.0)
:canonical: better_robot.optim.kernels.Huber

```{autodoc2-docstring} better_robot.optim.kernels.Huber
```

````{py:method} rho(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.Huber.rho

```{autodoc2-docstring} better_robot.optim.kernels.Huber.rho
```

````

````{py:method} weight(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.Huber.weight

```{autodoc2-docstring} better_robot.optim.kernels.Huber.weight
```

````

`````

`````{py:class} Cauchy(*, c: float = 1.0)
:canonical: better_robot.optim.kernels.Cauchy

```{autodoc2-docstring} better_robot.optim.kernels.Cauchy
```

````{py:method} rho(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.Cauchy.rho

```{autodoc2-docstring} better_robot.optim.kernels.Cauchy.rho
```

````

````{py:method} weight(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.Cauchy.weight

```{autodoc2-docstring} better_robot.optim.kernels.Cauchy.weight
```

````

`````

`````{py:class} Tukey(*, c: float = 4.685)
:canonical: better_robot.optim.kernels.Tukey

```{autodoc2-docstring} better_robot.optim.kernels.Tukey
```

````{py:method} rho(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.Tukey.rho

```{autodoc2-docstring} better_robot.optim.kernels.Tukey.rho
```

````

````{py:method} weight(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.Tukey.weight

```{autodoc2-docstring} better_robot.optim.kernels.Tukey.weight
```

````

`````

`````{py:class} GemanMcClure(*, c: float = 1.0)
:canonical: better_robot.optim.kernels.GemanMcClure

```{autodoc2-docstring} better_robot.optim.kernels.GemanMcClure
```

````{py:method} rho(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.GemanMcClure.rho

```{autodoc2-docstring} better_robot.optim.kernels.GemanMcClure.rho
```

````

````{py:method} weight(squared_norm: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.kernels.GemanMcClure.weight

```{autodoc2-docstring} better_robot.optim.kernels.GemanMcClure.weight
```

````

`````
