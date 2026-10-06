# {py:mod}`better_robot.optim2.manifold`

```{py:module} better_robot.optim2.manifold
```

```{autodoc2-docstring} better_robot.optim2.manifold
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Manifold <better_robot.optim2.manifold.Manifold>`
  - ```{autodoc2-docstring} better_robot.optim2.manifold.Manifold
    :summary:
    ```
* - {py:obj}`Euclidean <better_robot.optim2.manifold.Euclidean>`
  - ```{autodoc2-docstring} better_robot.optim2.manifold.Euclidean
    :summary:
    ```
````

### API

`````{py:class} Manifold
:canonical: better_robot.optim2.manifold.Manifold

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.optim2.manifold.Manifold
```

````{py:method} tangent_dim(x: torch.Tensor) -> int
:canonical: better_robot.optim2.manifold.Manifold.tangent_dim

```{autodoc2-docstring} better_robot.optim2.manifold.Manifold.tangent_dim
```

````

````{py:method} retract(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim2.manifold.Manifold.retract

```{autodoc2-docstring} better_robot.optim2.manifold.Manifold.retract
```

````

````{py:method} project(x: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim2.manifold.Manifold.project

```{autodoc2-docstring} better_robot.optim2.manifold.Manifold.project
```

````

````{py:method} tangent_bounds(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.optim2.manifold.Manifold.tangent_bounds

```{autodoc2-docstring} better_robot.optim2.manifold.Manifold.tangent_bounds
```

````

`````

`````{py:class} Euclidean
:canonical: better_robot.optim2.manifold.Euclidean

```{autodoc2-docstring} better_robot.optim2.manifold.Euclidean
```

````{py:method} tangent_dim(x: torch.Tensor) -> int
:canonical: better_robot.optim2.manifold.Euclidean.tangent_dim

```{autodoc2-docstring} better_robot.optim2.manifold.Euclidean.tangent_dim
```

````

````{py:method} retract(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim2.manifold.Euclidean.retract

```{autodoc2-docstring} better_robot.optim2.manifold.Euclidean.retract
```

````

````{py:method} project(x: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim2.manifold.Euclidean.project

```{autodoc2-docstring} better_robot.optim2.manifold.Euclidean.project
```

````

````{py:method} tangent_bounds(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.optim2.manifold.Euclidean.tangent_bounds

```{autodoc2-docstring} better_robot.optim2.manifold.Euclidean.tangent_bounds
```

````

`````
