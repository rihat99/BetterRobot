# {py:mod}`better_robot.optim.manifolds`

```{py:module} better_robot.optim.manifolds
```

```{autodoc2-docstring} better_robot.optim.manifolds
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Bounds <better_robot.optim.manifolds.Bounds>`
  - ```{autodoc2-docstring} better_robot.optim.manifolds.Bounds
    :summary:
    ```
* - {py:obj}`Manifold <better_robot.optim.manifolds.Manifold>`
  -
* - {py:obj}`Euclidean <better_robot.optim.manifolds.Euclidean>`
  - ```{autodoc2-docstring} better_robot.optim.manifolds.Euclidean
    :summary:
    ```
* - {py:obj}`SO3Manifold <better_robot.optim.manifolds.SO3Manifold>`
  - ```{autodoc2-docstring} better_robot.optim.manifolds.SO3Manifold
    :summary:
    ```
* - {py:obj}`SE3Manifold <better_robot.optim.manifolds.SE3Manifold>`
  - ```{autodoc2-docstring} better_robot.optim.manifolds.SE3Manifold
    :summary:
    ```
* - {py:obj}`RobotConfig <better_robot.optim.manifolds.RobotConfig>`
  - ```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig
    :summary:
    ```
````

### API

`````{py:class} Bounds
:canonical: better_robot.optim.manifolds.Bounds

```{autodoc2-docstring} better_robot.optim.manifolds.Bounds
```

````{py:attribute} lower
:canonical: better_robot.optim.manifolds.Bounds.lower
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.manifolds.Bounds.lower
```

````

````{py:attribute} upper
:canonical: better_robot.optim.manifolds.Bounds.upper
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.manifolds.Bounds.upper
```

````

`````

`````{py:class} Manifold
:canonical: better_robot.optim.manifolds.Manifold

Bases: {py:obj}`typing.Protocol`

````{py:method} retract(x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.manifolds.Manifold.retract

```{autodoc2-docstring} better_robot.optim.manifolds.Manifold.retract
```

````

````{py:method} difference(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.manifolds.Manifold.difference

```{autodoc2-docstring} better_robot.optim.manifolds.Manifold.difference
```

````

````{py:method} tangent_dim(shape: tuple[int, ...]) -> int
:canonical: better_robot.optim.manifolds.Manifold.tangent_dim

```{autodoc2-docstring} better_robot.optim.manifolds.Manifold.tangent_dim
```

````

````{py:method} project(x: torch.Tensor, bounds: better_robot.optim.manifolds.Bounds | None) -> torch.Tensor
:canonical: better_robot.optim.manifolds.Manifold.project

```{autodoc2-docstring} better_robot.optim.manifolds.Manifold.project
```

````

````{py:method} validate_bounds(bounds: better_robot.optim.manifolds.Bounds | None, *, name: str) -> None
:canonical: better_robot.optim.manifolds.Manifold.validate_bounds

```{autodoc2-docstring} better_robot.optim.manifolds.Manifold.validate_bounds
```

````

`````

`````{py:class} Euclidean
:canonical: better_robot.optim.manifolds.Euclidean

```{autodoc2-docstring} better_robot.optim.manifolds.Euclidean
```

````{py:method} retract(x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.manifolds.Euclidean.retract

```{autodoc2-docstring} better_robot.optim.manifolds.Euclidean.retract
```

````

````{py:method} difference(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.manifolds.Euclidean.difference

```{autodoc2-docstring} better_robot.optim.manifolds.Euclidean.difference
```

````

````{py:method} tangent_dim(shape: tuple[int, ...]) -> int
:canonical: better_robot.optim.manifolds.Euclidean.tangent_dim

```{autodoc2-docstring} better_robot.optim.manifolds.Euclidean.tangent_dim
```

````

````{py:method} project(x: torch.Tensor, bounds: better_robot.optim.manifolds.Bounds | None) -> torch.Tensor
:canonical: better_robot.optim.manifolds.Euclidean.project

```{autodoc2-docstring} better_robot.optim.manifolds.Euclidean.project
```

````

````{py:method} validate_bounds(bounds: better_robot.optim.manifolds.Bounds | None, *, name: str) -> None
:canonical: better_robot.optim.manifolds.Euclidean.validate_bounds

```{autodoc2-docstring} better_robot.optim.manifolds.Euclidean.validate_bounds
```

````

`````

`````{py:class} SO3Manifold
:canonical: better_robot.optim.manifolds.SO3Manifold

```{autodoc2-docstring} better_robot.optim.manifolds.SO3Manifold
```

````{py:method} retract(x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.manifolds.SO3Manifold.retract

```{autodoc2-docstring} better_robot.optim.manifolds.SO3Manifold.retract
```

````

````{py:method} difference(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.manifolds.SO3Manifold.difference

```{autodoc2-docstring} better_robot.optim.manifolds.SO3Manifold.difference
```

````

````{py:method} tangent_dim(shape: tuple[int, ...]) -> int
:canonical: better_robot.optim.manifolds.SO3Manifold.tangent_dim

```{autodoc2-docstring} better_robot.optim.manifolds.SO3Manifold.tangent_dim
```

````

````{py:method} project(x: torch.Tensor, bounds: better_robot.optim.manifolds.Bounds | None) -> torch.Tensor
:canonical: better_robot.optim.manifolds.SO3Manifold.project

```{autodoc2-docstring} better_robot.optim.manifolds.SO3Manifold.project
```

````

````{py:method} validate_bounds(bounds: better_robot.optim.manifolds.Bounds | None, *, name: str) -> None
:canonical: better_robot.optim.manifolds.SO3Manifold.validate_bounds

```{autodoc2-docstring} better_robot.optim.manifolds.SO3Manifold.validate_bounds
```

````

`````

`````{py:class} SE3Manifold
:canonical: better_robot.optim.manifolds.SE3Manifold

```{autodoc2-docstring} better_robot.optim.manifolds.SE3Manifold
```

````{py:method} retract(x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.manifolds.SE3Manifold.retract

```{autodoc2-docstring} better_robot.optim.manifolds.SE3Manifold.retract
```

````

````{py:method} difference(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.manifolds.SE3Manifold.difference

```{autodoc2-docstring} better_robot.optim.manifolds.SE3Manifold.difference
```

````

````{py:method} tangent_dim(shape: tuple[int, ...]) -> int
:canonical: better_robot.optim.manifolds.SE3Manifold.tangent_dim

```{autodoc2-docstring} better_robot.optim.manifolds.SE3Manifold.tangent_dim
```

````

````{py:method} project(x: torch.Tensor, bounds: better_robot.optim.manifolds.Bounds | None) -> torch.Tensor
:canonical: better_robot.optim.manifolds.SE3Manifold.project

```{autodoc2-docstring} better_robot.optim.manifolds.SE3Manifold.project
```

````

````{py:method} validate_bounds(bounds: better_robot.optim.manifolds.Bounds | None, *, name: str) -> None
:canonical: better_robot.optim.manifolds.SE3Manifold.validate_bounds

```{autodoc2-docstring} better_robot.optim.manifolds.SE3Manifold.validate_bounds
```

````

`````

`````{py:class} RobotConfig
:canonical: better_robot.optim.manifolds.RobotConfig

```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig
```

````{py:attribute} model
:canonical: better_robot.optim.manifolds.RobotConfig.model
:type: better_robot.data_model.model.Model
:value: >
   None

```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig.model
```

````

````{py:method} retract(x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.manifolds.RobotConfig.retract

```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig.retract
```

````

````{py:method} difference(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor
:canonical: better_robot.optim.manifolds.RobotConfig.difference

```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig.difference
```

````

````{py:method} tangent_dim(shape: tuple[int, ...]) -> int
:canonical: better_robot.optim.manifolds.RobotConfig.tangent_dim

```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig.tangent_dim
```

````

````{py:property} box_mask
:canonical: better_robot.optim.manifolds.RobotConfig.box_mask
:type: torch.Tensor

```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig.box_mask
```

````

````{py:method} joint_bounds() -> better_robot.optim.manifolds.Bounds
:canonical: better_robot.optim.manifolds.RobotConfig.joint_bounds

```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig.joint_bounds
```

````

````{py:property} unit_coordinate_slices
:canonical: better_robot.optim.manifolds.RobotConfig.unit_coordinate_slices
:type: tuple[slice, ...]

```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig.unit_coordinate_slices
```

````

````{py:method} validate_bounds(bounds: better_robot.optim.manifolds.Bounds | None, *, name: str) -> None
:canonical: better_robot.optim.manifolds.RobotConfig.validate_bounds

```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig.validate_bounds
```

````

````{py:method} project(x: torch.Tensor, bounds: better_robot.optim.manifolds.Bounds | None) -> torch.Tensor
:canonical: better_robot.optim.manifolds.RobotConfig.project

```{autodoc2-docstring} better_robot.optim.manifolds.RobotConfig.project
```

````

`````
