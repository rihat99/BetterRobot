# {py:mod}`better_robot.lie.types`

```{py:module} better_robot.lie.types
```

```{autodoc2-docstring} better_robot.lie.types
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SO3 <better_robot.lie.types.SO3>`
  - ```{autodoc2-docstring} better_robot.lie.types.SO3
    :summary:
    ```
* - {py:obj}`SE3 <better_robot.lie.types.SE3>`
  - ```{autodoc2-docstring} better_robot.lie.types.SE3
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Pose <better_robot.lie.types.Pose>`
  - ```{autodoc2-docstring} better_robot.lie.types.Pose
    :summary:
    ```
````

### API

`````{py:class} SO3
:canonical: better_robot.lie.types.SO3

```{autodoc2-docstring} better_robot.lie.types.SO3
```

````{py:attribute} tensor
:canonical: better_robot.lie.types.SO3.tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.lie.types.SO3.tensor
```

````

````{py:method} identity(*, batch_shape: tuple[int, ...] = (), device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> better_robot.lie.types.SO3
:canonical: better_robot.lie.types.SO3.identity
:classmethod:

```{autodoc2-docstring} better_robot.lie.types.SO3.identity
```

````

````{py:method} exp(w: torch.Tensor, *, backend: Backend | None = None) -> better_robot.lie.types.SO3
:canonical: better_robot.lie.types.SO3.exp
:classmethod:

```{autodoc2-docstring} better_robot.lie.types.SO3.exp
```

````

````{py:method} from_matrix(R: torch.Tensor, *, backend: Backend | None = None) -> better_robot.lie.types.SO3
:canonical: better_robot.lie.types.SO3.from_matrix
:classmethod:

```{autodoc2-docstring} better_robot.lie.types.SO3.from_matrix
```

````

````{py:method} inverse(*, backend: Backend | None = None) -> better_robot.lie.types.SO3
:canonical: better_robot.lie.types.SO3.inverse

```{autodoc2-docstring} better_robot.lie.types.SO3.inverse
```

````

````{py:method} log(*, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.types.SO3.log

```{autodoc2-docstring} better_robot.lie.types.SO3.log
```

````

````{py:method} to_matrix(*, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.types.SO3.to_matrix

```{autodoc2-docstring} better_robot.lie.types.SO3.to_matrix
```

````

````{py:method} normalize(*, backend: Backend | None = None) -> better_robot.lie.types.SO3
:canonical: better_robot.lie.types.SO3.normalize

```{autodoc2-docstring} better_robot.lie.types.SO3.normalize
```

````

````{py:method} compose(other: better_robot.lie.types.SO3, *, backend: Backend | None = None) -> better_robot.lie.types.SO3
:canonical: better_robot.lie.types.SO3.compose

```{autodoc2-docstring} better_robot.lie.types.SO3.compose
```

````

````{py:method} act(p: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.types.SO3.act

```{autodoc2-docstring} better_robot.lie.types.SO3.act
```

````

`````

`````{py:class} SE3
:canonical: better_robot.lie.types.SE3

```{autodoc2-docstring} better_robot.lie.types.SE3
```

````{py:attribute} tensor
:canonical: better_robot.lie.types.SE3.tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.lie.types.SE3.tensor
```

````

````{py:method} identity(*, batch_shape: tuple[int, ...] = (), device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> better_robot.lie.types.SE3
:canonical: better_robot.lie.types.SE3.identity
:classmethod:

```{autodoc2-docstring} better_robot.lie.types.SE3.identity
```

````

````{py:method} exp(xi: torch.Tensor, *, backend: Backend | None = None) -> better_robot.lie.types.SE3
:canonical: better_robot.lie.types.SE3.exp
:classmethod:

```{autodoc2-docstring} better_robot.lie.types.SE3.exp
```

````

````{py:property} translation
:canonical: better_robot.lie.types.SE3.translation
:type: torch.Tensor

```{autodoc2-docstring} better_robot.lie.types.SE3.translation
```

````

````{py:property} rotation
:canonical: better_robot.lie.types.SE3.rotation
:type: better_robot.lie.types.SO3

```{autodoc2-docstring} better_robot.lie.types.SE3.rotation
```

````

````{py:method} inverse(*, backend: Backend | None = None) -> better_robot.lie.types.SE3
:canonical: better_robot.lie.types.SE3.inverse

```{autodoc2-docstring} better_robot.lie.types.SE3.inverse
```

````

````{py:method} log(*, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.types.SE3.log

```{autodoc2-docstring} better_robot.lie.types.SE3.log
```

````

````{py:method} adjoint(*, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.types.SE3.adjoint

```{autodoc2-docstring} better_robot.lie.types.SE3.adjoint
```

````

````{py:method} adjoint_inv(*, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.types.SE3.adjoint_inv

```{autodoc2-docstring} better_robot.lie.types.SE3.adjoint_inv
```

````

````{py:method} normalize(*, backend: Backend | None = None) -> better_robot.lie.types.SE3
:canonical: better_robot.lie.types.SE3.normalize

```{autodoc2-docstring} better_robot.lie.types.SE3.normalize
```

````

````{py:method} compose(other: better_robot.lie.types.SE3, *, backend: Backend | None = None) -> better_robot.lie.types.SE3
:canonical: better_robot.lie.types.SE3.compose

```{autodoc2-docstring} better_robot.lie.types.SE3.compose
```

````

````{py:method} act(p: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.types.SE3.act

```{autodoc2-docstring} better_robot.lie.types.SE3.act
```

````

`````

````{py:data} Pose
:canonical: better_robot.lie.types.Pose
:value: >
   None

```{autodoc2-docstring} better_robot.lie.types.Pose
```

````
