# {py:mod}`better_robot.backends.torch_native.lie_ops`

```{py:module} better_robot.backends.torch_native.lie_ops
```

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TorchNativeLieOps <better_robot.backends.torch_native.lie_ops.TorchNativeLieOps>`
  - ```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps
    :summary:
    ```
````

### API

`````{py:class} TorchNativeLieOps
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps
```

````{py:method} se3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_compose

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_compose
```

````

````{py:method} se3_inverse(t: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_inverse

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_inverse
```

````

````{py:method} se3_log(t: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_log

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_log
```

````

````{py:method} se3_exp(v: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_exp

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_exp
```

````

````{py:method} se3_act(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_act

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_act
```

````

````{py:method} se3_adjoint(t: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_adjoint

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_adjoint
```

````

````{py:method} se3_adjoint_inv(t: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_adjoint_inv

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_adjoint_inv
```

````

````{py:method} se3_normalize(t: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_normalize

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.se3_normalize
```

````

````{py:method} so3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_compose

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_compose
```

````

````{py:method} so3_inverse(q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_inverse

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_inverse
```

````

````{py:method} so3_log(q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_log

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_log
```

````

````{py:method} so3_exp(w: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_exp

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_exp
```

````

````{py:method} so3_act(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_act

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_act
```

````

````{py:method} so3_to_matrix(q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_to_matrix

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_to_matrix
```

````

````{py:method} so3_from_matrix(R: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_from_matrix

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_from_matrix
```

````

````{py:method} so3_normalize(q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_normalize

```{autodoc2-docstring} better_robot.backends.torch_native.lie_ops.TorchNativeLieOps.so3_normalize
```

````

`````
