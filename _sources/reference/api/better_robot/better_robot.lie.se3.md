# {py:mod}`better_robot.lie.se3`

```{py:module} better_robot.lie.se3
```

```{autodoc2-docstring} better_robot.lie.se3
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`identity <better_robot.lie.se3.identity>`
  - ```{autodoc2-docstring} better_robot.lie.se3.identity
    :summary:
    ```
* - {py:obj}`compose <better_robot.lie.se3.compose>`
  - ```{autodoc2-docstring} better_robot.lie.se3.compose
    :summary:
    ```
* - {py:obj}`inverse <better_robot.lie.se3.inverse>`
  - ```{autodoc2-docstring} better_robot.lie.se3.inverse
    :summary:
    ```
* - {py:obj}`log <better_robot.lie.se3.log>`
  - ```{autodoc2-docstring} better_robot.lie.se3.log
    :summary:
    ```
* - {py:obj}`exp <better_robot.lie.se3.exp>`
  - ```{autodoc2-docstring} better_robot.lie.se3.exp
    :summary:
    ```
* - {py:obj}`act <better_robot.lie.se3.act>`
  - ```{autodoc2-docstring} better_robot.lie.se3.act
    :summary:
    ```
* - {py:obj}`adjoint <better_robot.lie.se3.adjoint>`
  - ```{autodoc2-docstring} better_robot.lie.se3.adjoint
    :summary:
    ```
* - {py:obj}`adjoint_inv <better_robot.lie.se3.adjoint_inv>`
  - ```{autodoc2-docstring} better_robot.lie.se3.adjoint_inv
    :summary:
    ```
* - {py:obj}`from_axis_angle <better_robot.lie.se3.from_axis_angle>`
  - ```{autodoc2-docstring} better_robot.lie.se3.from_axis_angle
    :summary:
    ```
* - {py:obj}`from_translation <better_robot.lie.se3.from_translation>`
  - ```{autodoc2-docstring} better_robot.lie.se3.from_translation
    :summary:
    ```
* - {py:obj}`normalize <better_robot.lie.se3.normalize>`
  - ```{autodoc2-docstring} better_robot.lie.se3.normalize
    :summary:
    ```
* - {py:obj}`apply_base <better_robot.lie.se3.apply_base>`
  - ```{autodoc2-docstring} better_robot.lie.se3.apply_base
    :summary:
    ```
* - {py:obj}`sclerp <better_robot.lie.se3.sclerp>`
  - ```{autodoc2-docstring} better_robot.lie.se3.sclerp
    :summary:
    ```
````

### API

````{py:function} identity(*, batch_shape: tuple[int, ...] = (), device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor
:canonical: better_robot.lie.se3.identity

```{autodoc2-docstring} better_robot.lie.se3.identity
```
````

````{py:function} compose(a: torch.Tensor, b: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.se3.compose

```{autodoc2-docstring} better_robot.lie.se3.compose
```
````

````{py:function} inverse(t: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.se3.inverse

```{autodoc2-docstring} better_robot.lie.se3.inverse
```
````

````{py:function} log(t: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.se3.log

```{autodoc2-docstring} better_robot.lie.se3.log
```
````

````{py:function} exp(v: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.se3.exp

```{autodoc2-docstring} better_robot.lie.se3.exp
```
````

````{py:function} act(t: torch.Tensor, p: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.se3.act

```{autodoc2-docstring} better_robot.lie.se3.act
```
````

````{py:function} adjoint(t: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.se3.adjoint

```{autodoc2-docstring} better_robot.lie.se3.adjoint
```
````

````{py:function} adjoint_inv(t: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.se3.adjoint_inv

```{autodoc2-docstring} better_robot.lie.se3.adjoint_inv
```
````

````{py:function} from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.se3.from_axis_angle

```{autodoc2-docstring} better_robot.lie.se3.from_axis_angle
```
````

````{py:function} from_translation(axis: torch.Tensor, disp: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.se3.from_translation

```{autodoc2-docstring} better_robot.lie.se3.from_translation
```
````

````{py:function} normalize(t: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.se3.normalize

```{autodoc2-docstring} better_robot.lie.se3.normalize
```
````

````{py:function} apply_base(base: torch.Tensor, poses: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.se3.apply_base

```{autodoc2-docstring} better_robot.lie.se3.apply_base
```
````

````{py:function} sclerp(T1: torch.Tensor, T2: torch.Tensor, t: torch.Tensor | float, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.se3.sclerp

```{autodoc2-docstring} better_robot.lie.se3.sclerp
```
````
