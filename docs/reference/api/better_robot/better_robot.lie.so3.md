# {py:mod}`better_robot.lie.so3`

```{py:module} better_robot.lie.so3
```

```{autodoc2-docstring} better_robot.lie.so3
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`identity <better_robot.lie.so3.identity>`
  - ```{autodoc2-docstring} better_robot.lie.so3.identity
    :summary:
    ```
* - {py:obj}`compose <better_robot.lie.so3.compose>`
  - ```{autodoc2-docstring} better_robot.lie.so3.compose
    :summary:
    ```
* - {py:obj}`inverse <better_robot.lie.so3.inverse>`
  - ```{autodoc2-docstring} better_robot.lie.so3.inverse
    :summary:
    ```
* - {py:obj}`log <better_robot.lie.so3.log>`
  - ```{autodoc2-docstring} better_robot.lie.so3.log
    :summary:
    ```
* - {py:obj}`exp <better_robot.lie.so3.exp>`
  - ```{autodoc2-docstring} better_robot.lie.so3.exp
    :summary:
    ```
* - {py:obj}`act <better_robot.lie.so3.act>`
  - ```{autodoc2-docstring} better_robot.lie.so3.act
    :summary:
    ```
* - {py:obj}`adjoint <better_robot.lie.so3.adjoint>`
  - ```{autodoc2-docstring} better_robot.lie.so3.adjoint
    :summary:
    ```
* - {py:obj}`from_matrix <better_robot.lie.so3.from_matrix>`
  - ```{autodoc2-docstring} better_robot.lie.so3.from_matrix
    :summary:
    ```
* - {py:obj}`to_matrix <better_robot.lie.so3.to_matrix>`
  - ```{autodoc2-docstring} better_robot.lie.so3.to_matrix
    :summary:
    ```
* - {py:obj}`from_axis_angle <better_robot.lie.so3.from_axis_angle>`
  - ```{autodoc2-docstring} better_robot.lie.so3.from_axis_angle
    :summary:
    ```
* - {py:obj}`normalize <better_robot.lie.so3.normalize>`
  - ```{autodoc2-docstring} better_robot.lie.so3.normalize
    :summary:
    ```
* - {py:obj}`slerp <better_robot.lie.so3.slerp>`
  - ```{autodoc2-docstring} better_robot.lie.so3.slerp
    :summary:
    ```
````

### API

````{py:function} identity(*, batch_shape: tuple[int, ...] = (), device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor
:canonical: better_robot.lie.so3.identity

```{autodoc2-docstring} better_robot.lie.so3.identity
```
````

````{py:function} compose(a: torch.Tensor, b: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.so3.compose

```{autodoc2-docstring} better_robot.lie.so3.compose
```
````

````{py:function} inverse(q: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.so3.inverse

```{autodoc2-docstring} better_robot.lie.so3.inverse
```
````

````{py:function} log(q: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.so3.log

```{autodoc2-docstring} better_robot.lie.so3.log
```
````

````{py:function} exp(w: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.so3.exp

```{autodoc2-docstring} better_robot.lie.so3.exp
```
````

````{py:function} act(q: torch.Tensor, p: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.so3.act

```{autodoc2-docstring} better_robot.lie.so3.act
```
````

````{py:function} adjoint(q: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.so3.adjoint

```{autodoc2-docstring} better_robot.lie.so3.adjoint
```
````

````{py:function} from_matrix(R: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.so3.from_matrix

```{autodoc2-docstring} better_robot.lie.so3.from_matrix
```
````

````{py:function} to_matrix(q: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.so3.to_matrix

```{autodoc2-docstring} better_robot.lie.so3.to_matrix
```
````

````{py:function} from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.so3.from_axis_angle

```{autodoc2-docstring} better_robot.lie.so3.from_axis_angle
```
````

````{py:function} normalize(q: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor
:canonical: better_robot.lie.so3.normalize

```{autodoc2-docstring} better_robot.lie.so3.normalize
```
````

````{py:function} slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor
:canonical: better_robot.lie.so3.slerp

```{autodoc2-docstring} better_robot.lie.so3.slerp
```
````
