# {py:mod}`better_robot.lie.tangents`

```{py:module} better_robot.lie.tangents
```

```{autodoc2-docstring} better_robot.lie.tangents
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`hat_so3 <better_robot.lie.tangents.hat_so3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.hat_so3
    :summary:
    ```
* - {py:obj}`vee_so3 <better_robot.lie.tangents.vee_so3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.vee_so3
    :summary:
    ```
* - {py:obj}`hat_se3 <better_robot.lie.tangents.hat_se3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.hat_se3
    :summary:
    ```
* - {py:obj}`vee_se3 <better_robot.lie.tangents.vee_se3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.vee_se3
    :summary:
    ```
* - {py:obj}`right_jacobian_so3 <better_robot.lie.tangents.right_jacobian_so3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.right_jacobian_so3
    :summary:
    ```
* - {py:obj}`right_jacobian_inv_so3 <better_robot.lie.tangents.right_jacobian_inv_so3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.right_jacobian_inv_so3
    :summary:
    ```
* - {py:obj}`left_jacobian_so3 <better_robot.lie.tangents.left_jacobian_so3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.left_jacobian_so3
    :summary:
    ```
* - {py:obj}`left_jacobian_inv_so3 <better_robot.lie.tangents.left_jacobian_inv_so3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.left_jacobian_inv_so3
    :summary:
    ```
* - {py:obj}`right_jacobian_se3 <better_robot.lie.tangents.right_jacobian_se3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.right_jacobian_se3
    :summary:
    ```
* - {py:obj}`right_jacobian_inv_se3 <better_robot.lie.tangents.right_jacobian_inv_se3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.right_jacobian_inv_se3
    :summary:
    ```
* - {py:obj}`left_jacobian_se3 <better_robot.lie.tangents.left_jacobian_se3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.left_jacobian_se3
    :summary:
    ```
* - {py:obj}`left_jacobian_inv_se3 <better_robot.lie.tangents.left_jacobian_inv_se3>`
  - ```{autodoc2-docstring} better_robot.lie.tangents.left_jacobian_inv_se3
    :summary:
    ```
````

### API

````{py:function} hat_so3(w: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.hat_so3

```{autodoc2-docstring} better_robot.lie.tangents.hat_so3
```
````

````{py:function} vee_so3(W: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.vee_so3

```{autodoc2-docstring} better_robot.lie.tangents.vee_so3
```
````

````{py:function} hat_se3(xi: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.hat_se3

```{autodoc2-docstring} better_robot.lie.tangents.hat_se3
```
````

````{py:function} vee_se3(X: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.vee_se3

```{autodoc2-docstring} better_robot.lie.tangents.vee_se3
```
````

````{py:function} right_jacobian_so3(omega: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.right_jacobian_so3

```{autodoc2-docstring} better_robot.lie.tangents.right_jacobian_so3
```
````

````{py:function} right_jacobian_inv_so3(omega: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.right_jacobian_inv_so3

```{autodoc2-docstring} better_robot.lie.tangents.right_jacobian_inv_so3
```
````

````{py:function} left_jacobian_so3(omega: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.left_jacobian_so3

```{autodoc2-docstring} better_robot.lie.tangents.left_jacobian_so3
```
````

````{py:function} left_jacobian_inv_so3(omega: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.left_jacobian_inv_so3

```{autodoc2-docstring} better_robot.lie.tangents.left_jacobian_inv_so3
```
````

````{py:function} right_jacobian_se3(xi: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.right_jacobian_se3

```{autodoc2-docstring} better_robot.lie.tangents.right_jacobian_se3
```
````

````{py:function} right_jacobian_inv_se3(xi: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.right_jacobian_inv_se3

```{autodoc2-docstring} better_robot.lie.tangents.right_jacobian_inv_se3
```
````

````{py:function} left_jacobian_se3(xi: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.left_jacobian_se3

```{autodoc2-docstring} better_robot.lie.tangents.left_jacobian_se3
```
````

````{py:function} left_jacobian_inv_se3(xi: torch.Tensor) -> torch.Tensor
:canonical: better_robot.lie.tangents.left_jacobian_inv_se3

```{autodoc2-docstring} better_robot.lie.tangents.left_jacobian_inv_se3
```
````
