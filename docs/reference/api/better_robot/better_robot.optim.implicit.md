# {py:mod}`better_robot.optim.implicit`

```{py:module} better_robot.optim.implicit
```

```{autodoc2-docstring} better_robot.optim.implicit
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImplicitDiffConfig <better_robot.optim.implicit.ImplicitDiffConfig>`
  - ```{autodoc2-docstring} better_robot.optim.implicit.ImplicitDiffConfig
    :summary:
    ```
* - {py:obj}`ImplicitTerminalState <better_robot.optim.implicit.ImplicitTerminalState>`
  - ```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`validate_implicit_input_roles <better_robot.optim.implicit.validate_implicit_input_roles>`
  - ```{autodoc2-docstring} better_robot.optim.implicit.validate_implicit_input_roles
    :summary:
    ```
* - {py:obj}`attach_implicit_gradients <better_robot.optim.implicit.attach_implicit_gradients>`
  - ```{autodoc2-docstring} better_robot.optim.implicit.attach_implicit_gradients
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ForwardLinearization <better_robot.optim.implicit.ForwardLinearization>`
  - ```{autodoc2-docstring} better_robot.optim.implicit.ForwardLinearization
    :summary:
    ```
````

### API

````{py:data} ForwardLinearization
:canonical: better_robot.optim.implicit.ForwardLinearization
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.implicit.ForwardLinearization
```

````

````{py:exception} ImplicitDifferentiationError(message: str, *, invalid_indices: tuple[tuple[int, ...], ...] = (), statuses: tuple[str, ...] = ())
:canonical: better_robot.optim.implicit.ImplicitDifferentiationError

Bases: {py:obj}`RuntimeError`

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitDifferentiationError
```

````

`````{py:class} ImplicitDiffConfig
:canonical: better_robot.optim.implicit.ImplicitDiffConfig

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitDiffConfig
```

````{py:attribute} max_dense_tangent_dim
:canonical: better_robot.optim.implicit.ImplicitDiffConfig.max_dense_tangent_dim
:type: int
:value: >
   512

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitDiffConfig.max_dense_tangent_dim
```

````

````{py:attribute} allow_banded_dense_backward
:canonical: better_robot.optim.implicit.ImplicitDiffConfig.allow_banded_dense_backward
:type: bool
:value: >
   False

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitDiffConfig.allow_banded_dense_backward
```

````

`````

`````{py:class} ImplicitTerminalState
:canonical: better_robot.optim.implicit.ImplicitTerminalState

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState
```

````{py:attribute} status
:canonical: better_robot.optim.implicit.ImplicitTerminalState.status
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState.status
```

````

````{py:attribute} implicit_valid
:canonical: better_robot.optim.implicit.ImplicitTerminalState.implicit_valid
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState.implicit_valid
```

````

````{py:attribute} active_mask
:canonical: better_robot.optim.implicit.ImplicitTerminalState.active_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState.active_mask
```

````

````{py:attribute} gradient
:canonical: better_robot.optim.implicit.ImplicitTerminalState.gradient
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState.gradient
```

````

````{py:attribute} projected_grad_norm
:canonical: better_robot.optim.implicit.ImplicitTerminalState.projected_grad_norm
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState.projected_grad_norm
```

````

````{py:attribute} bound_state_index
:canonical: better_robot.optim.implicit.ImplicitTerminalState.bound_state_index
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState.bound_state_index
```

````

````{py:attribute} bound_lower
:canonical: better_robot.optim.implicit.ImplicitTerminalState.bound_lower
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState.bound_lower
```

````

````{py:attribute} bound_upper
:canonical: better_robot.optim.implicit.ImplicitTerminalState.bound_upper
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState.bound_upper
```

````

````{py:attribute} bounded_mask
:canonical: better_robot.optim.implicit.ImplicitTerminalState.bounded_mask
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.implicit.ImplicitTerminalState.bounded_mask
```

````

`````

````{py:function} validate_implicit_input_roles(values: collections.abc.Mapping[str, torch.Tensor], problem: better_robot.optim.problem.Problem) -> None
:canonical: better_robot.optim.implicit.validate_implicit_input_roles

```{autodoc2-docstring} better_robot.optim.implicit.validate_implicit_input_roles
```
````

````{py:function} attach_implicit_gradients(terminal_values: collections.abc.Mapping[str, torch.Tensor], state: better_robot.optim.implicit.ImplicitTerminalState, problem: better_robot.optim.problem.Problem, *, default_kernel: better_robot.optim.kernels.RobustKernel | None = None, forward_linearization: better_robot.optim.implicit.ForwardLinearization = 'dense', config: better_robot.optim.implicit.ImplicitDiffConfig | None = None) -> better_robot.optim.variables.Values
:canonical: better_robot.optim.implicit.attach_implicit_gradients

```{autodoc2-docstring} better_robot.optim.implicit.attach_implicit_gradients
```
````
