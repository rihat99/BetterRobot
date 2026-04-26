# {py:mod}`better_robot.residuals.base`

```{py:module} better_robot.residuals.base
```

```{autodoc2-docstring} better_robot.residuals.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ResidualState <better_robot.residuals.base.ResidualState>`
  - ```{autodoc2-docstring} better_robot.residuals.base.ResidualState
    :summary:
    ```
* - {py:obj}`Residual <better_robot.residuals.base.Residual>`
  - ```{autodoc2-docstring} better_robot.residuals.base.Residual
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`default_apply_jac_transpose <better_robot.residuals.base.default_apply_jac_transpose>`
  - ```{autodoc2-docstring} better_robot.residuals.base.default_apply_jac_transpose
    :summary:
    ```
````

### API

`````{py:class} ResidualState
:canonical: better_robot.residuals.base.ResidualState

```{autodoc2-docstring} better_robot.residuals.base.ResidualState
```

````{py:attribute} model
:canonical: better_robot.residuals.base.ResidualState.model
:type: better_robot.data_model.model.Model
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.base.ResidualState.model
```

````

````{py:attribute} data
:canonical: better_robot.residuals.base.ResidualState.data
:type: better_robot.data_model.data.Data
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.base.ResidualState.data
```

````

````{py:attribute} variables
:canonical: better_robot.residuals.base.ResidualState.variables
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.base.ResidualState.variables
```

````

`````

`````{py:class} Residual
:canonical: better_robot.residuals.base.Residual

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.residuals.base.Residual
```

````{py:attribute} name
:canonical: better_robot.residuals.base.Residual.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.base.Residual.name
```

````

````{py:attribute} dim
:canonical: better_robot.residuals.base.Residual.dim
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.base.Residual.dim
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.base.Residual.jacobian

```{autodoc2-docstring} better_robot.residuals.base.Residual.jacobian
```

````

`````

````{py:function} default_apply_jac_transpose(residual: better_robot.residuals.base.Residual, state: better_robot.residuals.base.ResidualState, vec: torch.Tensor) -> torch.Tensor
:canonical: better_robot.residuals.base.default_apply_jac_transpose

```{autodoc2-docstring} better_robot.residuals.base.default_apply_jac_transpose
```
````
