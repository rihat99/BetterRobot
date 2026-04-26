# {py:mod}`better_robot.residuals.regularization`

```{py:module} better_robot.residuals.regularization
```

```{autodoc2-docstring} better_robot.residuals.regularization
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RestResidual <better_robot.residuals.regularization.RestResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.regularization.RestResidual
    :summary:
    ```
* - {py:obj}`ReferenceTrajectoryResidual <better_robot.residuals.regularization.ReferenceTrajectoryResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual
    :summary:
    ```
* - {py:obj}`NullspaceResidual <better_robot.residuals.regularization.NullspaceResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.regularization.NullspaceResidual
    :summary:
    ```
````

### API

`````{py:class} RestResidual(model: better_robot.data_model.model.Model, q_rest: torch.Tensor, *, weight: float = 1.0)
:canonical: better_robot.residuals.regularization.RestResidual

```{autodoc2-docstring} better_robot.residuals.regularization.RestResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.regularization.RestResidual.name
:type: str
:value: >
   'rest'

```{autodoc2-docstring} better_robot.residuals.regularization.RestResidual.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.regularization.RestResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.regularization.RestResidual.jacobian
```

````

`````

`````{py:class} ReferenceTrajectoryResidual(model: better_robot.data_model.model.Model, q_ref: torch.Tensor, *, weight: float = 1.0, weight_per_frame: torch.Tensor | None = None)
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.name
:type: str
:value: >
   'reference_trajectory'

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual.jacobian
```

````

````{py:method} apply_jac_transpose(state: better_robot.residuals.base.ResidualState, r: torch.Tensor) -> torch.Tensor
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.apply_jac_transpose

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual.apply_jac_transpose
```

````

`````

`````{py:class} NullspaceResidual(q_rest: torch.Tensor, *, weight: float = 1.0)
:canonical: better_robot.residuals.regularization.NullspaceResidual

```{autodoc2-docstring} better_robot.residuals.regularization.NullspaceResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.regularization.NullspaceResidual.name
:type: str
:value: >
   'nullspace'

```{autodoc2-docstring} better_robot.residuals.regularization.NullspaceResidual.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.regularization.NullspaceResidual.jacobian
:abstractmethod:

```{autodoc2-docstring} better_robot.residuals.regularization.NullspaceResidual.jacobian
```

````

`````
