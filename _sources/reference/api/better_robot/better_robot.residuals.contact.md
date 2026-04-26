# {py:mod}`better_robot.residuals.contact`

```{py:module} better_robot.residuals.contact
```

```{autodoc2-docstring} better_robot.residuals.contact
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ContactConsistencyResidual <better_robot.residuals.contact.ContactConsistencyResidual>`
  - ```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual
    :summary:
    ```
````

### API

`````{py:class} ContactConsistencyResidual(model: better_robot.data_model.model.Model, frame_ids: tuple[int, ...], contact_weights: torch.Tensor, *, dt: float, weight: float = 1.0, angular: bool = False)
:canonical: better_robot.residuals.contact.ContactConsistencyResidual

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual
```

````{py:attribute} name
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.name
:type: str
:value: >
   'contact_consistency'

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual.name
```

````

````{py:method} jacobian(state: better_robot.residuals.base.ResidualState) -> torch.Tensor | None
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.jacobian

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual.jacobian
```

````

````{py:method} apply_jac_transpose(state: better_robot.residuals.base.ResidualState, r: torch.Tensor) -> torch.Tensor
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.apply_jac_transpose

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual.apply_jac_transpose
```

````

`````
