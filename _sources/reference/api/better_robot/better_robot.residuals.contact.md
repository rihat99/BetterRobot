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

`````{py:class} ContactConsistencyResidual(model: better_robot.data_model.model.Model, frame_ids: tuple[int, ...], contact_weights: torch.Tensor, *, dt: float, weight: float = 1.0, angular: bool = False, name: str = 'contact_consistency')
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

````{py:attribute} reads
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.reads
:value: >
   ('q', 'data')

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual.reads
```

````

````{py:method} temporal_structure(variable_name: str) -> better_robot.residuals.structure.TemporalPattern | None
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.temporal_structure

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual.temporal_structure
```

````

````{py:method} temporal_jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any], variable_name: str) -> collections.abc.Mapping[int, torch.Tensor]
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.temporal_jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual.temporal_jacobian_blocks
```

````

````{py:method} jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any]) -> dict[str, torch.Tensor]
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual.jacobian_blocks
```

````

````{py:method} jacobian(value: better_robot.residuals.base.ResidualState | collections.abc.Mapping[str, typing.Any]) -> torch.Tensor | None
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
