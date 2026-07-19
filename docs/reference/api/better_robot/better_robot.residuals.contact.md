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

`````{py:class} ContactConsistencyResidual(q_or_state: better_robot.residuals._variables.RobotVariableLike | better_robot.residuals.nodes.RobotState, frame_ids: tuple[int, ...], contact_weights: better_robot.residuals._variables.VariableLike | torch.Tensor, *, dt: float, weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, angular: bool = False, name: str = 'contact_consistency')
:canonical: better_robot.residuals.contact.ContactConsistencyResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.error

````

````{py:method} temporal_structure(variable: better_robot.residuals._variables.RobotVariableLike | str) -> better_robot.residuals.structure.TemporalPattern | None
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.temporal_structure

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual.temporal_structure
```

````

````{py:method} temporal_jacobian_blocks(variable: better_robot.residuals._variables.RobotVariableLike | str) -> collections.abc.Mapping[int, torch.Tensor]
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.temporal_jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.contact.ContactConsistencyResidual.temporal_jacobian_blocks
```

````

````{py:method} jacobian() -> tuple[torch.Tensor, ...] | None
:canonical: better_robot.residuals.contact.ContactConsistencyResidual.jacobian

````

`````
