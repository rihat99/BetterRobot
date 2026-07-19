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
* - {py:obj}`JointRotationPrior <better_robot.residuals.regularization.JointRotationPrior>`
  - ```{autodoc2-docstring} better_robot.residuals.regularization.JointRotationPrior
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

`````{py:class} RestResidual(model: better_robot.data_model.model.Model, q_rest: torch.Tensor, *, weight: float = 1.0, name: str = 'rest', target_name: str | None = None)
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

````{py:attribute} reads
:canonical: better_robot.residuals.regularization.RestResidual.reads
:value: >
   ('q',)

```{autodoc2-docstring} better_robot.residuals.regularization.RestResidual.reads
```

````

````{py:method} jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any]) -> dict[str, torch.Tensor]
:canonical: better_robot.residuals.regularization.RestResidual.jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.regularization.RestResidual.jacobian_blocks
```

````

`````

`````{py:class} JointRotationPrior(model: better_robot.data_model.model.Model, q_mean: torch.Tensor, per_joint_weight: torch.Tensor, *, name: str = 'joint_rotation_prior')
:canonical: better_robot.residuals.regularization.JointRotationPrior

```{autodoc2-docstring} better_robot.residuals.regularization.JointRotationPrior
```

````{py:attribute} reads
:canonical: better_robot.residuals.regularization.JointRotationPrior.reads
:value: >
   ('q',)

```{autodoc2-docstring} better_robot.residuals.regularization.JointRotationPrior.reads
```

````

`````

`````{py:class} ReferenceTrajectoryResidual(model: better_robot.data_model.model.Model, q_ref: torch.Tensor, *, weight: float = 1.0, weight_per_frame: torch.Tensor | None = None, name: str = 'reference_trajectory')
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

````{py:attribute} reads
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.reads
:value: >
   ('q',)

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual.reads
```

````

````{py:method} temporal_structure(variable_name: str) -> better_robot.residuals.structure.TemporalPattern | None
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.temporal_structure

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual.temporal_structure
```

````

````{py:method} temporal_jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any], variable_name: str) -> collections.abc.Mapping[int, torch.Tensor]
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.temporal_jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual.temporal_jacobian_blocks
```

````

````{py:method} jacobian_blocks(ctx: collections.abc.Mapping[str, typing.Any]) -> dict[str, torch.Tensor]
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual.jacobian_blocks
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

````{py:attribute} reads
:canonical: better_robot.residuals.regularization.NullspaceResidual.reads
:value: >
   ('q',)

```{autodoc2-docstring} better_robot.residuals.regularization.NullspaceResidual.reads
```

````

`````
