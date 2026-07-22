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
````

### API

`````{py:class} RestResidual(q: better_robot.residuals.utils.RobotVariableLike, q_rest: torch.Tensor | better_robot.residuals.utils.VariableLike, *, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'rest')
:canonical: better_robot.residuals.regularization.RestResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.regularization.RestResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.regularization.RestResidual.error

````

````{py:method} jacobian() -> tuple[torch.Tensor, ...]
:canonical: better_robot.residuals.regularization.RestResidual.jacobian

````

`````

`````{py:class} JointRotationPrior(q: better_robot.residuals.utils.RobotVariableLike, q_mean: torch.Tensor | better_robot.residuals.utils.VariableLike, per_joint_weight: torch.Tensor, *, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, kernel: object | None = None, name: str = 'joint_rotation_prior')
:canonical: better_robot.residuals.regularization.JointRotationPrior

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.regularization.JointRotationPrior
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.regularization.JointRotationPrior.error

````

`````

`````{py:class} ReferenceTrajectoryResidual(q: better_robot.residuals.utils.RobotVariableLike, q_ref: torch.Tensor | better_robot.residuals.utils.VariableLike, *, weight: numbers.Real | torch.Tensor = 1.0, row_weight: better_robot.residuals.base.Weight | numbers.Real | torch.Tensor = 1.0, weight_per_frame: torch.Tensor | None = None, kernel: object | None = None, name: str = 'reference_trajectory')
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual

Bases: {py:obj}`better_robot.residuals.base.Residual`

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual
```

````{py:method} error() -> torch.Tensor
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.error

````

````{py:method} temporal_structure(variable: better_robot.residuals.utils.RobotVariableLike | str) -> better_robot.residuals.structure.TemporalPattern | None
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.temporal_structure

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual.temporal_structure
```

````

````{py:method} temporal_jacobian_blocks(variable: better_robot.residuals.utils.RobotVariableLike | str) -> collections.abc.Mapping[int, torch.Tensor]
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.temporal_jacobian_blocks

```{autodoc2-docstring} better_robot.residuals.regularization.ReferenceTrajectoryResidual.temporal_jacobian_blocks
```

````

````{py:method} jacobian() -> tuple[torch.Tensor, ...]
:canonical: better_robot.residuals.regularization.ReferenceTrajectoryResidual.jacobian

````

`````
