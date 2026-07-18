# {py:mod}`better_robot.tasks.contact_forces`

```{py:module} better_robot.tasks.contact_forces
```

```{autodoc2-docstring} better_robot.tasks.contact_forces
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ContactForceWeights <better_robot.tasks.contact_forces.ContactForceWeights>`
  - ```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceWeights
    :summary:
    ```
* - {py:obj}`ContactForceResult <better_robot.tasks.contact_forces.ContactForceResult>`
  - ```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`solve_contact_forces <better_robot.tasks.contact_forces.solve_contact_forces>`
  - ```{autodoc2-docstring} better_robot.tasks.contact_forces.solve_contact_forces
    :summary:
    ```
````

### API

`````{py:class} ContactForceWeights
:canonical: better_robot.tasks.contact_forces.ContactForceWeights

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceWeights
```

````{py:attribute} base_wrench
:canonical: better_robot.tasks.contact_forces.ContactForceWeights.base_wrench
:type: float
:value: >
   1.0

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceWeights.base_wrench
```

````

````{py:attribute} force_magnitude
:canonical: better_robot.tasks.contact_forces.ContactForceWeights.force_magnitude
:type: float
:value: >
   0.0001

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceWeights.force_magnitude
```

````

````{py:attribute} force_smooth
:canonical: better_robot.tasks.contact_forces.ContactForceWeights.force_smooth
:type: float
:value: >
   0.0

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceWeights.force_smooth
```

````

````{py:attribute} torque_smooth
:canonical: better_robot.tasks.contact_forces.ContactForceWeights.torque_smooth
:type: float
:value: >
   0.0

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceWeights.torque_smooth
```

````

`````

`````{py:class} ContactForceResult
:canonical: better_robot.tasks.contact_forces.ContactForceResult

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult
```

````{py:attribute} forces_world
:canonical: better_robot.tasks.contact_forces.ContactForceResult.forces_world
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult.forces_world
```

````

````{py:attribute} fext_local
:canonical: better_robot.tasks.contact_forces.ContactForceResult.fext_local
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult.fext_local
```

````

````{py:attribute} generalized_force
:canonical: better_robot.tasks.contact_forces.ContactForceResult.generalized_force
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult.generalized_force
```

````

````{py:attribute} residual
:canonical: better_robot.tasks.contact_forces.ContactForceResult.residual
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult.residual
```

````

````{py:attribute} cost
:canonical: better_robot.tasks.contact_forces.ContactForceResult.cost
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult.cost
```

````

````{py:attribute} iters
:canonical: better_robot.tasks.contact_forces.ContactForceResult.iters
:type: int | torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult.iters
```

````

````{py:attribute} converged
:canonical: better_robot.tasks.contact_forces.ContactForceResult.converged
:type: bool | torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult.converged
```

````

````{py:attribute} status
:canonical: better_robot.tasks.contact_forces.ContactForceResult.status
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult.status
```

````

````{py:attribute} model
:canonical: better_robot.tasks.contact_forces.ContactForceResult.model
:type: better_robot.data_model.model.Model
:value: >
   None

```{autodoc2-docstring} better_robot.tasks.contact_forces.ContactForceResult.model
```

````

`````

````{py:function} solve_contact_forces(model: better_robot.data_model.model.Model, q_traj: torch.Tensor, contact_joint_ids: torch.Tensor | collections.abc.Sequence[int], active_mask: torch.Tensor, *, dt: float, gravity: torch.Tensor | None = None, initial_forces: torch.Tensor | None = None, weights: better_robot.tasks.contact_forces.ContactForceWeights | None = None, max_iter: int = 50, damping_parameter: float = 0.001, tolerance: float = 1e-06) -> better_robot.tasks.contact_forces.ContactForceResult
:canonical: better_robot.tasks.contact_forces.solve_contact_forces

```{autodoc2-docstring} better_robot.tasks.contact_forces.solve_contact_forces
```
````
