# {py:mod}`better_robot.data_model.model`

```{py:module} better_robot.data_model.model
```

```{autodoc2-docstring} better_robot.data_model.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Model <better_robot.data_model.model.Model>`
  - ```{autodoc2-docstring} better_robot.data_model.model.Model
    :summary:
    ```
````

### API

`````{py:class} Model
:canonical: better_robot.data_model.model.Model

```{autodoc2-docstring} better_robot.data_model.model.Model
```

````{py:attribute} njoints
:canonical: better_robot.data_model.model.Model.njoints
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.njoints
```

````

````{py:attribute} nbodies
:canonical: better_robot.data_model.model.Model.nbodies
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.nbodies
```

````

````{py:attribute} nframes
:canonical: better_robot.data_model.model.Model.nframes
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.nframes
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.model.Model.nq
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.model.Model.nv
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.nv
```

````

````{py:attribute} name
:canonical: better_robot.data_model.model.Model.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.name
```

````

````{py:attribute} joint_names
:canonical: better_robot.data_model.model.Model.joint_names
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.joint_names
```

````

````{py:attribute} body_names
:canonical: better_robot.data_model.model.Model.body_names
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.body_names
```

````

````{py:attribute} frame_names
:canonical: better_robot.data_model.model.Model.frame_names
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.frame_names
```

````

````{py:attribute} joint_name_to_id
:canonical: better_robot.data_model.model.Model.joint_name_to_id
:type: dict[str, int]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.joint_name_to_id
```

````

````{py:attribute} body_name_to_id
:canonical: better_robot.data_model.model.Model.body_name_to_id
:type: dict[str, int]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.body_name_to_id
```

````

````{py:attribute} frame_name_to_id
:canonical: better_robot.data_model.model.Model.frame_name_to_id
:type: dict[str, int]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.frame_name_to_id
```

````

````{py:attribute} parents
:canonical: better_robot.data_model.model.Model.parents
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.parents
```

````

````{py:attribute} children
:canonical: better_robot.data_model.model.Model.children
:type: tuple[tuple[int, ...], ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.children
```

````

````{py:attribute} subtrees
:canonical: better_robot.data_model.model.Model.subtrees
:type: tuple[tuple[int, ...], ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.subtrees
```

````

````{py:attribute} supports
:canonical: better_robot.data_model.model.Model.supports
:type: tuple[tuple[int, ...], ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.supports
```

````

````{py:attribute} topo_order
:canonical: better_robot.data_model.model.Model.topo_order
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.topo_order
```

````

````{py:attribute} joint_models
:canonical: better_robot.data_model.model.Model.joint_models
:type: tuple[better_robot.data_model.joint_models.base.JointModel, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.joint_models
```

````

````{py:attribute} nqs
:canonical: better_robot.data_model.model.Model.nqs
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.nqs
```

````

````{py:attribute} nvs
:canonical: better_robot.data_model.model.Model.nvs
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.nvs
```

````

````{py:attribute} idx_qs
:canonical: better_robot.data_model.model.Model.idx_qs
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.idx_qs
```

````

````{py:attribute} idx_vs
:canonical: better_robot.data_model.model.Model.idx_vs
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.idx_vs
```

````

````{py:attribute} joint_placements
:canonical: better_robot.data_model.model.Model.joint_placements
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.joint_placements
```

````

````{py:attribute} body_inertias
:canonical: better_robot.data_model.model.Model.body_inertias
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.body_inertias
```

````

````{py:attribute} lower_pos_limit
:canonical: better_robot.data_model.model.Model.lower_pos_limit
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.lower_pos_limit
```

````

````{py:attribute} upper_pos_limit
:canonical: better_robot.data_model.model.Model.upper_pos_limit
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.upper_pos_limit
```

````

````{py:attribute} velocity_limit
:canonical: better_robot.data_model.model.Model.velocity_limit
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.velocity_limit
```

````

````{py:attribute} effort_limit
:canonical: better_robot.data_model.model.Model.effort_limit
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.effort_limit
```

````

````{py:attribute} rotor_inertia
:canonical: better_robot.data_model.model.Model.rotor_inertia
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.rotor_inertia
```

````

````{py:attribute} armature
:canonical: better_robot.data_model.model.Model.armature
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.armature
```

````

````{py:attribute} friction
:canonical: better_robot.data_model.model.Model.friction
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.friction
```

````

````{py:attribute} damping
:canonical: better_robot.data_model.model.Model.damping
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.damping
```

````

````{py:attribute} gravity
:canonical: better_robot.data_model.model.Model.gravity
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.gravity
```

````

````{py:attribute} mimic_multiplier
:canonical: better_robot.data_model.model.Model.mimic_multiplier
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.mimic_multiplier
```

````

````{py:attribute} mimic_offset
:canonical: better_robot.data_model.model.Model.mimic_offset
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.mimic_offset
```

````

````{py:attribute} mimic_source
:canonical: better_robot.data_model.model.Model.mimic_source
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.mimic_source
```

````

````{py:attribute} frames
:canonical: better_robot.data_model.model.Model.frames
:type: tuple[better_robot.data_model.frame.Frame, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.frames
```

````

````{py:attribute} reference_configurations
:canonical: better_robot.data_model.model.Model.reference_configurations
:type: dict[str, torch.Tensor]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.data_model.model.Model.reference_configurations
```

````

````{py:attribute} q_neutral
:canonical: better_robot.data_model.model.Model.q_neutral
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.data_model.model.Model.q_neutral
```

````

````{py:attribute} meta
:canonical: better_robot.data_model.model.Model.meta
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.data_model.model.Model.meta
```

````

````{py:method} to(device=None, dtype=None) -> better_robot.data_model.model.Model
:canonical: better_robot.data_model.model.Model.to

```{autodoc2-docstring} better_robot.data_model.model.Model.to
```

````

````{py:method} create_data(*, batch_shape: tuple[int, ...] = (), device: torch.device | None = None, dtype: torch.dtype | None = None)
:canonical: better_robot.data_model.model.Model.create_data

```{autodoc2-docstring} better_robot.data_model.model.Model.create_data
```

````

````{py:method} joint_id(name: str) -> int
:canonical: better_robot.data_model.model.Model.joint_id

```{autodoc2-docstring} better_robot.data_model.model.Model.joint_id
```

````

````{py:method} frame_id(name: str) -> int
:canonical: better_robot.data_model.model.Model.frame_id

```{autodoc2-docstring} better_robot.data_model.model.Model.frame_id
```

````

````{py:method} body_id(name: str) -> int
:canonical: better_robot.data_model.model.Model.body_id

```{autodoc2-docstring} better_robot.data_model.model.Model.body_id
```

````

````{py:method} body_inertia(body_id: int)
:canonical: better_robot.data_model.model.Model.body_inertia

```{autodoc2-docstring} better_robot.data_model.model.Model.body_inertia
```

````

````{py:method} get_subtree(joint_id: int) -> tuple[int, ...]
:canonical: better_robot.data_model.model.Model.get_subtree

```{autodoc2-docstring} better_robot.data_model.model.Model.get_subtree
```

````

````{py:method} get_support(joint_id: int) -> tuple[int, ...]
:canonical: better_robot.data_model.model.Model.get_support

```{autodoc2-docstring} better_robot.data_model.model.Model.get_support
```

````

````{py:method} integrate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.model.Model.integrate

```{autodoc2-docstring} better_robot.data_model.model.Model.integrate
```

````

````{py:method} difference(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.model.Model.difference

```{autodoc2-docstring} better_robot.data_model.model.Model.difference
```

````

````{py:method} random_configuration(generator: torch.Generator | None = None) -> torch.Tensor
:canonical: better_robot.data_model.model.Model.random_configuration

```{autodoc2-docstring} better_robot.data_model.model.Model.random_configuration
```

````

`````
