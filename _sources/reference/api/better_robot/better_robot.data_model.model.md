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

````{py:attribute} structure
:canonical: better_robot.data_model.model.Model.structure
:type: better_robot.data_model.model_structure.ModelStructure
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.structure
```

````

````{py:attribute} values
:canonical: better_robot.data_model.model.Model.values
:type: better_robot.data_model.model_values.ModelValues
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model.Model.values
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

````{py:attribute} meta
:canonical: better_robot.data_model.model.Model.meta
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.data_model.model.Model.meta
```

````

````{py:property} njoints
:canonical: better_robot.data_model.model.Model.njoints
:type: int

```{autodoc2-docstring} better_robot.data_model.model.Model.njoints
```

````

````{py:property} nbodies
:canonical: better_robot.data_model.model.Model.nbodies
:type: int

```{autodoc2-docstring} better_robot.data_model.model.Model.nbodies
```

````

````{py:property} nframes
:canonical: better_robot.data_model.model.Model.nframes
:type: int

```{autodoc2-docstring} better_robot.data_model.model.Model.nframes
```

````

````{py:property} nq
:canonical: better_robot.data_model.model.Model.nq
:type: int

```{autodoc2-docstring} better_robot.data_model.model.Model.nq
```

````

````{py:property} nv
:canonical: better_robot.data_model.model.Model.nv
:type: int

```{autodoc2-docstring} better_robot.data_model.model.Model.nv
```

````

````{py:property} nq_full
:canonical: better_robot.data_model.model.Model.nq_full
:type: int

```{autodoc2-docstring} better_robot.data_model.model.Model.nq_full
```

````

````{py:property} nv_full
:canonical: better_robot.data_model.model.Model.nv_full
:type: int

```{autodoc2-docstring} better_robot.data_model.model.Model.nv_full
```

````

````{py:property} name
:canonical: better_robot.data_model.model.Model.name
:type: str

```{autodoc2-docstring} better_robot.data_model.model.Model.name
```

````

````{py:property} joint_names
:canonical: better_robot.data_model.model.Model.joint_names
:type: tuple[str, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.joint_names
```

````

````{py:property} body_names
:canonical: better_robot.data_model.model.Model.body_names
:type: tuple[str, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.body_names
```

````

````{py:property} frame_names
:canonical: better_robot.data_model.model.Model.frame_names
:type: tuple[str, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.frame_names
```

````

````{py:property} joint_name_to_id
:canonical: better_robot.data_model.model.Model.joint_name_to_id
:type: dict[str, int]

```{autodoc2-docstring} better_robot.data_model.model.Model.joint_name_to_id
```

````

````{py:property} body_name_to_id
:canonical: better_robot.data_model.model.Model.body_name_to_id
:type: dict[str, int]

```{autodoc2-docstring} better_robot.data_model.model.Model.body_name_to_id
```

````

````{py:property} frame_name_to_id
:canonical: better_robot.data_model.model.Model.frame_name_to_id
:type: dict[str, int]

```{autodoc2-docstring} better_robot.data_model.model.Model.frame_name_to_id
```

````

````{py:property} parents
:canonical: better_robot.data_model.model.Model.parents
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.parents
```

````

````{py:property} children
:canonical: better_robot.data_model.model.Model.children
:type: tuple[tuple[int, ...], ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.children
```

````

````{py:property} subtrees
:canonical: better_robot.data_model.model.Model.subtrees
:type: tuple[tuple[int, ...], ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.subtrees
```

````

````{py:property} supports
:canonical: better_robot.data_model.model.Model.supports
:type: tuple[tuple[int, ...], ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.supports
```

````

````{py:property} topo_order
:canonical: better_robot.data_model.model.Model.topo_order
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.topo_order
```

````

````{py:property} joint_models
:canonical: better_robot.data_model.model.Model.joint_models
:type: tuple[better_robot.data_model.joint_models.base.JointModel, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.joint_models
```

````

````{py:property} nqs
:canonical: better_robot.data_model.model.Model.nqs
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.nqs
```

````

````{py:property} nvs
:canonical: better_robot.data_model.model.Model.nvs
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.nvs
```

````

````{py:property} idx_qs
:canonical: better_robot.data_model.model.Model.idx_qs
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.idx_qs
```

````

````{py:property} idx_vs
:canonical: better_robot.data_model.model.Model.idx_vs
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.idx_vs
```

````

````{py:property} nqs_full
:canonical: better_robot.data_model.model.Model.nqs_full
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.nqs_full
```

````

````{py:property} nvs_full
:canonical: better_robot.data_model.model.Model.nvs_full
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.nvs_full
```

````

````{py:property} idx_qs_full
:canonical: better_robot.data_model.model.Model.idx_qs_full
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.idx_qs_full
```

````

````{py:property} idx_vs_full
:canonical: better_robot.data_model.model.Model.idx_vs_full
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.idx_vs_full
```

````

````{py:property} joint_placements
:canonical: better_robot.data_model.model.Model.joint_placements
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.joint_placements
```

````

````{py:property} body_inertias
:canonical: better_robot.data_model.model.Model.body_inertias
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.body_inertias
```

````

````{py:property} lower_pos_limit
:canonical: better_robot.data_model.model.Model.lower_pos_limit
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.lower_pos_limit
```

````

````{py:property} upper_pos_limit
:canonical: better_robot.data_model.model.Model.upper_pos_limit
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.upper_pos_limit
```

````

````{py:property} velocity_limit
:canonical: better_robot.data_model.model.Model.velocity_limit
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.velocity_limit
```

````

````{py:property} effort_limit
:canonical: better_robot.data_model.model.Model.effort_limit
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.effort_limit
```

````

````{py:property} rotor_inertia
:canonical: better_robot.data_model.model.Model.rotor_inertia
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.rotor_inertia
```

````

````{py:property} armature
:canonical: better_robot.data_model.model.Model.armature
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.armature
```

````

````{py:property} friction
:canonical: better_robot.data_model.model.Model.friction
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.friction
```

````

````{py:property} damping
:canonical: better_robot.data_model.model.Model.damping
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.damping
```

````

````{py:property} gravity
:canonical: better_robot.data_model.model.Model.gravity
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.gravity
```

````

````{py:property} mimic_multiplier
:canonical: better_robot.data_model.model.Model.mimic_multiplier
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.mimic_multiplier
```

````

````{py:property} mimic_offset
:canonical: better_robot.data_model.model.Model.mimic_offset
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.mimic_offset
```

````

````{py:property} q_neutral
:canonical: better_robot.data_model.model.Model.q_neutral
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.q_neutral
```

````

````{py:property} mimic_source
:canonical: better_robot.data_model.model.Model.mimic_source
:type: tuple[int, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.mimic_source
```

````

````{py:property} q_expansion
:canonical: better_robot.data_model.model.Model.q_expansion
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.q_expansion
```

````

````{py:property} q_offset
:canonical: better_robot.data_model.model.Model.q_offset
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.q_offset
```

````

````{py:property} v_expansion
:canonical: better_robot.data_model.model.Model.v_expansion
:type: torch.Tensor

```{autodoc2-docstring} better_robot.data_model.model.Model.v_expansion
```

````

````{py:property} has_mimic
:canonical: better_robot.data_model.model.Model.has_mimic
:type: bool

```{autodoc2-docstring} better_robot.data_model.model.Model.has_mimic
```

````

````{py:property} frames
:canonical: better_robot.data_model.model.Model.frames
:type: tuple[better_robot.data_model.frame.Frame, ...]

```{autodoc2-docstring} better_robot.data_model.model.Model.frames
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

````{py:method} q_permutation(other_joint_order: collections.abc.Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.data_model.model.Model.q_permutation

```{autodoc2-docstring} better_robot.data_model.model.Model.q_permutation
```

````

````{py:method} body_inertia(body_id: int) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.data_model.model.Model.body_inertia

```{autodoc2-docstring} better_robot.data_model.model.Model.body_inertia
```

````

````{py:method} spatial_inertias() -> torch.Tensor
:canonical: better_robot.data_model.model.Model.spatial_inertias

```{autodoc2-docstring} better_robot.data_model.model.Model.spatial_inertias
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

````{py:method} create_data(*, batch_shape: tuple[int, ...] = (), device: torch.device | None = None, dtype: torch.dtype | None = None) -> better_robot.data_model.data.Data
:canonical: better_robot.data_model.model.Model.create_data

```{autodoc2-docstring} better_robot.data_model.model.Model.create_data
```

````

````{py:method} with_values(*, joint_placements: torch.Tensor | None = None, body_inertias: torch.Tensor | None = None, frame_placements: torch.Tensor | None = None) -> better_robot.data_model.model.Model
:canonical: better_robot.data_model.model.Model.with_values

```{autodoc2-docstring} better_robot.data_model.model.Model.with_values
```

````

````{py:method} to(device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> better_robot.data_model.model.Model
:canonical: better_robot.data_model.model.Model.to

```{autodoc2-docstring} better_robot.data_model.model.Model.to
```

````

`````
