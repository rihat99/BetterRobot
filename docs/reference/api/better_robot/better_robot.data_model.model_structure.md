# {py:mod}`better_robot.data_model.model_structure`

```{py:module} better_robot.data_model.model_structure
```

```{autodoc2-docstring} better_robot.data_model.model_structure
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModelStructure <better_robot.data_model.model_structure.ModelStructure>`
  - ```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JOINT_KIND_CODES <better_robot.data_model.model_structure.JOINT_KIND_CODES>`
  - ```{autodoc2-docstring} better_robot.data_model.model_structure.JOINT_KIND_CODES
    :summary:
    ```
````

### API

````{py:data} JOINT_KIND_CODES
:canonical: better_robot.data_model.model_structure.JOINT_KIND_CODES
:type: typing.Mapping[str, int]
:value: >
   'MappingProxyType(...)'

```{autodoc2-docstring} better_robot.data_model.model_structure.JOINT_KIND_CODES
```

````

`````{py:class} ModelStructure
:canonical: better_robot.data_model.model_structure.ModelStructure

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure
```

````{py:attribute} njoints
:canonical: better_robot.data_model.model_structure.ModelStructure.njoints
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.njoints
```

````

````{py:attribute} nbodies
:canonical: better_robot.data_model.model_structure.ModelStructure.nbodies
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.nbodies
```

````

````{py:attribute} nframes
:canonical: better_robot.data_model.model_structure.ModelStructure.nframes
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.nframes
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.model_structure.ModelStructure.nq
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.model_structure.ModelStructure.nv
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.nv
```

````

````{py:attribute} name
:canonical: better_robot.data_model.model_structure.ModelStructure.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.name
```

````

````{py:attribute} joint_names
:canonical: better_robot.data_model.model_structure.ModelStructure.joint_names
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.joint_names
```

````

````{py:attribute} body_names
:canonical: better_robot.data_model.model_structure.ModelStructure.body_names
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.body_names
```

````

````{py:attribute} frame_names
:canonical: better_robot.data_model.model_structure.ModelStructure.frame_names
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.frame_names
```

````

````{py:attribute} parents
:canonical: better_robot.data_model.model_structure.ModelStructure.parents
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.parents
```

````

````{py:attribute} children
:canonical: better_robot.data_model.model_structure.ModelStructure.children
:type: tuple[tuple[int, ...], ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.children
```

````

````{py:attribute} subtrees
:canonical: better_robot.data_model.model_structure.ModelStructure.subtrees
:type: tuple[tuple[int, ...], ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.subtrees
```

````

````{py:attribute} supports
:canonical: better_robot.data_model.model_structure.ModelStructure.supports
:type: tuple[tuple[int, ...], ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.supports
```

````

````{py:attribute} topo_order
:canonical: better_robot.data_model.model_structure.ModelStructure.topo_order
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.topo_order
```

````

````{py:attribute} nqs
:canonical: better_robot.data_model.model_structure.ModelStructure.nqs
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.nqs
```

````

````{py:attribute} nvs
:canonical: better_robot.data_model.model_structure.ModelStructure.nvs
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.nvs
```

````

````{py:attribute} idx_qs
:canonical: better_robot.data_model.model_structure.ModelStructure.idx_qs
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.idx_qs
```

````

````{py:attribute} idx_vs
:canonical: better_robot.data_model.model_structure.ModelStructure.idx_vs
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.idx_vs
```

````

````{py:attribute} joint_models
:canonical: better_robot.data_model.model_structure.ModelStructure.joint_models
:type: tuple[better_robot.data_model.joint_models.base.JointModel, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.joint_models
```

````

````{py:attribute} joint_kind_codes
:canonical: better_robot.data_model.model_structure.ModelStructure.joint_kind_codes
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.joint_kind_codes
```

````

````{py:attribute} mimic_source
:canonical: better_robot.data_model.model_structure.ModelStructure.mimic_source
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.mimic_source
```

````

````{py:attribute} joint_kind_tensor
:canonical: better_robot.data_model.model_structure.ModelStructure.joint_kind_tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.joint_kind_tensor
```

````

````{py:attribute} parents_tensor
:canonical: better_robot.data_model.model_structure.ModelStructure.parents_tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.parents_tensor
```

````

````{py:attribute} topo_order_tensor
:canonical: better_robot.data_model.model_structure.ModelStructure.topo_order_tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.topo_order_tensor
```

````

````{py:attribute} nqs_tensor
:canonical: better_robot.data_model.model_structure.ModelStructure.nqs_tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.nqs_tensor
```

````

````{py:attribute} nvs_tensor
:canonical: better_robot.data_model.model_structure.ModelStructure.nvs_tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.nvs_tensor
```

````

````{py:attribute} idx_qs_tensor
:canonical: better_robot.data_model.model_structure.ModelStructure.idx_qs_tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.idx_qs_tensor
```

````

````{py:attribute} idx_vs_tensor
:canonical: better_robot.data_model.model_structure.ModelStructure.idx_vs_tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.idx_vs_tensor
```

````

````{py:attribute} children_offsets
:canonical: better_robot.data_model.model_structure.ModelStructure.children_offsets
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.children_offsets
```

````

````{py:attribute} children_indices
:canonical: better_robot.data_model.model_structure.ModelStructure.children_indices
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.children_indices
```

````

````{py:attribute} subtree_offsets
:canonical: better_robot.data_model.model_structure.ModelStructure.subtree_offsets
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.subtree_offsets
```

````

````{py:attribute} subtree_indices
:canonical: better_robot.data_model.model_structure.ModelStructure.subtree_indices
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.subtree_indices
```

````

````{py:attribute} support_offsets
:canonical: better_robot.data_model.model_structure.ModelStructure.support_offsets
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.support_offsets
```

````

````{py:attribute} support_indices
:canonical: better_robot.data_model.model_structure.ModelStructure.support_indices
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.support_indices
```

````

````{py:attribute} frame_parent_joints
:canonical: better_robot.data_model.model_structure.ModelStructure.frame_parent_joints
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.frame_parent_joints
```

````

````{py:attribute} mimic_source_tensor
:canonical: better_robot.data_model.model_structure.ModelStructure.mimic_source_tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.mimic_source_tensor
```

````

````{py:attribute} joint_axes
:canonical: better_robot.data_model.model_structure.ModelStructure.joint_axes
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.joint_axes
```

````

````{py:attribute} joint_pitches
:canonical: better_robot.data_model.model_structure.ModelStructure.joint_pitches
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.joint_pitches
```

````

````{py:attribute} joint_motion_subspaces
:canonical: better_robot.data_model.model_structure.ModelStructure.joint_motion_subspaces
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.joint_motion_subspaces
```

````

````{py:method} from_model(model: better_robot.data_model.model.Model) -> better_robot.data_model.model_structure.ModelStructure
:canonical: better_robot.data_model.model_structure.ModelStructure.from_model
:classmethod:

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.from_model
```

````

````{py:method} to(device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> better_robot.data_model.model_structure.ModelStructure
:canonical: better_robot.data_model.model_structure.ModelStructure.to

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.to
```

````

````{py:method} validate_consistency() -> None
:canonical: better_robot.data_model.model_structure.ModelStructure.validate_consistency

```{autodoc2-docstring} better_robot.data_model.model_structure.ModelStructure.validate_consistency
```

````

`````
