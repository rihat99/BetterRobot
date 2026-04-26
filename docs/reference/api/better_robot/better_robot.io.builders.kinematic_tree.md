# {py:mod}`better_robot.io.builders.kinematic_tree`

```{py:module} better_robot.io.builders.kinematic_tree
```

```{autodoc2-docstring} better_robot.io.builders.kinematic_tree
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_kinematic_tree_body <better_robot.io.builders.kinematic_tree.build_kinematic_tree_body>`
  - ```{autodoc2-docstring} better_robot.io.builders.kinematic_tree.build_kinematic_tree_body
    :summary:
    ```
* - {py:obj}`build_kinematic_tree_model <better_robot.io.builders.kinematic_tree.build_kinematic_tree_model>`
  - ```{autodoc2-docstring} better_robot.io.builders.kinematic_tree.build_kinematic_tree_model
    :summary:
    ```
````

### API

````{py:function} build_kinematic_tree_body(*, name: str, joint_names: collections.abc.Sequence[str], parents: collections.abc.Sequence[int], translations: torch.Tensor, root_kind: str = 'free_flyer', child_kind: str = 'spherical', mass_per_body: float | collections.abc.Sequence[float] = 0.0, com_per_body: torch.Tensor | collections.abc.Sequence[torch.Tensor] | None = None, inertia_per_body: torch.Tensor | collections.abc.Sequence[torch.Tensor] | None = None) -> better_robot.io.ir.IRModel
:canonical: better_robot.io.builders.kinematic_tree.build_kinematic_tree_body

```{autodoc2-docstring} better_robot.io.builders.kinematic_tree.build_kinematic_tree_body
```
````

````{py:function} build_kinematic_tree_model(*, name: str, joint_names: collections.abc.Sequence[str], parents: collections.abc.Sequence[int], translations: torch.Tensor, root_kind: str = 'free_flyer', child_kind: str = 'spherical', mass_per_body: float | collections.abc.Sequence[float] = 0.0, com_per_body: torch.Tensor | collections.abc.Sequence[torch.Tensor] | None = None, inertia_per_body: torch.Tensor | collections.abc.Sequence[torch.Tensor] | None = None, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> better_robot.data_model.model.Model
:canonical: better_robot.io.builders.kinematic_tree.build_kinematic_tree_model

```{autodoc2-docstring} better_robot.io.builders.kinematic_tree.build_kinematic_tree_model
```
````
