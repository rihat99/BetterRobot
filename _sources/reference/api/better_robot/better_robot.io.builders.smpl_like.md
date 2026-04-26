# {py:mod}`better_robot.io.builders.smpl_like`

```{py:module} better_robot.io.builders.smpl_like
```

```{autodoc2-docstring} better_robot.io.builders.smpl_like
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`make_smpl_like_body <better_robot.io.builders.smpl_like.make_smpl_like_body>`
  - ```{autodoc2-docstring} better_robot.io.builders.smpl_like.make_smpl_like_body
    :summary:
    ```
* - {py:obj}`make_smpl_like_model <better_robot.io.builders.smpl_like.make_smpl_like_model>`
  - ```{autodoc2-docstring} better_robot.io.builders.smpl_like.make_smpl_like_model
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JOINT_NAMES <better_robot.io.builders.smpl_like.JOINT_NAMES>`
  - ```{autodoc2-docstring} better_robot.io.builders.smpl_like.JOINT_NAMES
    :summary:
    ```
* - {py:obj}`PARENTS <better_robot.io.builders.smpl_like.PARENTS>`
  - ```{autodoc2-docstring} better_robot.io.builders.smpl_like.PARENTS
    :summary:
    ```
````

### API

````{py:data} JOINT_NAMES
:canonical: better_robot.io.builders.smpl_like.JOINT_NAMES
:type: tuple[str, ...]
:value: >
   ('pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'ri...

```{autodoc2-docstring} better_robot.io.builders.smpl_like.JOINT_NAMES
```

````

````{py:data} PARENTS
:canonical: better_robot.io.builders.smpl_like.PARENTS
:type: tuple[int, ...]
:value: >
   ()

```{autodoc2-docstring} better_robot.io.builders.smpl_like.PARENTS
```

````

````{py:function} make_smpl_like_body(height: float = 1.75, mass: float = 70.0, *, name: str = 'smpl_body', shape_params: torch.Tensor | None = None, joint_offsets: torch.Tensor | None = None, mass_per_body: float | collections.abc.Sequence[float] | None = None, com_per_body: torch.Tensor | collections.abc.Sequence[torch.Tensor] | None = None, inertia_per_body: torch.Tensor | collections.abc.Sequence[torch.Tensor] | None = None) -> better_robot.io.ir.IRModel
:canonical: better_robot.io.builders.smpl_like.make_smpl_like_body

```{autodoc2-docstring} better_robot.io.builders.smpl_like.make_smpl_like_body
```
````

````{py:function} make_smpl_like_model(height: float = 1.75, mass: float = 70.0, *, name: str = 'smpl_body', shape_params: torch.Tensor | None = None, joint_offsets: torch.Tensor | None = None, mass_per_body: float | collections.abc.Sequence[float] | None = None, com_per_body: torch.Tensor | collections.abc.Sequence[torch.Tensor] | None = None, inertia_per_body: torch.Tensor | collections.abc.Sequence[torch.Tensor] | None = None, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> better_robot.data_model.model.Model
:canonical: better_robot.io.builders.smpl_like.make_smpl_like_model

```{autodoc2-docstring} better_robot.io.builders.smpl_like.make_smpl_like_model
```
````
