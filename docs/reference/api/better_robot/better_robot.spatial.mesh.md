# {py:mod}`better_robot.spatial.mesh`

```{py:module} better_robot.spatial.mesh
```

```{autodoc2-docstring} better_robot.spatial.mesh
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`validate_closed_manifold <better_robot.spatial.mesh.validate_closed_manifold>`
  - ```{autodoc2-docstring} better_robot.spatial.mesh.validate_closed_manifold
    :summary:
    ```
* - {py:obj}`orient_faces_consistently <better_robot.spatial.mesh.orient_faces_consistently>`
  - ```{autodoc2-docstring} better_robot.spatial.mesh.orient_faces_consistently
    :summary:
    ```
* - {py:obj}`orient_faces_by_component <better_robot.spatial.mesh.orient_faces_by_component>`
  - ```{autodoc2-docstring} better_robot.spatial.mesh.orient_faces_by_component
    :summary:
    ```
````

### API

````{py:function} validate_closed_manifold(faces: torch.Tensor) -> None
:canonical: better_robot.spatial.mesh.validate_closed_manifold

```{autodoc2-docstring} better_robot.spatial.mesh.validate_closed_manifold
```
````

````{py:function} orient_faces_consistently(faces: torch.Tensor) -> torch.Tensor
:canonical: better_robot.spatial.mesh.orient_faces_consistently

```{autodoc2-docstring} better_robot.spatial.mesh.orient_faces_consistently
```
````

````{py:function} orient_faces_by_component(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor
:canonical: better_robot.spatial.mesh.orient_faces_by_component

```{autodoc2-docstring} better_robot.spatial.mesh.orient_faces_by_component
```
````
