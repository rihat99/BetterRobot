# {py:mod}`better_robot.collision.distance`

```{py:module} better_robot.collision.distance
```

```{autodoc2-docstring} better_robot.collision.distance
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`sphere_sphere <better_robot.collision.distance.sphere_sphere>`
  - ```{autodoc2-docstring} better_robot.collision.distance.sphere_sphere
    :summary:
    ```
* - {py:obj}`sphere_capsule <better_robot.collision.distance.sphere_capsule>`
  - ```{autodoc2-docstring} better_robot.collision.distance.sphere_capsule
    :summary:
    ```
* - {py:obj}`sphere_halfspace <better_robot.collision.distance.sphere_halfspace>`
  - ```{autodoc2-docstring} better_robot.collision.distance.sphere_halfspace
    :summary:
    ```
* - {py:obj}`sphere_box <better_robot.collision.distance.sphere_box>`
  - ```{autodoc2-docstring} better_robot.collision.distance.sphere_box
    :summary:
    ```
* - {py:obj}`sphere_sdf <better_robot.collision.distance.sphere_sdf>`
  - ```{autodoc2-docstring} better_robot.collision.distance.sphere_sdf
    :summary:
    ```
````

### API

````{py:function} sphere_sphere(center_a: torch.Tensor, radius_a: torch.Tensor, center_b: torch.Tensor, radius_b: torch.Tensor) -> torch.Tensor
:canonical: better_robot.collision.distance.sphere_sphere

```{autodoc2-docstring} better_robot.collision.distance.sphere_sphere
```
````

````{py:function} sphere_capsule(center: torch.Tensor, radius: torch.Tensor, start: torch.Tensor, end: torch.Tensor, capsule_radius: torch.Tensor) -> torch.Tensor
:canonical: better_robot.collision.distance.sphere_capsule

```{autodoc2-docstring} better_robot.collision.distance.sphere_capsule
```
````

````{py:function} sphere_halfspace(center: torch.Tensor, radius: torch.Tensor, normal: torch.Tensor, offset: torch.Tensor | float) -> torch.Tensor
:canonical: better_robot.collision.distance.sphere_halfspace

```{autodoc2-docstring} better_robot.collision.distance.sphere_halfspace
```
````

````{py:function} sphere_box(center: torch.Tensor, radius: torch.Tensor, box_center: torch.Tensor, box_rotation: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor
:canonical: better_robot.collision.distance.sphere_box

```{autodoc2-docstring} better_robot.collision.distance.sphere_box
```
````

````{py:function} sphere_sdf(center: torch.Tensor, radius: torch.Tensor, grid: torch.Tensor, origin: torch.Tensor, voxel_size: float) -> torch.Tensor
:canonical: better_robot.collision.distance.sphere_sdf

```{autodoc2-docstring} better_robot.collision.distance.sphere_sdf
```
````
