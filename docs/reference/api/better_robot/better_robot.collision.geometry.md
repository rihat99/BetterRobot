# {py:mod}`better_robot.collision.geometry`

```{py:module} better_robot.collision.geometry
```

```{autodoc2-docstring} better_robot.collision.geometry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Sphere <better_robot.collision.geometry.Sphere>`
  - ```{autodoc2-docstring} better_robot.collision.geometry.Sphere
    :summary:
    ```
* - {py:obj}`Capsule <better_robot.collision.geometry.Capsule>`
  - ```{autodoc2-docstring} better_robot.collision.geometry.Capsule
    :summary:
    ```
* - {py:obj}`Box <better_robot.collision.geometry.Box>`
  - ```{autodoc2-docstring} better_robot.collision.geometry.Box
    :summary:
    ```
* - {py:obj}`HalfSpace <better_robot.collision.geometry.HalfSpace>`
  - ```{autodoc2-docstring} better_robot.collision.geometry.HalfSpace
    :summary:
    ```
* - {py:obj}`Plane <better_robot.collision.geometry.Plane>`
  - ```{autodoc2-docstring} better_robot.collision.geometry.Plane
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`colldist_from_sdf <better_robot.collision.geometry.colldist_from_sdf>`
  - ```{autodoc2-docstring} better_robot.collision.geometry.colldist_from_sdf
    :summary:
    ```
````

### API

`````{py:class} Sphere
:canonical: better_robot.collision.geometry.Sphere

```{autodoc2-docstring} better_robot.collision.geometry.Sphere
```

````{py:attribute} center
:canonical: better_robot.collision.geometry.Sphere.center
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.geometry.Sphere.center
```

````

````{py:attribute} radius
:canonical: better_robot.collision.geometry.Sphere.radius
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.geometry.Sphere.radius
```

````

`````

`````{py:class} Capsule
:canonical: better_robot.collision.geometry.Capsule

```{autodoc2-docstring} better_robot.collision.geometry.Capsule
```

````{py:attribute} a
:canonical: better_robot.collision.geometry.Capsule.a
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.geometry.Capsule.a
```

````

````{py:attribute} b
:canonical: better_robot.collision.geometry.Capsule.b
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.geometry.Capsule.b
```

````

````{py:attribute} radius
:canonical: better_robot.collision.geometry.Capsule.radius
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.geometry.Capsule.radius
```

````

`````

`````{py:class} Box
:canonical: better_robot.collision.geometry.Box

```{autodoc2-docstring} better_robot.collision.geometry.Box
```

````{py:attribute} center
:canonical: better_robot.collision.geometry.Box.center
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.geometry.Box.center
```

````

````{py:attribute} half_extents
:canonical: better_robot.collision.geometry.Box.half_extents
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.geometry.Box.half_extents
```

````

````{py:attribute} rotation
:canonical: better_robot.collision.geometry.Box.rotation
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.geometry.Box.rotation
```

````

`````

`````{py:class} HalfSpace
:canonical: better_robot.collision.geometry.HalfSpace

```{autodoc2-docstring} better_robot.collision.geometry.HalfSpace
```

````{py:attribute} normal
:canonical: better_robot.collision.geometry.HalfSpace.normal
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.geometry.HalfSpace.normal
```

````

````{py:attribute} offset
:canonical: better_robot.collision.geometry.HalfSpace.offset
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.collision.geometry.HalfSpace.offset
```

````

`````

````{py:class} Plane
:canonical: better_robot.collision.geometry.Plane

Bases: {py:obj}`better_robot.collision.geometry.HalfSpace`

```{autodoc2-docstring} better_robot.collision.geometry.Plane
```

````

````{py:function} colldist_from_sdf(d: torch.Tensor, margin: float) -> torch.Tensor
:canonical: better_robot.collision.geometry.colldist_from_sdf

```{autodoc2-docstring} better_robot.collision.geometry.colldist_from_sdf
```
````
