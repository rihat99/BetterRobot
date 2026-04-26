# {py:mod}`better_robot.spatial.inertia`

```{py:module} better_robot.spatial.inertia
```

```{autodoc2-docstring} better_robot.spatial.inertia
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Inertia <better_robot.spatial.inertia.Inertia>`
  - ```{autodoc2-docstring} better_robot.spatial.inertia.Inertia
    :summary:
    ```
````

### API

`````{py:class} Inertia
:canonical: better_robot.spatial.inertia.Inertia

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia
```

````{py:attribute} data
:canonical: better_robot.spatial.inertia.Inertia.data
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.data
```

````

````{py:property} mass
:canonical: better_robot.spatial.inertia.Inertia.mass
:type: torch.Tensor

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.mass
```

````

````{py:property} com
:canonical: better_robot.spatial.inertia.Inertia.com
:type: torch.Tensor

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.com
```

````

````{py:property} inertia_matrix
:canonical: better_robot.spatial.inertia.Inertia.inertia_matrix
:type: torch.Tensor

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.inertia_matrix
```

````

````{py:method} zero(*, batch_shape: tuple[int, ...] = (), device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.spatial.inertia.Inertia.zero
:classmethod:

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.zero
```

````

````{py:method} from_sphere(mass: float, radius: float) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.spatial.inertia.Inertia.from_sphere
:classmethod:

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.from_sphere
```

````

````{py:method} from_box(mass: float, size: torch.Tensor) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.spatial.inertia.Inertia.from_box
:classmethod:

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.from_box
```

````

````{py:method} from_capsule(mass: float, radius: float, length: float) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.spatial.inertia.Inertia.from_capsule
:classmethod:

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.from_capsule
```

````

````{py:method} from_ellipsoid(mass: float, radii: torch.Tensor) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.spatial.inertia.Inertia.from_ellipsoid
:classmethod:

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.from_ellipsoid
```

````

````{py:method} from_mass_com_sym3(mass: torch.Tensor, com: torch.Tensor, sym3: torch.Tensor) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.spatial.inertia.Inertia.from_mass_com_sym3
:classmethod:

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.from_mass_com_sym3
```

````

````{py:method} from_mass_com_matrix(mass: torch.Tensor, com: torch.Tensor, I: torch.Tensor) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.spatial.inertia.Inertia.from_mass_com_matrix
:classmethod:

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.from_mass_com_matrix
```

````

````{py:method} se3_action(T) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.spatial.inertia.Inertia.se3_action

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.se3_action
```

````

````{py:method} apply(v) -> better_robot.spatial.force.Force
:canonical: better_robot.spatial.inertia.Inertia.apply

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.apply
```

````

````{py:method} add(other: better_robot.spatial.inertia.Inertia) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.spatial.inertia.Inertia.add

```{autodoc2-docstring} better_robot.spatial.inertia.Inertia.add
```

````

`````
