# {py:mod}`better_robot.io.parsers.programmatic`

```{py:module} better_robot.io.parsers.programmatic
```

```{autodoc2-docstring} better_robot.io.parsers.programmatic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModelBuilder <better_robot.io.parsers.programmatic.ModelBuilder>`
  - ```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder
    :summary:
    ```
````

### API

`````{py:class} ModelBuilder(name: str)
:canonical: better_robot.io.parsers.programmatic.ModelBuilder

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder
```

````{py:method} add_body(name: str, *, mass: float = 0.0, com: torch.Tensor | None = None, inertia: torch.Tensor | None = None) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_body

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_body
```

````

````{py:method} add_frame(name: str, *, parent_body: str, placement: torch.Tensor, frame_type: str = 'op') -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_frame

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_frame
```

````

````{py:method} add_collision_geom(body: str, kind: str, params: dict, origin: torch.Tensor) -> None
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_collision_geom

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_collision_geom
```

````

````{py:method} add_revolute(name: str, *, parent: str, child: str, axis: torch.Tensor, origin: torch.Tensor | None = None, lower: float | None = None, upper: float | None = None, velocity_limit: float | None = None, effort_limit: float | None = None, unbounded: bool = False, mimic_source: str | None = None, mimic_multiplier: float = 1.0, mimic_offset: float = 0.0) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_revolute

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_revolute
```

````

````{py:method} add_revolute_x(name: str, **kwargs) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_revolute_x

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_revolute_x
```

````

````{py:method} add_revolute_y(name: str, **kwargs) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_revolute_y

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_revolute_y
```

````

````{py:method} add_revolute_z(name: str, **kwargs) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_revolute_z

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_revolute_z
```

````

````{py:method} add_prismatic(name: str, *, parent: str, child: str, axis: torch.Tensor, origin: torch.Tensor | None = None, lower: float | None = None, upper: float | None = None, velocity_limit: float | None = None, effort_limit: float | None = None) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_prismatic

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_prismatic
```

````

````{py:method} add_prismatic_x(name: str, **kwargs) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_prismatic_x

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_prismatic_x
```

````

````{py:method} add_prismatic_y(name: str, **kwargs) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_prismatic_y

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_prismatic_y
```

````

````{py:method} add_prismatic_z(name: str, **kwargs) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_prismatic_z

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_prismatic_z
```

````

````{py:method} add_spherical(name: str, *, parent: str, child: str, origin: torch.Tensor | None = None) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_spherical

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_spherical
```

````

````{py:method} add_planar(name: str, *, parent: str, child: str, origin: torch.Tensor | None = None) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_planar

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_planar
```

````

````{py:method} add_helical(name: str, *, parent: str, child: str, axis: torch.Tensor, pitch: float, origin: torch.Tensor | None = None, lower: float | None = None, upper: float | None = None) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_helical

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_helical
```

````

````{py:method} add_free_flyer_root(name: str = 'free_flyer', *, child: str, origin: torch.Tensor | None = None) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_free_flyer_root

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_free_flyer_root
```

````

````{py:method} add_fixed(name: str, *, parent: str, child: str, origin: torch.Tensor | None = None) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_fixed

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_fixed
```

````

````{py:method} add_joint(name: str, *, kind=None, parent: str, child: str, origin: torch.Tensor, axis: torch.Tensor | None = None, lower: float | None = None, upper: float | None = None, velocity_limit: float | None = None, effort_limit: float | None = None, mimic_source: str | None = None, mimic_multiplier: float = 1.0, mimic_offset: float = 0.0) -> str
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.add_joint

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.add_joint
```

````

````{py:method} finalize() -> better_robot.io.ir.IRModel
:canonical: better_robot.io.parsers.programmatic.ModelBuilder.finalize

```{autodoc2-docstring} better_robot.io.parsers.programmatic.ModelBuilder.finalize
```

````

`````
