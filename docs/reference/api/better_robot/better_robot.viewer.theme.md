# {py:mod}`better_robot.viewer.theme`

```{py:module} better_robot.viewer.theme
```

```{autodoc2-docstring} better_robot.viewer.theme
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Theme <better_robot.viewer.theme.Theme>`
  - ```{autodoc2-docstring} better_robot.viewer.theme.Theme
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LIGHT <better_robot.viewer.theme.LIGHT>`
  - ```{autodoc2-docstring} better_robot.viewer.theme.LIGHT
    :summary:
    ```
* - {py:obj}`DARK <better_robot.viewer.theme.DARK>`
  - ```{autodoc2-docstring} better_robot.viewer.theme.DARK
    :summary:
    ```
````

### API

`````{py:class} Theme
:canonical: better_robot.viewer.theme.Theme

```{autodoc2-docstring} better_robot.viewer.theme.Theme
```

````{py:attribute} dark
:canonical: better_robot.viewer.theme.Theme.dark
:type: bool
:value: >
   False

```{autodoc2-docstring} better_robot.viewer.theme.Theme.dark
```

````

````{py:attribute} brand
:canonical: better_robot.viewer.theme.Theme.brand
:type: better_robot.viewer._math.RGB
:value: >
   (0.16, 0.45, 0.85)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.brand
```

````

````{py:attribute} joint
:canonical: better_robot.viewer.theme.Theme.joint
:type: better_robot.viewer._math.RGB
:value: >
   (0.2, 0.6, 1.0)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.joint
```

````

````{py:attribute} link
:canonical: better_robot.viewer.theme.Theme.link
:type: better_robot.viewer._math.RGB
:value: >
   (0.72, 0.72, 0.72)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.link
```

````

````{py:attribute} robot
:canonical: better_robot.viewer.theme.Theme.robot
:type: better_robot.viewer._math.RGB
:value: >
   (0.8, 0.8, 0.8)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.robot
```

````

````{py:attribute} human
:canonical: better_robot.viewer.theme.Theme.human
:type: better_robot.viewer._math.RGB
:value: >
   (0.85, 0.72, 0.62)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.human
```

````

````{py:attribute} point
:canonical: better_robot.viewer.theme.Theme.point
:type: better_robot.viewer._math.RGB
:value: >
   (0.25, 0.25, 0.25)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.point
```

````

````{py:attribute} force
:canonical: better_robot.viewer.theme.Theme.force
:type: better_robot.viewer._math.RGB
:value: >
   (0.9, 0.25, 0.2)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.force
```

````

````{py:attribute} torque
:canonical: better_robot.viewer.theme.Theme.torque
:type: better_robot.viewer._math.RGB
:value: >
   (0.95, 0.6, 0.1)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.torque
```

````

````{py:attribute} target
:canonical: better_robot.viewer.theme.Theme.target
:type: better_robot.viewer._math.RGB
:value: >
   (0.1, 0.75, 0.35)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.target
```

````

````{py:attribute} grid_cell
:canonical: better_robot.viewer.theme.Theme.grid_cell
:type: better_robot.viewer._math.RGB
:value: >
   (0.78, 0.78, 0.78)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.grid_cell
```

````

````{py:attribute} grid_section
:canonical: better_robot.viewer.theme.Theme.grid_section
:type: better_robot.viewer._math.RGB
:value: >
   (0.55, 0.55, 0.55)

```{autodoc2-docstring} better_robot.viewer.theme.Theme.grid_section
```

````

````{py:attribute} key_light_direction
:canonical: better_robot.viewer.theme.Theme.key_light_direction
:type: tuple[float, float, float]
:value: >
   ()

```{autodoc2-docstring} better_robot.viewer.theme.Theme.key_light_direction
```

````

````{py:attribute} up
:canonical: better_robot.viewer.theme.Theme.up
:type: str
:value: >
   '+z'

```{autodoc2-docstring} better_robot.viewer.theme.Theme.up
```

````

````{py:method} apply(server: typing.Any) -> None
:canonical: better_robot.viewer.theme.Theme.apply

```{autodoc2-docstring} better_robot.viewer.theme.Theme.apply
```

````

`````

````{py:data} LIGHT
:canonical: better_robot.viewer.theme.LIGHT
:value: >
   'Theme(...)'

```{autodoc2-docstring} better_robot.viewer.theme.LIGHT
```

````

````{py:data} DARK
:canonical: better_robot.viewer.theme.DARK
:value: >
   'Theme(...)'

```{autodoc2-docstring} better_robot.viewer.theme.DARK
```

````
