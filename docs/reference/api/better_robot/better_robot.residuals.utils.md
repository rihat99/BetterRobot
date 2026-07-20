# {py:mod}`better_robot.residuals.utils`

```{py:module} better_robot.residuals.utils
```

```{autodoc2-docstring} better_robot.residuals.utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ValueLike <better_robot.residuals.utils.ValueLike>`
  -
* - {py:obj}`VariableLike <better_robot.residuals.utils.VariableLike>`
  -
* - {py:obj}`RobotValueLike <better_robot.residuals.utils.RobotValueLike>`
  -
* - {py:obj}`RobotLike <better_robot.residuals.utils.RobotLike>`
  -
* - {py:obj}`RobotVariableLike <better_robot.residuals.utils.RobotVariableLike>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`value <better_robot.residuals.utils.value>`
  - ```{autodoc2-docstring} better_robot.residuals.utils.value
    :summary:
    ```
* - {py:obj}`variables <better_robot.residuals.utils.variables>`
  - ```{autodoc2-docstring} better_robot.residuals.utils.variables
    :summary:
    ```
* - {py:obj}`static_value <better_robot.residuals.utils.static_value>`
  - ```{autodoc2-docstring} better_robot.residuals.utils.static_value
    :summary:
    ```
* - {py:obj}`current_value <better_robot.residuals.utils.current_value>`
  - ```{autodoc2-docstring} better_robot.residuals.utils.current_value
    :summary:
    ```
* - {py:obj}`matches <better_robot.residuals.utils.matches>`
  - ```{autodoc2-docstring} better_robot.residuals.utils.matches
    :summary:
    ```
* - {py:obj}`require_robot <better_robot.residuals.utils.require_robot>`
  - ```{autodoc2-docstring} better_robot.residuals.utils.require_robot
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TemporalLike <better_robot.residuals.utils.TemporalLike>`
  - ```{autodoc2-docstring} better_robot.residuals.utils.TemporalLike
    :summary:
    ```
````

### API

`````{py:class} ValueLike
:canonical: better_robot.residuals.utils.ValueLike

Bases: {py:obj}`typing.Protocol`

````{py:attribute} tensor
:canonical: better_robot.residuals.utils.ValueLike.tensor
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.utils.ValueLike.tensor
```

````

````{py:attribute} name
:canonical: better_robot.residuals.utils.ValueLike.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.utils.ValueLike.name
```

````

````{py:attribute} trainable
:canonical: better_robot.residuals.utils.ValueLike.trainable
:type: bool
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.utils.ValueLike.trainable
```

````

`````

`````{py:class} VariableLike
:canonical: better_robot.residuals.utils.VariableLike

Bases: {py:obj}`better_robot.residuals.utils.ValueLike`, {py:obj}`typing.Protocol`

````{py:attribute} shape
:canonical: better_robot.residuals.utils.VariableLike.shape
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.utils.VariableLike.shape
```

````

````{py:attribute} time_axis
:canonical: better_robot.residuals.utils.VariableLike.time_axis
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.utils.VariableLike.time_axis
```

````

````{py:method} tangent_dim() -> int
:canonical: better_robot.residuals.utils.VariableLike.tangent_dim

```{autodoc2-docstring} better_robot.residuals.utils.VariableLike.tangent_dim
```

````

````{py:method} difference(other: torch.Tensor) -> torch.Tensor
:canonical: better_robot.residuals.utils.VariableLike.difference

```{autodoc2-docstring} better_robot.residuals.utils.VariableLike.difference
```

````

`````

`````{py:class} RobotValueLike
:canonical: better_robot.residuals.utils.RobotValueLike

Bases: {py:obj}`better_robot.residuals.utils.ValueLike`, {py:obj}`typing.Protocol`

````{py:attribute} model
:canonical: better_robot.residuals.utils.RobotValueLike.model
:type: better_robot.data_model.model.Model
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.utils.RobotValueLike.model
```

````

`````

`````{py:class} RobotLike
:canonical: better_robot.residuals.utils.RobotLike

Bases: {py:obj}`better_robot.residuals.utils.VariableLike`, {py:obj}`typing.Protocol`

````{py:attribute} model
:canonical: better_robot.residuals.utils.RobotLike.model
:type: better_robot.data_model.model.Model
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.utils.RobotLike.model
```

````

`````

`````{py:class} RobotVariableLike
:canonical: better_robot.residuals.utils.RobotVariableLike

Bases: {py:obj}`better_robot.residuals.utils.RobotLike`, {py:obj}`typing.Protocol`

````{py:attribute} time_length
:canonical: better_robot.residuals.utils.RobotVariableLike.time_length
:type: int
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.utils.RobotVariableLike.time_length
```

````

`````

````{py:data} TemporalLike
:canonical: better_robot.residuals.utils.TemporalLike
:value: >
   None

```{autodoc2-docstring} better_robot.residuals.utils.TemporalLike
```

````

````{py:function} value(value: better_robot.residuals.utils.VariableLike | torch.Tensor, label: str) -> torch.Tensor
:canonical: better_robot.residuals.utils.value

```{autodoc2-docstring} better_robot.residuals.utils.value
```
````

````{py:function} variables(*values: object) -> tuple[better_robot.residuals.utils.VariableLike, ...]
:canonical: better_robot.residuals.utils.variables

```{autodoc2-docstring} better_robot.residuals.utils.variables
```
````

````{py:function} static_value(value: torch.Tensor | better_robot.residuals.utils.VariableLike, *, name: str) -> tuple[torch.Tensor, tuple[better_robot.residuals.utils.VariableLike, ...]]
:canonical: better_robot.residuals.utils.static_value

```{autodoc2-docstring} better_robot.residuals.utils.static_value
```
````

````{py:function} current_value(value: torch.Tensor | better_robot.residuals.utils.VariableLike, exemplar: torch.Tensor, *, name: str, preserve: bool = True) -> torch.Tensor
:canonical: better_robot.residuals.utils.current_value

```{autodoc2-docstring} better_robot.residuals.utils.current_value
```
````

````{py:function} matches(variable: better_robot.residuals.utils.ValueLike | str, expected: better_robot.residuals.utils.ValueLike) -> bool
:canonical: better_robot.residuals.utils.matches

```{autodoc2-docstring} better_robot.residuals.utils.matches
```
````

````{py:function} require_robot(variable: better_robot.residuals.utils.RobotVariableLike, name: str, *, temporal: bool) -> None
:canonical: better_robot.residuals.utils.require_robot

```{autodoc2-docstring} better_robot.residuals.utils.require_robot
```
````
