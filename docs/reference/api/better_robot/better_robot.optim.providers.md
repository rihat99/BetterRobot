# {py:mod}`better_robot.optim.providers`

```{py:module} better_robot.optim.providers
```

```{autodoc2-docstring} better_robot.optim.providers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Provider <better_robot.optim.providers.Provider>`
  -
* - {py:obj}`EvaluationContext <better_robot.optim.providers.EvaluationContext>`
  -
* - {py:obj}`RobotStateProvider <better_robot.optim.providers.RobotStateProvider>`
  - ```{autodoc2-docstring} better_robot.optim.providers.RobotStateProvider
    :summary:
    ```
````

### API

`````{py:class} Provider
:canonical: better_robot.optim.providers.Provider

Bases: {py:obj}`typing.Protocol`

````{py:attribute} name
:canonical: better_robot.optim.providers.Provider.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.optim.providers.Provider.name
```

````

````{py:attribute} reads
:canonical: better_robot.optim.providers.Provider.reads
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.optim.providers.Provider.reads
```

````

````{py:attribute} outputs
:canonical: better_robot.optim.providers.Provider.outputs
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.optim.providers.Provider.outputs
```

````

`````

`````{py:class} EvaluationContext(values: collections.abc.Mapping[str, typing.Any], providers_by_output: collections.abc.Mapping[str, better_robot.optim.providers.Provider], free_indices: collections.abc.Mapping[str, typing.Any], temporal_free_indices: collections.abc.Mapping[str, typing.Any] | None = None)
:canonical: better_robot.optim.providers.EvaluationContext

Bases: {py:obj}`collections.abc.Mapping`\[{py:obj}`str`\, {py:obj}`typing.Any`\]

````{py:method} free_indices(variable: str)
:canonical: better_robot.optim.providers.EvaluationContext.free_indices

```{autodoc2-docstring} better_robot.optim.providers.EvaluationContext.free_indices
```

````

````{py:method} temporal_free_indices(variable: str)
:canonical: better_robot.optim.providers.EvaluationContext.temporal_free_indices

```{autodoc2-docstring} better_robot.optim.providers.EvaluationContext.temporal_free_indices
```

````

`````

`````{py:class} RobotStateProvider
:canonical: better_robot.optim.providers.RobotStateProvider

```{autodoc2-docstring} better_robot.optim.providers.RobotStateProvider
```

````{py:attribute} model
:canonical: better_robot.optim.providers.RobotStateProvider.model
:type: better_robot.data_model.model.Model
:value: >
   None

```{autodoc2-docstring} better_robot.optim.providers.RobotStateProvider.model
```

````

````{py:attribute} var
:canonical: better_robot.optim.providers.RobotStateProvider.var
:type: str
:value: >
   'q'

```{autodoc2-docstring} better_robot.optim.providers.RobotStateProvider.var
```

````

````{py:attribute} output
:canonical: better_robot.optim.providers.RobotStateProvider.output
:type: str
:value: >
   'data'

```{autodoc2-docstring} better_robot.optim.providers.RobotStateProvider.output
```

````

````{py:attribute} name
:canonical: better_robot.optim.providers.RobotStateProvider.name
:type: str
:value: >
   'robot_state'

```{autodoc2-docstring} better_robot.optim.providers.RobotStateProvider.name
```

````

````{py:property} reads
:canonical: better_robot.optim.providers.RobotStateProvider.reads
:type: tuple[str, ...]

```{autodoc2-docstring} better_robot.optim.providers.RobotStateProvider.reads
```

````

````{py:property} outputs
:canonical: better_robot.optim.providers.RobotStateProvider.outputs
:type: tuple[str, ...]

```{autodoc2-docstring} better_robot.optim.providers.RobotStateProvider.outputs
```

````

`````
