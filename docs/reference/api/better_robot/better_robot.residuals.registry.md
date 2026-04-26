# {py:mod}`better_robot.residuals.registry`

```{py:module} better_robot.residuals.registry
```

```{autodoc2-docstring} better_robot.residuals.registry
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`register_residual <better_robot.residuals.registry.register_residual>`
  - ```{autodoc2-docstring} better_robot.residuals.registry.register_residual
    :summary:
    ```
* - {py:obj}`get_residual <better_robot.residuals.registry.get_residual>`
  - ```{autodoc2-docstring} better_robot.residuals.registry.get_residual
    :summary:
    ```
* - {py:obj}`registered_residuals <better_robot.residuals.registry.registered_residuals>`
  - ```{autodoc2-docstring} better_robot.residuals.registry.registered_residuals
    :summary:
    ```
````

### API

````{py:function} register_residual(name: str) -> typing.Callable[[type], type]
:canonical: better_robot.residuals.registry.register_residual

```{autodoc2-docstring} better_robot.residuals.registry.register_residual
```
````

````{py:function} get_residual(name: str) -> type
:canonical: better_robot.residuals.registry.get_residual

```{autodoc2-docstring} better_robot.residuals.registry.get_residual
```
````

````{py:function} registered_residuals() -> tuple[str, ...]
:canonical: better_robot.residuals.registry.registered_residuals

```{autodoc2-docstring} better_robot.residuals.registry.registered_residuals
```
````
