# {py:mod}`better_robot.dynamics.state_manifold`

```{py:module} better_robot.dynamics.state_manifold
```

```{autodoc2-docstring} better_robot.dynamics.state_manifold
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StateMultibody <better_robot.dynamics.state_manifold.StateMultibody>`
  - ```{autodoc2-docstring} better_robot.dynamics.state_manifold.StateMultibody
    :summary:
    ```
````

### API

`````{py:class} StateMultibody(model: better_robot.data_model.model.Model)
:canonical: better_robot.dynamics.state_manifold.StateMultibody

```{autodoc2-docstring} better_robot.dynamics.state_manifold.StateMultibody
```

````{py:method} zero() -> torch.Tensor
:canonical: better_robot.dynamics.state_manifold.StateMultibody.zero

```{autodoc2-docstring} better_robot.dynamics.state_manifold.StateMultibody.zero
```

````

````{py:method} integrate(x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor
:canonical: better_robot.dynamics.state_manifold.StateMultibody.integrate

```{autodoc2-docstring} better_robot.dynamics.state_manifold.StateMultibody.integrate
```

````

````{py:method} diff(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor
:canonical: better_robot.dynamics.state_manifold.StateMultibody.diff

```{autodoc2-docstring} better_robot.dynamics.state_manifold.StateMultibody.diff
```

````

````{py:method} jacobian_integrate(x: torch.Tensor, dx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.dynamics.state_manifold.StateMultibody.jacobian_integrate

```{autodoc2-docstring} better_robot.dynamics.state_manifold.StateMultibody.jacobian_integrate
```

````

````{py:method} jacobian_diff(x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.dynamics.state_manifold.StateMultibody.jacobian_diff

```{autodoc2-docstring} better_robot.dynamics.state_manifold.StateMultibody.jacobian_diff
```

````

`````
