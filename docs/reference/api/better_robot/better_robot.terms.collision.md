# {py:mod}`better_robot.terms.collision`

```{py:module} better_robot.terms.collision
```

```{autodoc2-docstring} better_robot.terms.collision
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`sphere_clearance <better_robot.terms.collision.sphere_clearance>`
  - ```{autodoc2-docstring} better_robot.terms.collision.sphere_clearance
    :summary:
    ```
* - {py:obj}`self_clearance <better_robot.terms.collision.self_clearance>`
  - ```{autodoc2-docstring} better_robot.terms.collision.self_clearance
    :summary:
    ```
````

### API

````{py:function} sphere_clearance(spheres: better_robot.collision.RobotSpheres, distance_fn: collections.abc.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], *, margin: float, hinge: bool = True, state: str = 'state') -> better_robot.optim2.TermFn
:canonical: better_robot.terms.collision.sphere_clearance

```{autodoc2-docstring} better_robot.terms.collision.sphere_clearance
```
````

````{py:function} self_clearance(spheres: better_robot.collision.RobotSpheres, pairs: torch.Tensor, *, margin: float, hinge: bool = True, state: str = 'state') -> better_robot.optim2.TermFn
:canonical: better_robot.terms.collision.self_clearance

```{autodoc2-docstring} better_robot.terms.collision.self_clearance
```
````
