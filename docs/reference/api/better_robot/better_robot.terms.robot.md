# {py:mod}`better_robot.terms.robot`

```{py:module} better_robot.terms.robot
```

```{autodoc2-docstring} better_robot.terms.robot
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RobotManifold <better_robot.terms.robot.RobotManifold>`
  - ```{autodoc2-docstring} better_robot.terms.robot.RobotManifold
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`kinematic_state <better_robot.terms.robot.kinematic_state>`
  - ```{autodoc2-docstring} better_robot.terms.robot.kinematic_state
    :summary:
    ```
* - {py:obj}`frame_pose <better_robot.terms.robot.frame_pose>`
  - ```{autodoc2-docstring} better_robot.terms.robot.frame_pose
    :summary:
    ```
* - {py:obj}`frame_position <better_robot.terms.robot.frame_position>`
  - ```{autodoc2-docstring} better_robot.terms.robot.frame_position
    :summary:
    ```
* - {py:obj}`joint_limits <better_robot.terms.robot.joint_limits>`
  - ```{autodoc2-docstring} better_robot.terms.robot.joint_limits
    :summary:
    ```
* - {py:obj}`rest <better_robot.terms.robot.rest>`
  - ```{autodoc2-docstring} better_robot.terms.robot.rest
    :summary:
    ```
* - {py:obj}`finite_difference <better_robot.terms.robot.finite_difference>`
  - ```{autodoc2-docstring} better_robot.terms.robot.finite_difference
    :summary:
    ```
````

### API

`````{py:class} RobotManifold(model: better_robot.data_model.model.Model)
:canonical: better_robot.terms.robot.RobotManifold

```{autodoc2-docstring} better_robot.terms.robot.RobotManifold
```

````{py:method} tangent_dim(x: torch.Tensor) -> int
:canonical: better_robot.terms.robot.RobotManifold.tangent_dim

```{autodoc2-docstring} better_robot.terms.robot.RobotManifold.tangent_dim
```

````

````{py:method} retract(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor
:canonical: better_robot.terms.robot.RobotManifold.retract

```{autodoc2-docstring} better_robot.terms.robot.RobotManifold.retract
```

````

````{py:method} project(x: torch.Tensor) -> torch.Tensor
:canonical: better_robot.terms.robot.RobotManifold.project

```{autodoc2-docstring} better_robot.terms.robot.RobotManifold.project
```

````

````{py:method} tangent_bounds(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.terms.robot.RobotManifold.tangent_bounds

```{autodoc2-docstring} better_robot.terms.robot.RobotManifold.tangent_bounds
```

````

`````

````{py:function} kinematic_state(model: better_robot.data_model.model.Model, *, q: str = 'q') -> better_robot.optim2.TermFn
:canonical: better_robot.terms.robot.kinematic_state

```{autodoc2-docstring} better_robot.terms.robot.kinematic_state
```
````

````{py:function} frame_pose(model: better_robot.data_model.model.Model, frame: int | str, *, target: str, state: str = 'state', q: str = 'q') -> better_robot.optim2.TermFn
:canonical: better_robot.terms.robot.frame_pose

```{autodoc2-docstring} better_robot.terms.robot.frame_pose
```
````

````{py:function} frame_position(model: better_robot.data_model.model.Model, frame: int | str, *, target: str, state: str = 'state', q: str = 'q') -> better_robot.optim2.TermFn
:canonical: better_robot.terms.robot.frame_position

```{autodoc2-docstring} better_robot.terms.robot.frame_position
```
````

````{py:function} joint_limits(model: better_robot.data_model.model.Model, *, q: str = 'q') -> better_robot.optim2.TermFn
:canonical: better_robot.terms.robot.joint_limits

```{autodoc2-docstring} better_robot.terms.robot.joint_limits
```
````

````{py:function} rest(model: better_robot.data_model.model.Model, q_rest: torch.Tensor, *, q: str = 'q') -> better_robot.optim2.TermFn
:canonical: better_robot.terms.robot.rest

```{autodoc2-docstring} better_robot.terms.robot.rest
```
````

````{py:function} finite_difference(model: better_robot.data_model.model.Model, *, order: int, dt: float, q: str = 'q') -> better_robot.optim2.TermFn
:canonical: better_robot.terms.robot.finite_difference

```{autodoc2-docstring} better_robot.terms.robot.finite_difference
```
````
