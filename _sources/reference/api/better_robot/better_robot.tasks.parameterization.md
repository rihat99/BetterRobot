# {py:mod}`better_robot.tasks.parameterization`

```{py:module} better_robot.tasks.parameterization
```

```{autodoc2-docstring} better_robot.tasks.parameterization
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrajectoryParameterization <better_robot.tasks.parameterization.TrajectoryParameterization>`
  - ```{autodoc2-docstring} better_robot.tasks.parameterization.TrajectoryParameterization
    :summary:
    ```
* - {py:obj}`KnotTrajectory <better_robot.tasks.parameterization.KnotTrajectory>`
  - ```{autodoc2-docstring} better_robot.tasks.parameterization.KnotTrajectory
    :summary:
    ```
* - {py:obj}`BSplineTrajectory <better_robot.tasks.parameterization.BSplineTrajectory>`
  - ```{autodoc2-docstring} better_robot.tasks.parameterization.BSplineTrajectory
    :summary:
    ```
````

### API

`````{py:class} TrajectoryParameterization
:canonical: better_robot.tasks.parameterization.TrajectoryParameterization

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.tasks.parameterization.TrajectoryParameterization
```

````{py:method} init(q_traj_seed: torch.Tensor) -> torch.Tensor
:canonical: better_robot.tasks.parameterization.TrajectoryParameterization.init

```{autodoc2-docstring} better_robot.tasks.parameterization.TrajectoryParameterization.init
```

````

````{py:method} expand(z: torch.Tensor, *, T: int, nq: int) -> torch.Tensor
:canonical: better_robot.tasks.parameterization.TrajectoryParameterization.expand

```{autodoc2-docstring} better_robot.tasks.parameterization.TrajectoryParameterization.expand
```

````

````{py:method} tangent_dim_per_step() -> int
:canonical: better_robot.tasks.parameterization.TrajectoryParameterization.tangent_dim_per_step

```{autodoc2-docstring} better_robot.tasks.parameterization.TrajectoryParameterization.tangent_dim_per_step
```

````

`````

`````{py:class} KnotTrajectory()
:canonical: better_robot.tasks.parameterization.KnotTrajectory

```{autodoc2-docstring} better_robot.tasks.parameterization.KnotTrajectory
```

````{py:method} init(q_traj_seed: torch.Tensor) -> torch.Tensor
:canonical: better_robot.tasks.parameterization.KnotTrajectory.init

```{autodoc2-docstring} better_robot.tasks.parameterization.KnotTrajectory.init
```

````

````{py:method} expand(z: torch.Tensor, *, T: int, nq: int) -> torch.Tensor
:canonical: better_robot.tasks.parameterization.KnotTrajectory.expand

```{autodoc2-docstring} better_robot.tasks.parameterization.KnotTrajectory.expand
```

````

````{py:method} tangent_dim_per_step() -> int
:canonical: better_robot.tasks.parameterization.KnotTrajectory.tangent_dim_per_step

```{autodoc2-docstring} better_robot.tasks.parameterization.KnotTrajectory.tangent_dim_per_step
```

````

`````

`````{py:class} BSplineTrajectory(*, num_control_points: int, degree: int = 3)
:canonical: better_robot.tasks.parameterization.BSplineTrajectory

```{autodoc2-docstring} better_robot.tasks.parameterization.BSplineTrajectory
```

````{py:method} init(q_traj_seed: torch.Tensor) -> torch.Tensor
:canonical: better_robot.tasks.parameterization.BSplineTrajectory.init

```{autodoc2-docstring} better_robot.tasks.parameterization.BSplineTrajectory.init
```

````

````{py:method} expand(z: torch.Tensor, *, T: int, nq: int) -> torch.Tensor
:canonical: better_robot.tasks.parameterization.BSplineTrajectory.expand

```{autodoc2-docstring} better_robot.tasks.parameterization.BSplineTrajectory.expand
```

````

````{py:method} tangent_dim_per_step() -> int
:canonical: better_robot.tasks.parameterization.BSplineTrajectory.tangent_dim_per_step

```{autodoc2-docstring} better_robot.tasks.parameterization.BSplineTrajectory.tangent_dim_per_step
```

````

`````
