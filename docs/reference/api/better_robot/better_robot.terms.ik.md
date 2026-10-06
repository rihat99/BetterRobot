# {py:mod}`better_robot.terms.ik`

```{py:module} better_robot.terms.ik
```

```{autodoc2-docstring} better_robot.terms.ik
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ik_problem <better_robot.terms.ik.ik_problem>`
  - ```{autodoc2-docstring} better_robot.terms.ik.ik_problem
    :summary:
    ```
* - {py:obj}`random_start <better_robot.terms.ik.random_start>`
  - ```{autodoc2-docstring} better_robot.terms.ik.random_start
    :summary:
    ```
* - {py:obj}`solve_ik <better_robot.terms.ik.solve_ik>`
  - ```{autodoc2-docstring} better_robot.terms.ik.solve_ik
    :summary:
    ```
````

### API

````{py:function} ik_problem(model: better_robot.data_model.model.Model, targets: collections.abc.Mapping[str, torch.Tensor], q0: torch.Tensor, *, limit_weight: float = 0.0, rest_weight: float = 0.0, q_rest: torch.Tensor | None = None) -> better_robot.optim2.Problem
:canonical: better_robot.terms.ik.ik_problem

```{autodoc2-docstring} better_robot.terms.ik.ik_problem
```
````

````{py:function} random_start(model: better_robot.data_model.model.Model, q0: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor
:canonical: better_robot.terms.ik.random_start

```{autodoc2-docstring} better_robot.terms.ik.random_start
```
````

````{py:function} solve_ik(model: better_robot.data_model.model.Model, targets: collections.abc.Mapping[str, torch.Tensor], q0: torch.Tensor, *, solver: better_robot.optim2.Solver | None = None, iters: int = 100, restarts: int = 0, generator: torch.Generator | None = None) -> better_robot.optim2.Result
:canonical: better_robot.terms.ik.solve_ik

```{autodoc2-docstring} better_robot.terms.ik.solve_ik
```
````
