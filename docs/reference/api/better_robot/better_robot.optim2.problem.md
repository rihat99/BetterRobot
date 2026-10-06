# {py:mod}`better_robot.optim2.problem`

```{py:module} better_robot.optim2.problem
```

```{autodoc2-docstring} better_robot.optim2.problem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TermFn <better_robot.optim2.problem.TermFn>`
  - ```{autodoc2-docstring} better_robot.optim2.problem.TermFn
    :summary:
    ```
* - {py:obj}`Term <better_robot.optim2.problem.Term>`
  - ```{autodoc2-docstring} better_robot.optim2.problem.Term
    :summary:
    ```
* - {py:obj}`Evaluation <better_robot.optim2.problem.Evaluation>`
  - ```{autodoc2-docstring} better_robot.optim2.problem.Evaluation
    :summary:
    ```
* - {py:obj}`Problem <better_robot.optim2.problem.Problem>`
  - ```{autodoc2-docstring} better_robot.optim2.problem.Problem
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`check_jacobian <better_robot.optim2.problem.check_jacobian>`
  - ```{autodoc2-docstring} better_robot.optim2.problem.check_jacobian
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Values <better_robot.optim2.problem.Values>`
  - ```{autodoc2-docstring} better_robot.optim2.problem.Values
    :summary:
    ```
````

### API

````{py:data} Values
:canonical: better_robot.optim2.problem.Values
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.Values
```

````

`````{py:class} TermFn
:canonical: better_robot.optim2.problem.TermFn

```{autodoc2-docstring} better_robot.optim2.problem.TermFn
```

````{py:attribute} fn
:canonical: better_robot.optim2.problem.TermFn.fn
:type: collections.abc.Callable[..., torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.TermFn.fn
```

````

````{py:attribute} reads
:canonical: better_robot.optim2.problem.TermFn.reads
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.TermFn.reads
```

````

````{py:attribute} jac
:canonical: better_robot.optim2.problem.TermFn.jac
:type: collections.abc.Callable[..., dict[str, torch.Tensor]] | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.TermFn.jac
```

````

````{py:attribute} window
:canonical: better_robot.optim2.problem.TermFn.window
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.TermFn.window
```

````

`````

`````{py:class} Term
:canonical: better_robot.optim2.problem.Term

```{autodoc2-docstring} better_robot.optim2.problem.Term
```

````{py:attribute} name
:canonical: better_robot.optim2.problem.Term.name
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.Term.name
```

````

````{py:attribute} fn
:canonical: better_robot.optim2.problem.Term.fn
:type: collections.abc.Callable[..., torch.Tensor]
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.Term.fn
```

````

````{py:attribute} kind
:canonical: better_robot.optim2.problem.Term.kind
:type: str
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.Term.kind
```

````

````{py:attribute} reads
:canonical: better_robot.optim2.problem.Term.reads
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.Term.reads
```

````

````{py:attribute} weight
:canonical: better_robot.optim2.problem.Term.weight
:type: float | torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.Term.weight
```

````

````{py:attribute} kernel
:canonical: better_robot.optim2.problem.Term.kernel
:type: better_robot.optim2.kernels.RobustKernel | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.Term.kernel
```

````

````{py:attribute} jac
:canonical: better_robot.optim2.problem.Term.jac
:type: collections.abc.Callable[..., dict[str, torch.Tensor]] | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.Term.jac
```

````

````{py:attribute} window
:canonical: better_robot.optim2.problem.Term.window
:type: int | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim2.problem.Term.window
```

````

`````

`````{py:class} Evaluation(problem: better_robot.optim2.problem.Problem, x: better_robot.optim2.problem.Values)
:canonical: better_robot.optim2.problem.Evaluation

```{autodoc2-docstring} better_robot.optim2.problem.Evaluation
```

````{py:method} args(reads: tuple[str, ...]) -> list[typing.Any]
:canonical: better_robot.optim2.problem.Evaluation.args

```{autodoc2-docstring} better_robot.optim2.problem.Evaluation.args
```

````

`````

`````{py:class} Problem(variables: collections.abc.Mapping[str, torch.Tensor], *, manifolds: collections.abc.Mapping[str, better_robot.optim2.manifold.Manifold] | None = None, bounds: collections.abc.Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None, data: collections.abc.Mapping[str, typing.Any] | None = None, time: tuple[str, ...] = (), dense: bool = False)
:canonical: better_robot.optim2.problem.Problem

```{autodoc2-docstring} better_robot.optim2.problem.Problem
```

````{py:method} cached(fn: collections.abc.Callable[..., typing.Any], *, name: str | None = None, reads: tuple[str, ...] | None = None)
:canonical: better_robot.optim2.problem.Problem.cached

```{autodoc2-docstring} better_robot.optim2.problem.Problem.cached
```

````

````{py:method} residual(name: str, fn: collections.abc.Callable[..., torch.Tensor], *, weight: float | torch.Tensor = 1.0, kernel: better_robot.optim2.kernels.RobustKernel | None = None, jac: collections.abc.Callable[..., dict[str, torch.Tensor]] | None = None, reads: tuple[str, ...] | None = None, window: int | None = None) -> None
:canonical: better_robot.optim2.problem.Problem.residual

```{autodoc2-docstring} better_robot.optim2.problem.Problem.residual
```

````

````{py:method} scalar(name: str, fn: collections.abc.Callable[..., torch.Tensor], *, weight: float | torch.Tensor = 1.0, reads: tuple[str, ...] | None = None) -> None
:canonical: better_robot.optim2.problem.Problem.scalar

```{autodoc2-docstring} better_robot.optim2.problem.Problem.scalar
```

````

````{py:method} with_(*, weights: collections.abc.Mapping[str, float] | None = None, free: collections.abc.Mapping[str, torch.Tensor] | None = None)
:canonical: better_robot.optim2.problem.Problem.with_

```{autodoc2-docstring} better_robot.optim2.problem.Problem.with_
```

````

````{py:method} losses(x: better_robot.optim2.problem.Values | None = None) -> dict[str, torch.Tensor]
:canonical: better_robot.optim2.problem.Problem.losses

```{autodoc2-docstring} better_robot.optim2.problem.Problem.losses
```

````

````{py:method} loss(x: better_robot.optim2.problem.Values | None = None) -> torch.Tensor
:canonical: better_robot.optim2.problem.Problem.loss

```{autodoc2-docstring} better_robot.optim2.problem.Problem.loss
```

````

````{py:method} residuals(x: better_robot.optim2.problem.Values | None = None) -> dict[str, torch.Tensor]
:canonical: better_robot.optim2.problem.Problem.residuals

```{autodoc2-docstring} better_robot.optim2.problem.Problem.residuals
```

````

````{py:method} linearize(x: better_robot.optim2.problem.Values | None = None) -> tuple[better_robot.optim2.linear.Normal | better_robot.optim2.linear.BandedNormal | better_robot.optim2.linear.ArrowNormal, torch.Tensor]
:canonical: better_robot.optim2.problem.Problem.linearize

```{autodoc2-docstring} better_robot.optim2.problem.Problem.linearize
```

````

````{py:method} gradient(x: better_robot.optim2.problem.Values | None = None, *, create_graph: bool = False) -> tuple[torch.Tensor, torch.Tensor]
:canonical: better_robot.optim2.problem.Problem.gradient

```{autodoc2-docstring} better_robot.optim2.problem.Problem.gradient
```

````

````{py:method} retract(x: better_robot.optim2.problem.Values, delta: torch.Tensor) -> dict[str, torch.Tensor]
:canonical: better_robot.optim2.problem.Problem.retract

```{autodoc2-docstring} better_robot.optim2.problem.Problem.retract
```

````

````{py:method} project(x: better_robot.optim2.problem.Values) -> dict[str, torch.Tensor]
:canonical: better_robot.optim2.problem.Problem.project

```{autodoc2-docstring} better_robot.optim2.problem.Problem.project
```

````

````{py:method} step_bounds(x: better_robot.optim2.problem.Values) -> tuple[torch.Tensor, torch.Tensor] | None
:canonical: better_robot.optim2.problem.Problem.step_bounds

```{autodoc2-docstring} better_robot.optim2.problem.Problem.step_bounds
```

````

````{py:method} slices() -> dict[str, slice]
:canonical: better_robot.optim2.problem.Problem.slices

```{autodoc2-docstring} better_robot.optim2.problem.Problem.slices
```

````

`````

````{py:function} check_jacobian(problem: better_robot.optim2.problem.Problem, x: better_robot.optim2.problem.Values | None = None) -> dict[str, float]
:canonical: better_robot.optim2.problem.check_jacobian

```{autodoc2-docstring} better_robot.optim2.problem.check_jacobian
```
````
