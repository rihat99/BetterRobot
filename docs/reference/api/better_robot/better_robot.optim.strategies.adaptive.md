# {py:mod}`better_robot.optim.strategies.adaptive`

```{py:module} better_robot.optim.strategies.adaptive
```

```{autodoc2-docstring} better_robot.optim.strategies.adaptive
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Adaptive <better_robot.optim.strategies.adaptive.Adaptive>`
  - ```{autodoc2-docstring} better_robot.optim.strategies.adaptive.Adaptive
    :summary:
    ```
````

### API

`````{py:class} Adaptive(*, lam0: float = 0.0001, factor: float = 2.0)
:canonical: better_robot.optim.strategies.adaptive.Adaptive

```{autodoc2-docstring} better_robot.optim.strategies.adaptive.Adaptive
```

````{py:method} init(problem) -> float
:canonical: better_robot.optim.strategies.adaptive.Adaptive.init

```{autodoc2-docstring} better_robot.optim.strategies.adaptive.Adaptive.init
```

````

````{py:method} accept(lam: float) -> float
:canonical: better_robot.optim.strategies.adaptive.Adaptive.accept

```{autodoc2-docstring} better_robot.optim.strategies.adaptive.Adaptive.accept
```

````

````{py:method} reject(lam: float) -> float
:canonical: better_robot.optim.strategies.adaptive.Adaptive.reject

```{autodoc2-docstring} better_robot.optim.strategies.adaptive.Adaptive.reject
```

````

`````
