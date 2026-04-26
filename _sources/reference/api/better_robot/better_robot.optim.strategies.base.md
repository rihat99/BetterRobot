# {py:mod}`better_robot.optim.strategies.base`

```{py:module} better_robot.optim.strategies.base
```

```{autodoc2-docstring} better_robot.optim.strategies.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DampingStrategy <better_robot.optim.strategies.base.DampingStrategy>`
  - ```{autodoc2-docstring} better_robot.optim.strategies.base.DampingStrategy
    :summary:
    ```
````

### API

`````{py:class} DampingStrategy
:canonical: better_robot.optim.strategies.base.DampingStrategy

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.optim.strategies.base.DampingStrategy
```

````{py:method} init(problem) -> float
:canonical: better_robot.optim.strategies.base.DampingStrategy.init

```{autodoc2-docstring} better_robot.optim.strategies.base.DampingStrategy.init
```

````

````{py:method} accept(lam: float) -> float
:canonical: better_robot.optim.strategies.base.DampingStrategy.accept

```{autodoc2-docstring} better_robot.optim.strategies.base.DampingStrategy.accept
```

````

````{py:method} reject(lam: float) -> float
:canonical: better_robot.optim.strategies.base.DampingStrategy.reject

```{autodoc2-docstring} better_robot.optim.strategies.base.DampingStrategy.reject
```

````

`````
