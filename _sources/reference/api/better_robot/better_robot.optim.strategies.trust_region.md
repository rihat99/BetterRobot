# {py:mod}`better_robot.optim.strategies.trust_region`

```{py:module} better_robot.optim.strategies.trust_region
```

```{autodoc2-docstring} better_robot.optim.strategies.trust_region
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrustRegion <better_robot.optim.strategies.trust_region.TrustRegion>`
  - ```{autodoc2-docstring} better_robot.optim.strategies.trust_region.TrustRegion
    :summary:
    ```
````

### API

`````{py:class} TrustRegion(*, radius: float = 1.0)
:canonical: better_robot.optim.strategies.trust_region.TrustRegion

```{autodoc2-docstring} better_robot.optim.strategies.trust_region.TrustRegion
```

````{py:method} init(problem) -> float
:canonical: better_robot.optim.strategies.trust_region.TrustRegion.init
:abstractmethod:

```{autodoc2-docstring} better_robot.optim.strategies.trust_region.TrustRegion.init
```

````

````{py:method} accept(lam: float) -> float
:canonical: better_robot.optim.strategies.trust_region.TrustRegion.accept
:abstractmethod:

```{autodoc2-docstring} better_robot.optim.strategies.trust_region.TrustRegion.accept
```

````

````{py:method} reject(lam: float) -> float
:canonical: better_robot.optim.strategies.trust_region.TrustRegion.reject
:abstractmethod:

```{autodoc2-docstring} better_robot.optim.strategies.trust_region.TrustRegion.reject
```

````

`````
