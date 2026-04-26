# {py:mod}`better_robot.optim.optimizers.lm_then_lbfgs`

```{py:module} better_robot.optim.optimizers.lm_then_lbfgs
```

```{autodoc2-docstring} better_robot.optim.optimizers.lm_then_lbfgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LMThenLBFGS <better_robot.optim.optimizers.lm_then_lbfgs.LMThenLBFGS>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.lm_then_lbfgs.LMThenLBFGS
    :summary:
    ```
````

### API

`````{py:class} LMThenLBFGS(*, stage1_max_iter: int = 50, stage2_max_iter: int = 50, stage2_disabled_items: collections.abc.Iterable[str] = (), tol: float = 1e-06, lm: better_robot.optim.optimizers.levenberg_marquardt.LevenbergMarquardt | None = None, lbfgs: better_robot.optim.optimizers.lbfgs.LBFGS | None = None)
:canonical: better_robot.optim.optimizers.lm_then_lbfgs.LMThenLBFGS

```{autodoc2-docstring} better_robot.optim.optimizers.lm_then_lbfgs.LMThenLBFGS
```

````{py:method} minimize(problem, *, max_iter=None, linear_solver=None, kernel=None, strategy=None, scheduler=None)
:canonical: better_robot.optim.optimizers.lm_then_lbfgs.LMThenLBFGS.minimize

```{autodoc2-docstring} better_robot.optim.optimizers.lm_then_lbfgs.LMThenLBFGS.minimize
```

````

`````
