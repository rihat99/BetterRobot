# {py:mod}`better_robot.tasks.retarget`

```{py:module} better_robot.tasks.retarget
```

```{autodoc2-docstring} better_robot.tasks.retarget
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RetargetCostConfig <better_robot.tasks.retarget.RetargetCostConfig>`
  - ```{autodoc2-docstring} better_robot.tasks.retarget.RetargetCostConfig
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`retarget <better_robot.tasks.retarget.retarget>`
  - ```{autodoc2-docstring} better_robot.tasks.retarget.retarget
    :summary:
    ```
````

### API

`````{py:class} RetargetCostConfig
:canonical: better_robot.tasks.retarget.RetargetCostConfig

```{autodoc2-docstring} better_robot.tasks.retarget.RetargetCostConfig
```

````{py:attribute} pose_weight
:canonical: better_robot.tasks.retarget.RetargetCostConfig.pose_weight
:type: float
:value: >
   1.0

```{autodoc2-docstring} better_robot.tasks.retarget.RetargetCostConfig.pose_weight
```

````

````{py:attribute} smoothness_weight
:canonical: better_robot.tasks.retarget.RetargetCostConfig.smoothness_weight
:type: float
:value: >
   0.01

```{autodoc2-docstring} better_robot.tasks.retarget.RetargetCostConfig.smoothness_weight
```

````

````{py:attribute} limit_weight
:canonical: better_robot.tasks.retarget.RetargetCostConfig.limit_weight
:type: float
:value: >
   0.1

```{autodoc2-docstring} better_robot.tasks.retarget.RetargetCostConfig.limit_weight
```

````

`````

````{py:function} retarget(source_model: better_robot.data_model.model.Model, target_model: better_robot.data_model.model.Model, source_trajectory: better_robot.tasks.trajectory.Trajectory, *, frame_map: dict[str, str], cost_cfg: better_robot.tasks.retarget.RetargetCostConfig | None = None, optimizer_cfg: better_robot.tasks.ik.OptimizerConfig | None = None) -> better_robot.tasks.trajopt.TrajOptResult
:canonical: better_robot.tasks.retarget.retarget

```{autodoc2-docstring} better_robot.tasks.retarget.retarget
```
````
