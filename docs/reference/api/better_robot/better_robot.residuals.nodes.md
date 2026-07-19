# {py:mod}`better_robot.residuals.nodes`

```{py:module} better_robot.residuals.nodes
```

```{autodoc2-docstring} better_robot.residuals.nodes
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Node <better_robot.residuals.nodes.Node>`
  - ```{autodoc2-docstring} better_robot.residuals.nodes.Node
    :summary:
    ```
* - {py:obj}`RobotState <better_robot.residuals.nodes.RobotState>`
  - ```{autodoc2-docstring} better_robot.residuals.nodes.RobotState
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`robot_state <better_robot.residuals.nodes.robot_state>`
  - ```{autodoc2-docstring} better_robot.residuals.nodes.robot_state
    :summary:
    ```
````

### API

`````{py:class} Node(*variables: better_robot.residuals._variables.ValueLike)
:canonical: better_robot.residuals.nodes.Node

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} better_robot.residuals.nodes.Node
```

````{py:property} merge_key
:canonical: better_robot.residuals.nodes.Node.merge_key
:type: tuple[object, ...] | None

```{autodoc2-docstring} better_robot.residuals.nodes.Node.merge_key
```

````

````{py:method} value()
:canonical: better_robot.residuals.nodes.Node.value

```{autodoc2-docstring} better_robot.residuals.nodes.Node.value
```

````

````{py:method} compute()
:canonical: better_robot.residuals.nodes.Node.compute
:abstractmethod:

```{autodoc2-docstring} better_robot.residuals.nodes.Node.compute
```

````

`````

`````{py:class} RobotState(q: better_robot.residuals._variables.RobotValueLike)
:canonical: better_robot.residuals.nodes.RobotState

Bases: {py:obj}`better_robot.residuals.nodes.Node`

```{autodoc2-docstring} better_robot.residuals.nodes.RobotState
```

````{py:property} merge_key
:canonical: better_robot.residuals.nodes.RobotState.merge_key
:type: tuple[object, ...]

````

````{py:method} compute()
:canonical: better_robot.residuals.nodes.RobotState.compute

````

`````

````{py:function} robot_state(value: better_robot.residuals._variables.RobotVariableLike | better_robot.residuals.nodes.RobotState) -> tuple[better_robot.residuals._variables.RobotVariableLike, better_robot.residuals.nodes.RobotState]
:canonical: better_robot.residuals.nodes.robot_state

```{autodoc2-docstring} better_robot.residuals.nodes.robot_state
```
````
