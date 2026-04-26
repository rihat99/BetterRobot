# {py:mod}`better_robot.data_model.topology`

```{py:module} better_robot.data_model.topology
```

```{autodoc2-docstring} better_robot.data_model.topology
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`topo_sort <better_robot.data_model.topology.topo_sort>`
  - ```{autodoc2-docstring} better_robot.data_model.topology.topo_sort
    :summary:
    ```
* - {py:obj}`build_children <better_robot.data_model.topology.build_children>`
  - ```{autodoc2-docstring} better_robot.data_model.topology.build_children
    :summary:
    ```
* - {py:obj}`build_subtrees <better_robot.data_model.topology.build_subtrees>`
  - ```{autodoc2-docstring} better_robot.data_model.topology.build_subtrees
    :summary:
    ```
* - {py:obj}`build_supports <better_robot.data_model.topology.build_supports>`
  - ```{autodoc2-docstring} better_robot.data_model.topology.build_supports
    :summary:
    ```
* - {py:obj}`get_subtree <better_robot.data_model.topology.get_subtree>`
  - ```{autodoc2-docstring} better_robot.data_model.topology.get_subtree
    :summary:
    ```
* - {py:obj}`get_support <better_robot.data_model.topology.get_support>`
  - ```{autodoc2-docstring} better_robot.data_model.topology.get_support
    :summary:
    ```
````

### API

````{py:function} topo_sort(parents: tuple[int, ...]) -> tuple[int, ...]
:canonical: better_robot.data_model.topology.topo_sort

```{autodoc2-docstring} better_robot.data_model.topology.topo_sort
```
````

````{py:function} build_children(parents: tuple[int, ...]) -> tuple[tuple[int, ...], ...]
:canonical: better_robot.data_model.topology.build_children

```{autodoc2-docstring} better_robot.data_model.topology.build_children
```
````

````{py:function} build_subtrees(parents: tuple[int, ...]) -> tuple[tuple[int, ...], ...]
:canonical: better_robot.data_model.topology.build_subtrees

```{autodoc2-docstring} better_robot.data_model.topology.build_subtrees
```
````

````{py:function} build_supports(parents: tuple[int, ...]) -> tuple[tuple[int, ...], ...]
:canonical: better_robot.data_model.topology.build_supports

```{autodoc2-docstring} better_robot.data_model.topology.build_supports
```
````

````{py:function} get_subtree(parents: tuple[int, ...], joint_id: int) -> tuple[int, ...]
:canonical: better_robot.data_model.topology.get_subtree

```{autodoc2-docstring} better_robot.data_model.topology.get_subtree
```
````

````{py:function} get_support(parents: tuple[int, ...], joint_id: int) -> tuple[int, ...]
:canonical: better_robot.data_model.topology.get_support

```{autodoc2-docstring} better_robot.data_model.topology.get_support
```
````
