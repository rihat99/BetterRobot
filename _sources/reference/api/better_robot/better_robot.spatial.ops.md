# {py:mod}`better_robot.spatial.ops`

```{py:module} better_robot.spatial.ops
```

```{autodoc2-docstring} better_robot.spatial.ops
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ad <better_robot.spatial.ops.ad>`
  - ```{autodoc2-docstring} better_robot.spatial.ops.ad
    :summary:
    ```
* - {py:obj}`ad_star <better_robot.spatial.ops.ad_star>`
  - ```{autodoc2-docstring} better_robot.spatial.ops.ad_star
    :summary:
    ```
* - {py:obj}`cross_mm <better_robot.spatial.ops.cross_mm>`
  - ```{autodoc2-docstring} better_robot.spatial.ops.cross_mm
    :summary:
    ```
* - {py:obj}`cross_mf <better_robot.spatial.ops.cross_mf>`
  - ```{autodoc2-docstring} better_robot.spatial.ops.cross_mf
    :summary:
    ```
* - {py:obj}`act_motion <better_robot.spatial.ops.act_motion>`
  - ```{autodoc2-docstring} better_robot.spatial.ops.act_motion
    :summary:
    ```
* - {py:obj}`act_force <better_robot.spatial.ops.act_force>`
  - ```{autodoc2-docstring} better_robot.spatial.ops.act_force
    :summary:
    ```
* - {py:obj}`act_inertia <better_robot.spatial.ops.act_inertia>`
  - ```{autodoc2-docstring} better_robot.spatial.ops.act_inertia
    :summary:
    ```
````

### API

````{py:function} ad(v: better_robot.spatial.motion.Motion) -> torch.Tensor
:canonical: better_robot.spatial.ops.ad

```{autodoc2-docstring} better_robot.spatial.ops.ad
```
````

````{py:function} ad_star(f: better_robot.spatial.force.Force) -> torch.Tensor
:canonical: better_robot.spatial.ops.ad_star

```{autodoc2-docstring} better_robot.spatial.ops.ad_star
```
````

````{py:function} cross_mm(a: better_robot.spatial.motion.Motion, b: better_robot.spatial.motion.Motion) -> better_robot.spatial.motion.Motion
:canonical: better_robot.spatial.ops.cross_mm

```{autodoc2-docstring} better_robot.spatial.ops.cross_mm
```
````

````{py:function} cross_mf(a: better_robot.spatial.motion.Motion, b: better_robot.spatial.force.Force) -> better_robot.spatial.force.Force
:canonical: better_robot.spatial.ops.cross_mf

```{autodoc2-docstring} better_robot.spatial.ops.cross_mf
```
````

````{py:function} act_motion(T: torch.Tensor, m: better_robot.spatial.motion.Motion) -> better_robot.spatial.motion.Motion
:canonical: better_robot.spatial.ops.act_motion

```{autodoc2-docstring} better_robot.spatial.ops.act_motion
```
````

````{py:function} act_force(T: torch.Tensor, f: better_robot.spatial.force.Force) -> better_robot.spatial.force.Force
:canonical: better_robot.spatial.ops.act_force

```{autodoc2-docstring} better_robot.spatial.ops.act_force
```
````

````{py:function} act_inertia(T: torch.Tensor, I: better_robot.spatial.inertia.Inertia) -> better_robot.spatial.inertia.Inertia
:canonical: better_robot.spatial.ops.act_inertia

```{autodoc2-docstring} better_robot.spatial.ops.act_inertia
```
````
