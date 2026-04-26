# {py:mod}`better_robot.data_model.joint_models.prismatic`

```{py:module} better_robot.data_model.joint_models.prismatic
```

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointPX <better_robot.data_model.joint_models.prismatic.JointPX>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX
    :summary:
    ```
* - {py:obj}`JointPY <better_robot.data_model.joint_models.prismatic.JointPY>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY
    :summary:
    ```
* - {py:obj}`JointPZ <better_robot.data_model.joint_models.prismatic.JointPZ>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ
    :summary:
    ```
* - {py:obj}`JointPrismaticUnaligned <better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned>`
  - ```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned
    :summary:
    ```
````

### API

`````{py:class} JointPX
:canonical: better_robot.data_model.joint_models.prismatic.JointPX

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.kind
:type: str
:value: >
   'prismatic_px'

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.nq
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.nv
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.axis
```

````

````{py:method} joint_transform(q_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.integrate
```

````

````{py:method} difference(q0_slice, q1_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.difference
```

````

````{py:method} random_configuration(generator, lower, upper)
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.random_configuration
```

````

````{py:method} neutral()
:canonical: better_robot.data_model.joint_models.prismatic.JointPX.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPX.neutral
```

````

`````

`````{py:class} JointPY
:canonical: better_robot.data_model.joint_models.prismatic.JointPY

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.kind
:type: str
:value: >
   'prismatic_py'

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.nq
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.nv
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.axis
```

````

````{py:method} joint_transform(q_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.integrate
```

````

````{py:method} difference(q0_slice, q1_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.difference
```

````

````{py:method} random_configuration(generator, lower, upper)
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.random_configuration
```

````

````{py:method} neutral()
:canonical: better_robot.data_model.joint_models.prismatic.JointPY.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPY.neutral
```

````

`````

`````{py:class} JointPZ
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ
```

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.kind
:type: str
:value: >
   'prismatic_pz'

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.nq
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.nv
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.nv
```

````

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.axis
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.axis
```

````

````{py:method} joint_transform(q_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.integrate
```

````

````{py:method} difference(q0_slice, q1_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.difference
```

````

````{py:method} random_configuration(generator, lower, upper)
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.random_configuration
```

````

````{py:method} neutral()
:canonical: better_robot.data_model.joint_models.prismatic.JointPZ.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPZ.neutral
```

````

`````

`````{py:class} JointPrismaticUnaligned
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned
```

````{py:attribute} axis
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.axis
:type: torch.Tensor
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.axis
```

````

````{py:attribute} kind
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.kind
:type: str
:value: >
   'prismatic_unaligned'

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.kind
```

````

````{py:attribute} nq
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.nq
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.nq
```

````

````{py:attribute} nv
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.nv
:type: int
:value: >
   1

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.nv
```

````

````{py:method} joint_transform(q_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.joint_transform

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.joint_transform
```

````

````{py:method} joint_motion_subspace(q_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.joint_motion_subspace

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.joint_motion_subspace
```

````

````{py:method} joint_velocity(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.joint_velocity

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.joint_velocity
```

````

````{py:method} integrate(q_slice, v_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.integrate

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.integrate
```

````

````{py:method} difference(q0_slice, q1_slice)
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.difference

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.difference
```

````

````{py:method} random_configuration(generator, lower, upper)
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.random_configuration

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.random_configuration
```

````

````{py:method} neutral()
:canonical: better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.neutral

```{autodoc2-docstring} better_robot.data_model.joint_models.prismatic.JointPrismaticUnaligned.neutral
```

````

`````
