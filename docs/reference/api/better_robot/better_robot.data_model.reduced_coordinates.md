# {py:mod}`better_robot.data_model.reduced_coordinates`

```{py:module} better_robot.data_model.reduced_coordinates
```

```{autodoc2-docstring} better_robot.data_model.reduced_coordinates
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`expand_configuration <better_robot.data_model.reduced_coordinates.expand_configuration>`
  - ```{autodoc2-docstring} better_robot.data_model.reduced_coordinates.expand_configuration
    :summary:
    ```
* - {py:obj}`expand_tangent <better_robot.data_model.reduced_coordinates.expand_tangent>`
  - ```{autodoc2-docstring} better_robot.data_model.reduced_coordinates.expand_tangent
    :summary:
    ```
* - {py:obj}`reduce_generalized_force <better_robot.data_model.reduced_coordinates.reduce_generalized_force>`
  - ```{autodoc2-docstring} better_robot.data_model.reduced_coordinates.reduce_generalized_force
    :summary:
    ```
* - {py:obj}`reduce_jacobian <better_robot.data_model.reduced_coordinates.reduce_jacobian>`
  - ```{autodoc2-docstring} better_robot.data_model.reduced_coordinates.reduce_jacobian
    :summary:
    ```
* - {py:obj}`reduce_mass_matrix <better_robot.data_model.reduced_coordinates.reduce_mass_matrix>`
  - ```{autodoc2-docstring} better_robot.data_model.reduced_coordinates.reduce_mass_matrix
    :summary:
    ```
````

### API

````{py:function} expand_configuration(structure: better_robot.data_model.model_structure.ModelStructure, q: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.reduced_coordinates.expand_configuration

```{autodoc2-docstring} better_robot.data_model.reduced_coordinates.expand_configuration
```
````

````{py:function} expand_tangent(structure: better_robot.data_model.model_structure.ModelStructure, value: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.reduced_coordinates.expand_tangent

```{autodoc2-docstring} better_robot.data_model.reduced_coordinates.expand_tangent
```
````

````{py:function} reduce_generalized_force(structure: better_robot.data_model.model_structure.ModelStructure, value_full: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.reduced_coordinates.reduce_generalized_force

```{autodoc2-docstring} better_robot.data_model.reduced_coordinates.reduce_generalized_force
```
````

````{py:function} reduce_jacobian(structure: better_robot.data_model.model_structure.ModelStructure, jacobian_full: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.reduced_coordinates.reduce_jacobian

```{autodoc2-docstring} better_robot.data_model.reduced_coordinates.reduce_jacobian
```
````

````{py:function} reduce_mass_matrix(structure: better_robot.data_model.model_structure.ModelStructure, mass_full: torch.Tensor) -> torch.Tensor
:canonical: better_robot.data_model.reduced_coordinates.reduce_mass_matrix

```{autodoc2-docstring} better_robot.data_model.reduced_coordinates.reduce_mass_matrix
```
````
