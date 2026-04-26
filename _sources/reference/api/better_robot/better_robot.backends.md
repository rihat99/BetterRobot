# {py:mod}`better_robot.backends`

```{py:module} better_robot.backends
```

```{autodoc2-docstring} better_robot.backends
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

better_robot.backends.torch_native
better_robot.backends.warp
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

better_robot.backends.protocol
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_backend <better_robot.backends.get_backend>`
  - ```{autodoc2-docstring} better_robot.backends.get_backend
    :summary:
    ```
* - {py:obj}`default_backend <better_robot.backends.default_backend>`
  - ```{autodoc2-docstring} better_robot.backends.default_backend
    :summary:
    ```
* - {py:obj}`current_backend <better_robot.backends.current_backend>`
  - ```{autodoc2-docstring} better_robot.backends.current_backend
    :summary:
    ```
* - {py:obj}`current <better_robot.backends.current>`
  - ```{autodoc2-docstring} better_robot.backends.current
    :summary:
    ```
* - {py:obj}`set_backend <better_robot.backends.set_backend>`
  - ```{autodoc2-docstring} better_robot.backends.set_backend
    :summary:
    ```
* - {py:obj}`graph_capture <better_robot.backends.graph_capture>`
  - ```{autodoc2-docstring} better_robot.backends.graph_capture
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BackendName <better_robot.backends.BackendName>`
  - ```{autodoc2-docstring} better_robot.backends.BackendName
    :summary:
    ```
````

### API

````{py:data} BackendName
:canonical: better_robot.backends.BackendName
:value: >
   None

```{autodoc2-docstring} better_robot.backends.BackendName
```

````

````{py:function} get_backend(name: str) -> better_robot.backends.protocol.Backend
:canonical: better_robot.backends.get_backend

```{autodoc2-docstring} better_robot.backends.get_backend
```
````

````{py:function} default_backend() -> better_robot.backends.protocol.Backend
:canonical: better_robot.backends.default_backend

```{autodoc2-docstring} better_robot.backends.default_backend
```
````

````{py:function} current_backend() -> str
:canonical: better_robot.backends.current_backend

```{autodoc2-docstring} better_robot.backends.current_backend
```
````

````{py:function} current() -> better_robot.backends.protocol.Backend
:canonical: better_robot.backends.current

```{autodoc2-docstring} better_robot.backends.current
```
````

````{py:function} set_backend(name: str) -> None
:canonical: better_robot.backends.set_backend

```{autodoc2-docstring} better_robot.backends.set_backend
```
````

````{py:function} graph_capture(fn)
:canonical: better_robot.backends.graph_capture

```{autodoc2-docstring} better_robot.backends.graph_capture
```
````
