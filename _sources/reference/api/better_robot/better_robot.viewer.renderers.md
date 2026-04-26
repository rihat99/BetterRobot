# {py:mod}`better_robot.viewer.renderers`

```{py:module} better_robot.viewer.renderers
```

```{autodoc2-docstring} better_robot.viewer.renderers
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

better_robot.viewer.renderers.testing
better_robot.viewer.renderers.base
better_robot.viewer.renderers.offscreen_backend
better_robot.viewer.renderers.viser_backend
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_renderer <better_robot.viewer.renderers.get_renderer>`
  - ```{autodoc2-docstring} better_robot.viewer.renderers.get_renderer
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RENDERER_REGISTRY <better_robot.viewer.renderers.RENDERER_REGISTRY>`
  - ```{autodoc2-docstring} better_robot.viewer.renderers.RENDERER_REGISTRY
    :summary:
    ```
````

### API

````{py:data} RENDERER_REGISTRY
:canonical: better_robot.viewer.renderers.RENDERER_REGISTRY
:type: dict[str, type]
:value: >
   None

```{autodoc2-docstring} better_robot.viewer.renderers.RENDERER_REGISTRY
```

````

````{py:function} get_renderer(name: str) -> type
:canonical: better_robot.viewer.renderers.get_renderer

```{autodoc2-docstring} better_robot.viewer.renderers.get_renderer
```
````
