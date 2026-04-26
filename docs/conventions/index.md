# Conventions

Cross-cutting **normative** specs — everything the library expects at
its boundaries. When the design specs in {doc}`/design/index` are the
*what*, these are the *how*.

```{toctree}
:maxdepth: 1

13_NAMING
14_PERFORMANCE
15_EXTENSION
16_TESTING
17_CONTRACTS
19_STYLE
20_PACKAGING
```

## When to consult which doc

| Spec | Read when |
|------|-----------|
| {doc}`13_NAMING` | Adding any new identifier; staring at the Pinocchio → BetterRobot rename table. |
| {doc}`14_PERFORMANCE` | Touching hot paths, debugging compile / latency / memory. |
| {doc}`15_EXTENSION` | Plugging in a new residual / joint / solver / backend / parameterisation. |
| {doc}`16_TESTING` | Adding tests; promoting a benchmark from advisory to blocking. |
| {doc}`17_CONTRACTS` | Touching the public API, raising a new exception, changing numerical guarantees. |
| {doc}`19_STYLE` | Writing any new code or docstring. |
| {doc}`20_PACKAGING` | Adding a dependency, cutting a release, deprecating a symbol. |

```{note}
Numbers 18 (roadmap) lives in {doc}`/reference/roadmap`. There is no
spec 18 in this folder.
```
