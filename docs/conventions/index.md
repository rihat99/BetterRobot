# Conventions

Cross-cutting normative specs — everything the library expects at its
boundaries. Where {doc}`/concepts/index` is the *what*, these are the
*how*: the rules that the test suite enforces and that contributors
have to keep in mind.

```{toctree}
:maxdepth: 1

naming
performance
extension
testing
contracts
engineering
source_and_license
style
packaging
```

## When to consult which doc

| Spec | Read when |
|------|-----------|
| {doc}`naming` | Adding any new identifier, or staring at the Pinocchio → BetterRobot rename table. |
| {doc}`performance` | Touching hot paths, debugging compile / latency / memory. |
| {doc}`extension` | Plugging in a new residual / joint / solver / backend / parameterisation. |
| {doc}`testing` | Adding tests, or promoting a benchmark from advisory to blocking. |
| {doc}`contracts` | Touching the public API, raising a new exception, or changing numerical guarantees. |
| {doc}`engineering` | Deciding dtype, quaternion, concurrency, serialization, differentiation, or compile behavior. |
| {doc}`source_and_license` | Apache-2.0 decision record and external-source ledger rules. |
| {doc}`style` | Writing any new code or docstring. |
| {doc}`packaging` | Adding a dependency, cutting a release, deprecating a symbol. |
