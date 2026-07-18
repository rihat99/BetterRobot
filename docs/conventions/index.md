# Conventions

These pages collect the rules that keep BetterRobot predictable. They explain
what public inputs must look like, how extension points work, how performance
claims are measured, and what a contribution must prove.

```{toctree}
:maxdepth: 1

naming
contracts
extension
testing
performance
style
packaging
source_and_license
```

## Choose a page

| Page | Read it when you are… |
|---|---|
| {doc}`naming` | choosing a public field, function, or tensor name |
| {doc}`contracts` | checking shapes, dtypes, errors, gradients, or state ownership |
| {doc}`extension` | adding a residual, joint, solver, parser, viewer mode, or asset resolver |
| {doc}`testing` | deciding which checks and numerical comparisons a change needs |
| {doc}`performance` | measuring a hot path or changing compile, batching, or allocation behavior |
| {doc}`style` | writing or reviewing source code and docstrings |
| {doc}`packaging` | changing dependencies, extras, versions, or releases |
| {doc}`source_and_license` | adapting material from another project |

The code and tests are the final authority. If a page disagrees with a public
signature or an executable contract, fix the page with the code change.
