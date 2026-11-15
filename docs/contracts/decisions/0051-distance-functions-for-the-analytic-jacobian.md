# 0051 A `distance_fn` reads its own centre and runs under `torch.func`

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-15

## Context
C-EXT-8 makes `distance_fn(centers: (B..., S, 3), radii: (S,)) -> (B..., S)` or `(B..., S, K)` the seam for any
obstacle shape. Until 0.3.0 it asked only for a signed distance differentiable in `centers`, and
`terms.sphere_clearance` took its Jacobian by autodiff through forward kinematics. In 0.3.0 the Jacobian is analytic:
the distance gradient at each sphere centre times `kinematics.point_jacobian`. One `torch.func.vjp` of `distance_fn`
alone gives that gradient (seeded with ones for all `S` entries, `torch.func.vmap` over `K` obstacles), which
linearises 2-16x faster than autodiff (Panda, float64, nine spheres, one CPU core, batch 1 to 1024). The pullback
assumes two things the contract did not ask for:

- entry `s` depends on centre `s` only. A function whose entry reads other centres, such as a distance to a shape
  the spheres span together, gets a wrong Jacobian with no error (`optim.check_jacobian` error 7.9 for the distance
  of each sphere to the mean of all centres);
- `torch.func` can transform the function. An old-style `torch.autograd.Function` (no `setup_context`), the usual
  form of an external SDF or mesh kernel, raises torch's `RuntimeError` at the first linearisation.

The triggers in `docs/contracts/00-index.md` do not name a tightened rule, but a narrower seam is the owner's call.

## Decision
We will require both of a `distance_fn`: entry `s` of its result depends on centre `s` only, and `torch.func` can
transform it. `sphere_clearance` keeps the analytic Jacobian as its only route. A function that breaks either
requirement is registered as `optim.TermFn(fn=term.fn, reads=term.reads)`: the same residual, differentiated by
autodiff, with no new public name.

## Example
```python { .no-run }
# meets both: a closure over a collision primitive, each sphere against the table
problem.residual("table", terms.sphere_clearance(spheres, to_table, margin=0.02))
# an external kernel written as an old-style autograd.Function takes the autodiff route
term = terms.sphere_clearance(spheres, external_kernel, margin=0.02)
problem.residual("table", optim.TermFn(fn=term.fn, reads=term.reads))
```

## Consequences
Every shipped shape (sphere, half-space, box, capsule, sampled SDF) meets both requirements, and a caller's function
that meets them gets the analytic Jacobian without code of its own. A function that breaks the first requirement
linearises without an error but with a wrong Jacobian, which `optim.check_jacobian` reports; one that breaks the
second fails at the first linearisation. The `TermFn` registration costs that one term the autodiff time it had in
0.2.

## Alternatives
An autodiff keyword on `sphere_clearance`: a public option for what `TermFn(fn=, reads=)` already does. Catch
torch's error and fall back to autodiff: a silent fallback that hides real errors and cannot detect the first
requirement. One pullback per sphere, so a coupled function is exact: `S` times the cost for every shape, for a case
no shipped shape has. Closed-form gradients per shipped shape: a second seam (a distance function that returns its
gradient) beside `distance_fn`.

## Compatibility
A `distance_fn` that worked with 0.2 and breaks a requirement needs the `TermFn` registration (a row in
`docs/reference/migration-0.3.0.md`). No public name or signature changes.

## Enforcement
`tests/collision/test_collision.py::test_clearance_jacobians_cover_every_shape_and_the_closest_pairs` (every shipped
shape against autodiff); `tests/collision/test_collision.py::test_a_distance_fn_torch_func_cannot_transform_takes_the_autodiff_route`
(the `TermFn` registration); `optim.check_jacobian` and `[review]` for a caller's function.

## Open questions
- Accept the two requirements on `distance_fn`, or keep an explicit autodiff option on `sphere_clearance`?
