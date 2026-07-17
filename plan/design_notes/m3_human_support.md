# M3 human support decisions

T3.6 implements `SwingTwistLimitResidual`, `JointRotationPrior`, and batched
differentiable `Inertia.from_mesh`. The residuals intentionally omit analytic
blocks: swing/twist is singular at pure-pi swing and piecewise at limit/wrap
cuts, while a rotation prior's exact derivative contains Lie right-Jacobian
factors. Named-block problems therefore use tangent AD and the legacy lane uses
its explicit finite-difference fallback; neither residual mislabels a
small-angle approximation as exact.

For one oriented mesh face `(a, b, c)`, integration uses
`V = dot(a, cross(b, c)) / 6`, first moment `V*(a+b+c)/4`, and second moment
`V/20 * ((a+b+c)(a+b+c)^T + aa^T + bb^T + cc^T)`. A global orientation fold
makes complete face reversal physically invariant before converting the raw
second moment to inertia at the COM.

`inertia_from_vertex_parts` is deferred. The proposed
`faces_per_body | vertex_to_body` union has no second in-tree caller, does not
define how boundary triangles are assigned, and leaves joint-relative frame
semantics ambiguous. The consumer repositories are explicitly out of scope,
so they cannot supply the missing two-caller evidence in M3.
