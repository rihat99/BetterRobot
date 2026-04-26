# 09 · Human-body extension lane (anatomical joints, muscles, SMPL)

★★ **Structural.** Reserves clean integration seams for human-body
modelling — SMPL/SMPL-X loaders, anatomical knees, muscle force —
without polluting the core robotics layers. References
[`better_human/`](../../../../better_human/) and
[04 §6 SMPL-like body as motivation](../../design/04_PARSERS.md).

## Problem

The repository has a sibling project, `better_human/`, that hosts
SMPL-family body models. The architectural choice today is correct:
*better_human is a separate package; BetterRobot doesn't import from
it*. The intent is that BetterRobot's data model is **expressive
enough** that an SMPL-like body can be constructed via the
programmatic builder.

The non-trivial parts of human-body modelling that don't fit the
standard rigid-multibody assumptions:

1. **Anatomical joints**. A real knee is not a `JointRevoluteUnaligned`;
   it has translation coupled to flexion via a spline (OpenSim's
   `CustomJoint` + `SpatialTransform`). URDF cannot describe this.
2. **Muscle force**. A muscle is a non-linear actuator parameterised
   by activation, fiber length, and fiber velocity (DeGrooteFregly
   2016 is the canonical differentiable model). Joint torque is the
   sum over muscle moment arms.
3. **Subject-specific scaling**. A generic skeleton + bone-length
   scale factors → a subject-specific model with conserved mass
   distribution and proportional muscle lengths.
4. **SMPL/SMPL-X loaders**. Read SMPL `.pkl`/`.npz`, build a
   23-spherical-joint model with shape and pose blendshapes.

Without explicit seams for these, the choices are:

- Build them inside `better_human/` and force users to glue two
  packages. Loses the SMPL-like body benchmark in
  [04 §6](../../design/04_PARSERS.md).
- Build them inside BetterRobot's core. Bloats the core and
  imports `chumpy` / SMPL data into `better_robot.io`.
- Don't build them. Lose a major v2 use case.

The right answer is none of those. It is to **document the seams
clearly enough that a future `better_robot[human]` extra is a clean
addition**.

## The proposal

### 9.A Anatomical joints fit the existing extension seam

[15_EXTENSION.md §2](../../conventions/15_EXTENSION.md) already
specifies how to add a custom `JointModel`. A coupled joint (knee
with spline-driven translation) is exactly that:

```python
# better_robot_human/joints/coupled.py
from better_robot.data_model.joint_models.base import JointModel
import torch

class JointCoupled(JointModel):
    """A 1-DOF revolute joint whose translation is coupled to its
    rotation via a piecewise-linear spline.

    Storage:
        axis : (3,)
        knots_q : (K,)              # rotation angles
        knots_t : (K, 3)             # translation at each knot
    """
    nq = 1
    nv = 1
    kind = "revolute_coupled"

    def joint_transform(self, q): ...        # SE3 7-vector
    def joint_motion_subspace(self, q): ...  # (..., 6, 1)
    def integrate(self, q, v): ...
    def difference(self, q0, q1): ...
    def random_configuration(self, rng, lo, hi): ...
    def neutral(self): ...
```

The contract is unchanged from
[02_DATA_MODEL §4](../../design/02_DATA_MODEL.md). Anatomical joints
slot in via the `Model.joint_models` tuple and the FK loop is
unchanged.

**No core changes required.** `better_robot[human]` ships its
anatomical-joint module and the user attaches it to a model via
`ModelBuilder.add_joint(name=..., kind=JointCoupled(...))`.

### 9.B Muscles need a new layer that does not exist yet

Muscles are not joints; they're actuators. Joint torque is the sum
over muscle pulls × moment arms. The right home is a new sibling
to `dynamics/`:

```
src/better_robot/
└── (existing modules)

src/better_robot_human/
├── joints/
│   └── coupled.py
├── muscles/
│   ├── base.py            # Muscle Protocol
│   ├── degroote_fregly.py # DeGrooteFregly2016
│   └── moment_arms.py
├── scaling.py             # subject scaling
└── parsers/
    ├── smpl.py            # SMPL .npz loader → IRModel
    └── osim.py            # OpenSim .osim parser (later)
```

The `Muscle` Protocol is shaped to compose with `tasks/`:

```python
# src/better_robot_human/muscles/base.py
from typing import Protocol
import torch

class Muscle(Protocol):
    """A scalar-output force generator parameterised by joint state."""
    name: str
    nu: int                                 # control dim (typically 1: activation)

    def compute_force(
        self,
        q: torch.Tensor,                    # (B..., nq)
        v: torch.Tensor,                    # (B..., nv)
        u: torch.Tensor,                    # (B..., nu)
    ) -> torch.Tensor:                      # (B..., 1) scalar tensile force

    def moment_arm(
        self,
        model: "Model",
        q: torch.Tensor,
    ) -> torch.Tensor:                      # (B..., nv)
```

The dispatch into `rnea`/`aba` is via an additive torque term:
`tau += sum(muscle.compute_force(q,v,u) * muscle.moment_arm(model,q))`.

That keeps `dynamics/` agnostic to "muscles" — it just sees an
external joint-space torque. Same way `fext` works in
[06 §2](../../design/06_DYNAMICS.md).

### 9.C Subject-specific scaling lives in `better_human` (or its successor)

`scale_model(generic_model, marker_data, measurements,
preserve_mass=True)` returns a new `Model` with adjusted
`joint_placements` and `body_inertias`. This is a transformation
*on* a `Model` and should live next to the loaders. It does not
require any change to the core `Model` schema.

### 9.D SMPL/SMPL-X parsers emit `IRModel`

[04_PARSERS.md §6](../../design/04_PARSERS.md) defines the
programmatic builder. The SMPL loader is one more parser:

```python
# src/better_robot_human/parsers/smpl.py
from better_robot.io import register_parser, IRModel, ModelBuilder

@register_parser(suffix=".smpl_npz")
def parse_smpl(source) -> IRModel:
    """Load an SMPL .npz, build an IRModel with a free-flyer pelvis
    and 23 spherical joints. Shape parameters fix bone lengths;
    pose parameters become initial q."""
    data = np.load(source)
    b = ModelBuilder("smpl_body")
    pelvis = b.add_body("pelvis", mass=...)
    b.add_joint("root", kind="free_flyer", parent="world", child=pelvis)
    for i, bone_name in enumerate(SMPL_BONES):
        ...
        b.add_joint(name=bone_name, kind="spherical", parent=parent, child=child)
    return b.finalize()
```

Then `better_robot.load("body.smpl_npz")` works. No core change.

### 9.E The optional extra — declare the seam, defer the dependency

The `better_robot_human` package does not exist on PyPI yet. An
earlier draft of this proposal added a real
`better-robot[human]` extra pointing at it; the gpt-plan review
correctly flagged this as adding a real dependency on an
unpublished package.

The phased approach:

**Phase 1 (now).** Document the seam, do not declare a
dependency. `pyproject.toml` carries no `[human]` extra. Inside
the repo we add:

- A placeholder test in `tests/io/test_smpl_like_skeleton.py`
  that constructs an SMPL-skeleton-topology IRModel via the
  programmatic builder and runs FK on it. No SMPL data, no
  `chumpy` import. This proves the seam works.
- A docstring at `better_robot.io.register_parser` explaining
  how an external `better_robot_human` package would register
  `.smpl_npz` and `.osim` parsers.
- A line in [15_EXTENSION.md](../../conventions/15_EXTENSION.md)
  noting the planned extras package and its registration pattern.

**Phase 2 (when `better_robot_human` is published).** Add the
extra:

```toml
# pyproject.toml
[project.optional-dependencies]
human = [
    "better_robot_human>=0.1.0",       # the human extras package
    "chumpy>=0.78",                     # only needed for legacy SMPL .pkl
]
```

`pip install better-robot[human]` pulls in the package; nothing in
core BetterRobot imports it.

`better_robot_human` (in Phase 2) ships:

- `JointCoupled` and any other anatomical joint kinds.
- `Muscle` base classes and `DeGrooteFregly2016`.
- `parse_smpl`, `parse_osim`.
- Scaling helpers.
- Tests against a small SMPL fixture.

It registers itself on import:

```python
# better_robot_human/__init__.py
from better_robot.io import register_parser
from .parsers.smpl import parse_smpl
from .parsers.osim import parse_osim

register_parser(".smpl_npz", parse_smpl)
register_parser(".osim", parse_osim)
```

This is the [15 §11 registry](../../conventions/15_EXTENSION.md)
pattern, not auto-discovery via entry points.

### 9.F What the core *adds* to host this

The change in core BetterRobot to enable the above is intentionally
minimal:

1. **Document the seam in
   [15_EXTENSION.md](../../conventions/15_EXTENSION.md)** — add a
   sub-section "Adding actuators (e.g. muscles)" that pins the
   `Muscle` Protocol and the additive-torque dispatch.
2. **Make `dynamics.rnea(..., fext=None)` future-proof** — the
   parameter is already in the signature
   ([06 §2](../../design/06_DYNAMICS.md)); document that
   `better_robot_human` uses it.
3. **No core import of `chumpy`, SMPL, or OpenSim** — confirmed by
   the layer DAG test.

That's it. The seam already exists; this proposal documents it and
ensures it does not get accidentally closed.

## Tradeoffs

| For | Against |
|-----|---------|
| Two clean codebases: BetterRobot is general; `better_robot_human` is the domain layer. | Users have to know about the extras. Mitigation: README has a "Domains: humans" pointer. |
| `JointCoupled` is the canonical extension test — if it works for knees, it works for any custom kinematic. | A small risk that the `JointModel` Protocol is missing a hook needed for muscle moment arms. Mitigation: Muscle `moment_arm(model, q)` is independent of `JointModel`; the protocol is sufficient as-is. |
| The OpenSim DeGrooteFregly2016 muscle is the only differentiable model with a published reference; we ship exactly that and not more. | We do not ship contact-rich biomechanics in v1. That's correct — leave to v2 / domain layer. |

## Acceptance criteria

- [15_EXTENSION.md](../../conventions/15_EXTENSION.md) has a new
  section "Adding actuators (e.g. muscles)" with the `Muscle`
  Protocol and the dispatch contract.
- The 04 SMPL-like body example continues to work with the new
  `Trajectory` and `Inertia` types from Proposals 5 and 8.
- *Phase 1:* `pyproject.toml` carries **no** `[human]` extra. The
  seam is documented in
  [15_EXTENSION.md](../../conventions/15_EXTENSION.md) and a placeholder
  test demonstrates the SMPL-skeleton topology can be built via the
  programmatic builder.
- *Phase 2 (when `better_robot_human` is published):* `pyproject.toml`
  declares the optional `[human]` extra pointing at the published
  package.
- A regression test in `tests/io/test_smpl_like.py` builds an
  SMPL-skeleton-topology IRModel via the programmatic builder and
  runs FK on it — no `chumpy` or SMPL data import.
- The layer DAG test ensures no `core/` module imports `chumpy`,
  `smpl`, `osim`, `opensim`, or any human-domain symbol.

## Cross-references

- [04 §6](../../design/04_PARSERS.md) — SMPL-like body as the
  expressiveness test.
- [06 §2](../../design/06_DYNAMICS.md) — `fext` parameter that muscle
  forces use.
- [15 §2](../../conventions/15_EXTENSION.md) — `JointModel` Protocol
  shape.
- [`better_human/README.md`](../../../../better_human/README.md) — the
  separate package this proposal explicitly does not absorb.
