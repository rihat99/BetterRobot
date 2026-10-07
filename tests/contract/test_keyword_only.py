"""C-API-2: public callables take at most two positional parameters.

Every callable listed in a public ``__all__`` (top-level package and each
subpackage) is inspected; parameters after the second must be keyword-only.
``KEYWORD_ONLY_ALLOWLIST`` names the current violations so the ledger in
``agent_notes`` can shrink one entry at a time; adding a name is a contract
decision, not a convenience.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

import better_robot

PACKAGES = [
    "better_robot",
    "better_robot.lie",
    "better_robot.spatial",
    "better_robot.data_model",
    "better_robot.kinematics",
    "better_robot.dynamics",
    "better_robot.collision",
    "better_robot.io",
    "better_robot.optim",
    "better_robot.terms",
    "better_robot.viewer",
]

MAX_POSITIONAL = 2

# Current violations (package.name -> positional count). Shrink, never grow.
KEYWORD_ONLY_ALLOWLIST: dict[str, int] = {
    "better_robot.ModelBuilder.add_collision_geom": 4,  # io/parsers/programmatic.py:129
    "better_robot.frame_jacobian": 3,  # kinematics/state.py:183
    "better_robot.frame_jacobian_time_variation": 4,  # kinematics/state.py:222
    "better_robot.joint_jacobian": 3,  # kinematics/state.py:206
    "better_robot.joint_jacobian_time_variation": 4,  # kinematics/state.py:241
    "better_robot.inverse_dynamics": 4,  # dynamics/state.py:55
    "better_robot.forward_dynamics": 4,  # dynamics/state.py:83
    "better_robot.centroidal_momentum": 3,  # dynamics/state.py:109
    "better_robot.lie.umeyama": 3,  # lie/alignment.py:67
    "better_robot.kinematics.frame_jacobian": 3,  # kinematics/state.py:183
    "better_robot.kinematics.frame_jacobian_time_variation": 4,  # kinematics/state.py:222
    "better_robot.kinematics.joint_jacobian": 3,  # kinematics/state.py:206
    "better_robot.kinematics.joint_jacobian_time_variation": 4,  # kinematics/state.py:241
    "better_robot.dynamics.inverse_dynamics": 4,  # dynamics/state.py:55
    "better_robot.dynamics.forward_dynamics": 4,  # dynamics/state.py:83
    "better_robot.dynamics.centroidal_momentum": 3,  # dynamics/state.py:109
    "better_robot.collision.self_distances": 3,  # collision/robot_spheres.py:90
    "better_robot.collision.sphere_box": 5,  # collision/distance.py:47
    "better_robot.collision.sphere_capsule": 5,  # collision/distance.py:27
    "better_robot.collision.sphere_halfspace": 4,  # collision/distance.py:40
    "better_robot.collision.sphere_sdf": 5,  # collision/distance.py:63
    "better_robot.collision.sphere_sphere": 4,  # collision/distance.py:20
    "better_robot.io.ModelBuilder.add_collision_geom": 4,  # io/parsers/programmatic.py:129
    "better_robot.optim.Euclidean.tangent_bounds": 3,  # optim/manifold.py:49
    "better_robot.optim.Manifold.tangent_bounds": 3,  # optim/manifold.py:30
    "better_robot.optim.run": 3,  # optim/phases.py:48
    "better_robot.terms.RobotManifold.tangent_bounds": 3,  # terms/robot.py:40
    "better_robot.terms.ik_problem": 3,  # terms/ik.py:15
    "better_robot.terms.random_start": 3,  # terms/ik.py:47
    "better_robot.terms.solve_ik": 3,  # terms/ik.py:58
}


def _public_callables() -> list[tuple[str, object]]:
    found: list[tuple[str, object]] = []
    for package in PACKAGES:
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if inspect.isfunction(obj):
                found.append((f"{package}.{name}", obj))
            elif inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if inspect.isfunction(member) and not attr.startswith("_"):
                        found.append((f"{package}.{name}.{attr}", member))
    return found


def _positional_count(fn: object) -> int:
    params = list(inspect.signature(fn).parameters.values())
    if params and params[0].name in {"self", "cls"}:
        params = params[1:]
    return sum(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params
    )


CALLABLES = _public_callables()


@pytest.mark.parametrize("qualname,fn", CALLABLES, ids=[q for q, _ in CALLABLES])
def test_public_callable_is_keyword_only_after_second_argument(
    qualname: str, fn: object
) -> None:
    count = _positional_count(fn)
    allowed = KEYWORD_ONLY_ALLOWLIST.get(qualname, MAX_POSITIONAL)
    assert count <= allowed, (
        f"{qualname} takes {count} positional parameters (limit {allowed}); "
        "make the extra ones keyword-only or add it to KEYWORD_ONLY_ALLOWLIST."
    )


def test_allowlist_has_no_stale_entries() -> None:
    present = {q: fn for q, fn in CALLABLES}
    stale = [
        q
        for q, limit in KEYWORD_ONLY_ALLOWLIST.items()
        if q not in present or _positional_count(present[q]) <= MAX_POSITIONAL
    ]
    assert not stale, f"remove fixed entries from KEYWORD_ONLY_ALLOWLIST: {stale}"


def test_top_level_package_exports_exist() -> None:
    assert better_robot.__all__
