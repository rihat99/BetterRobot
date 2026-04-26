"""``ModelBuilder`` — programmatic fluent builder for an ``IRModel``.

mjlab's best idea: the robot is produced by a Python function, not an XML
file.

The builder exposes **named per-kind helpers** (``add_revolute_x``,
``add_prismatic``, ``add_free_flyer_root``, …) instead of a stringly-typed
``kind="revolute_z"`` switch. The catch-all :meth:`ModelBuilder.add_joint`
accepts a :class:`JointModel` instance only — passing a string raises
:class:`TypeError` pointing at the right named helper.

See ``docs/concepts/parsers_and_ir.md §6``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..ir import IRBody, IRFrame, IRGeom, IRJoint, IRModel

if TYPE_CHECKING:
    from ...data_model.joint_models.base import JointModel


# Named-kind names that the builder forbids passing as a string. The
# message names the helper to use instead.
_LEGACY_KIND_HELPER: dict[str, str] = {
    "revolute": "add_revolute (or add_revolute_x/y/z)",
    "revolute_x": "add_revolute_x",
    "revolute_y": "add_revolute_y",
    "revolute_z": "add_revolute_z",
    "continuous": "add_revolute (kind='continuous' for unbounded)",
    "prismatic": "add_prismatic (or add_prismatic_x/y/z)",
    "prismatic_x": "add_prismatic_x",
    "prismatic_y": "add_prismatic_y",
    "prismatic_z": "add_prismatic_z",
    "spherical": "add_spherical",
    "ball": "add_spherical",
    "planar": "add_planar",
    "helical": "add_helical",
    "free_flyer": "add_free_flyer_root",
    "free": "add_free_flyer_root",
    "floating": "add_free_flyer_root",
    "fixed": "add_fixed",
    "translation": "add_prismatic (translation kind)",
    "world": "add_fixed",
}


_IDENTITY_SE3 = torch.tensor([0., 0., 0., 0., 0., 0., 1.])


def _axis_x() -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0])


def _axis_y() -> torch.Tensor:
    return torch.tensor([0.0, 1.0, 0.0])


def _axis_z() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 1.0])


class ModelBuilder:
    """Fluent, imperative builder that emits an ``IRModel``.

    Example
    -------
    >>> b = ModelBuilder("my_arm")
    >>> base = b.add_body("base", mass=0.0)
    >>> link1 = b.add_body("link1", mass=1.2)
    >>> b.add_revolute_z("j1", parent=base, child=link1,
    ...                  origin=torch.zeros(7), lower=-3.14, upper=3.14)
    >>> ir = b.finalize()
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._bodies: list[IRBody] = []
        self._body_names: set[str] = set()
        self._joints: list[IRJoint] = []
        self._joint_names: set[str] = set()
        self._frames: list[IRFrame] = []

    # ── bodies / frames / geoms ──────────────────────────────────────

    def add_body(
        self,
        name: str,
        *,
        mass: float = 0.0,
        com: torch.Tensor | None = None,
        inertia: torch.Tensor | None = None,
    ) -> str:
        """Add a body and return its name."""
        if name in self._body_names:
            raise ValueError(f"Body {name!r} already exists")
        self._body_names.add(name)
        self._bodies.append(IRBody(
            name=name,
            mass=mass,
            com=com if com is not None else torch.zeros(3),
            inertia=inertia if inertia is not None else torch.zeros(3, 3),
        ))
        return name

    def add_frame(
        self,
        name: str,
        *,
        parent_body: str,
        placement: torch.Tensor,
        frame_type: str = "op",
    ) -> str:
        """Add a named operational frame and return its name."""
        self._frames.append(IRFrame(
            name=name,
            parent_body=parent_body,
            placement=placement,
            frame_type=frame_type,
        ))
        return name

    def add_collision_geom(
        self,
        body: str,
        kind: str,
        params: dict,
        origin: torch.Tensor,
    ) -> None:
        """Attach a collision primitive to a body."""
        for b in self._bodies:
            if b.name == body:
                b.collision_geoms.append(IRGeom(kind=kind, params=params, origin=origin))
                return
        raise ValueError(f"Body {body!r} not found")

    # ── private joint helper ─────────────────────────────────────────

    def _push_joint(
        self,
        name: str,
        *,
        kind: str,
        parent: str,
        child: str,
        origin: torch.Tensor,
        axis: torch.Tensor | None,
        lower: float | None,
        upper: float | None,
        velocity_limit: float | None,
        effort_limit: float | None,
        mimic_source: str | None,
        mimic_multiplier: float,
        mimic_offset: float,
    ) -> str:
        if name in self._joint_names:
            raise ValueError(f"Joint {name!r} already exists")
        self._joint_names.add(name)
        self._joints.append(IRJoint(
            name=name,
            parent_body=parent,
            child_body=child,
            kind=kind,
            axis=axis,
            origin=origin,
            lower=lower,
            upper=upper,
            velocity_limit=velocity_limit,
            effort_limit=effort_limit,
            mimic_source=mimic_source,
            mimic_multiplier=mimic_multiplier,
            mimic_offset=mimic_offset,
        ))
        return name

    # ── named revolute helpers ───────────────────────────────────────

    def add_revolute(
        self,
        name: str,
        *,
        parent: str,
        child: str,
        axis: torch.Tensor,
        origin: torch.Tensor | None = None,
        lower: float | None = None,
        upper: float | None = None,
        velocity_limit: float | None = None,
        effort_limit: float | None = None,
        unbounded: bool = False,
        mimic_source: str | None = None,
        mimic_multiplier: float = 1.0,
        mimic_offset: float = 0.0,
    ) -> str:
        """Generic 1-DoF revolute joint with explicit ``axis``.

        ``unbounded=True`` selects URDF-``continuous`` semantics
        (no lower/upper) and produces a
        :class:`~better_robot.data_model.joint_models.JointRevoluteUnbounded`.
        """
        return self._push_joint(
            name,
            kind="continuous" if unbounded else "revolute",
            parent=parent, child=child,
            origin=origin if origin is not None else _IDENTITY_SE3.clone(),
            axis=axis, lower=lower, upper=upper,
            velocity_limit=velocity_limit, effort_limit=effort_limit,
            mimic_source=mimic_source,
            mimic_multiplier=mimic_multiplier,
            mimic_offset=mimic_offset,
        )

    def add_revolute_x(self, name: str, **kwargs) -> str:
        """Revolute joint about the X axis."""
        return self.add_revolute(name, axis=_axis_x(), **kwargs)

    def add_revolute_y(self, name: str, **kwargs) -> str:
        """Revolute joint about the Y axis."""
        return self.add_revolute(name, axis=_axis_y(), **kwargs)

    def add_revolute_z(self, name: str, **kwargs) -> str:
        """Revolute joint about the Z axis."""
        return self.add_revolute(name, axis=_axis_z(), **kwargs)

    # ── named prismatic helpers ──────────────────────────────────────

    def add_prismatic(
        self,
        name: str,
        *,
        parent: str,
        child: str,
        axis: torch.Tensor,
        origin: torch.Tensor | None = None,
        lower: float | None = None,
        upper: float | None = None,
        velocity_limit: float | None = None,
        effort_limit: float | None = None,
    ) -> str:
        """Generic 1-DoF prismatic joint with explicit ``axis``."""
        return self._push_joint(
            name, kind="prismatic", parent=parent, child=child,
            origin=origin if origin is not None else _IDENTITY_SE3.clone(),
            axis=axis, lower=lower, upper=upper,
            velocity_limit=velocity_limit, effort_limit=effort_limit,
            mimic_source=None, mimic_multiplier=1.0, mimic_offset=0.0,
        )

    def add_prismatic_x(self, name: str, **kwargs) -> str:
        return self.add_prismatic(name, axis=_axis_x(), **kwargs)

    def add_prismatic_y(self, name: str, **kwargs) -> str:
        return self.add_prismatic(name, axis=_axis_y(), **kwargs)

    def add_prismatic_z(self, name: str, **kwargs) -> str:
        return self.add_prismatic(name, axis=_axis_z(), **kwargs)

    # ── ball / planar / helical / free / fixed ───────────────────────

    def add_spherical(
        self,
        name: str,
        *,
        parent: str,
        child: str,
        origin: torch.Tensor | None = None,
    ) -> str:
        """Spherical (ball) joint — 3 rotational DoF, SO(3) ``q``."""
        return self._push_joint(
            name, kind="spherical", parent=parent, child=child,
            origin=origin if origin is not None else _IDENTITY_SE3.clone(),
            axis=None, lower=None, upper=None,
            velocity_limit=None, effort_limit=None,
            mimic_source=None, mimic_multiplier=1.0, mimic_offset=0.0,
        )

    def add_planar(
        self,
        name: str,
        *,
        parent: str,
        child: str,
        origin: torch.Tensor | None = None,
    ) -> str:
        """Planar joint (2-DoF translation + 1-DoF in-plane rotation)."""
        return self._push_joint(
            name, kind="planar", parent=parent, child=child,
            origin=origin if origin is not None else _IDENTITY_SE3.clone(),
            axis=None, lower=None, upper=None,
            velocity_limit=None, effort_limit=None,
            mimic_source=None, mimic_multiplier=1.0, mimic_offset=0.0,
        )

    def add_helical(
        self,
        name: str,
        *,
        parent: str,
        child: str,
        axis: torch.Tensor,
        pitch: float,
        origin: torch.Tensor | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> str:
        """Helical / screw joint along ``axis`` with linear/angular ratio ``pitch``."""
        del pitch  # parsed downstream from joint metadata once helical lands
        return self._push_joint(
            name, kind="helical", parent=parent, child=child,
            origin=origin if origin is not None else _IDENTITY_SE3.clone(),
            axis=axis, lower=lower, upper=upper,
            velocity_limit=None, effort_limit=None,
            mimic_source=None, mimic_multiplier=1.0, mimic_offset=0.0,
        )

    def add_free_flyer_root(
        self,
        name: str = "free_flyer",
        *,
        child: str,
        origin: torch.Tensor | None = None,
    ) -> str:
        """Free-flyer (6-DoF SE(3)) root joint connecting ``world → child``."""
        return self._push_joint(
            name, kind="free_flyer", parent="world", child=child,
            origin=origin if origin is not None else _IDENTITY_SE3.clone(),
            axis=None, lower=None, upper=None,
            velocity_limit=None, effort_limit=None,
            mimic_source=None, mimic_multiplier=1.0, mimic_offset=0.0,
        )

    def add_fixed(
        self,
        name: str,
        *,
        parent: str,
        child: str,
        origin: torch.Tensor | None = None,
    ) -> str:
        """Rigid (fixed) attachment between two bodies."""
        return self._push_joint(
            name, kind="fixed", parent=parent, child=child,
            origin=origin if origin is not None else _IDENTITY_SE3.clone(),
            axis=None, lower=None, upper=None,
            velocity_limit=None, effort_limit=None,
            mimic_source=None, mimic_multiplier=1.0, mimic_offset=0.0,
        )

    # ── catch-all ────────────────────────────────────────────────────

    def add_joint(
        self,
        name: str,
        *,
        kind=None,
        parent: str,
        child: str,
        origin: torch.Tensor,
        axis: torch.Tensor | None = None,
        lower: float | None = None,
        upper: float | None = None,
        velocity_limit: float | None = None,
        effort_limit: float | None = None,
        mimic_source: str | None = None,
        mimic_multiplier: float = 1.0,
        mimic_offset: float = 0.0,
    ) -> str:
        """Catch-all: register a joint via a :class:`JointModel` instance.

        Strings are not accepted; passing one raises :class:`TypeError`
        with a pointer at the named helper for that kind.
        """
        if isinstance(kind, str):
            helper = _LEGACY_KIND_HELPER.get(kind, "the matching add_<kind> helper")
            raise TypeError(
                f"ModelBuilder.add_joint(kind={kind!r}, ...) is not supported. "
                f"Use {helper!s} instead — see docs/concepts/parsers_and_ir.md §6."
            )
        if kind is None:
            raise TypeError(
                "ModelBuilder.add_joint requires kind=<JointModel instance>; "
                "use one of the named add_<kind> helpers instead."
            )
        # Read the joint kind off the JointModel instance.
        kind_str = getattr(kind, "kind", None)
        if not isinstance(kind_str, str):
            raise TypeError(
                f"kind={kind!r} does not look like a JointModel "
                f"(missing string `.kind` attribute)."
            )
        return self._push_joint(
            name, kind=kind_str, parent=parent, child=child,
            origin=origin, axis=axis,
            lower=lower, upper=upper,
            velocity_limit=velocity_limit, effort_limit=effort_limit,
            mimic_source=mimic_source,
            mimic_multiplier=mimic_multiplier, mimic_offset=mimic_offset,
        )

    # ── finalize ─────────────────────────────────────────────────────

    def finalize(self) -> IRModel:
        """Validate and return the built ``IRModel``."""
        # Exclude joints whose parent is the "world" sentinel — those bodies
        # connect to the universe and are not children of any real body.
        child_bodies = {j.child_body for j in self._joints if j.parent_body != "world"}
        root_candidates = [b.name for b in self._bodies if b.name not in child_bodies]
        if len(root_candidates) != 1:
            raise ValueError(
                f"Expected exactly 1 root body (body with no incoming joint), "
                f"found {len(root_candidates)}: {root_candidates}"
            )
        return IRModel(
            name=self._name,
            bodies=list(self._bodies),
            joints=list(self._joints),
            frames=list(self._frames),
            root_body=root_candidates[0],
        )
