"""``ModelBuilder`` — programmatic fluent builder for an ``IRModel``.

mjlab's best idea: the robot is produced by a Python function, not an XML
file.

See ``docs/04_PARSERS.md §6``.
"""

from __future__ import annotations

import torch

from ..ir import IRBody, IRFrame, IRGeom, IRJoint, IRModel


class ModelBuilder:
    """Fluent, imperative builder that emits an ``IRModel``.

    Example
    -------
    >>> b = ModelBuilder("my_arm")
    >>> base = b.add_body("base", mass=0.0)
    >>> link1 = b.add_body("link1", mass=1.2)
    >>> b.add_joint("j1", kind="revolute_z", parent=base, child=link1,
    ...             origin=torch.zeros(7), lower=-3.14, upper=3.14)
    >>> ir = b.finalize()
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._bodies: list[IRBody] = []
        self._body_names: set[str] = set()
        self._joints: list[IRJoint] = []
        self._joint_names: set[str] = set()
        self._frames: list[IRFrame] = []

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

    def add_joint(
        self,
        name: str,
        *,
        kind: str,
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
        """Add a joint and return its name."""
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
