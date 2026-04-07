"""RobotCollision: capsule and sphere decomposition of robot links."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import torch

if TYPE_CHECKING:
    import yourdfpy
    import trimesh as tm

from ...models.robot_model import RobotModel
from ...math.so3 import so3_rotation_matrix
from .primitives import CollGeom, Sphere, Capsule
from .distance import compute_distance
from . import _utils

__all__ = [
    "RobotCollision",
]


@dataclass
class RobotCollision:
    """Robot collision model supporting both capsule and sphere decomposition.

    **Capsule mode** (recommended): One capsule per link, auto-generated from
    URDF collision geometry via :meth:`from_urdf`, or manually specified via
    :meth:`from_capsule_decomposition`. Capsules approximate the full body of
    each link, not just joints.

    **Sphere mode** (legacy): Multiple spheres per link, manually specified via
    :meth:`from_sphere_decomposition`.

    In both modes, adjacent-link pairs are excluded from self-collision checking
    and `compute_self_collision_distance` / `compute_world_collision_distance`
    return signed distances for all active pairs.
    """

    _mode: str
    """'capsule' or 'sphere'."""

    # --- Sphere mode fields (None when mode == 'capsule') ---
    _local_centers: Optional[torch.Tensor]
    """(num_spheres, 3) sphere centers in link-local frame."""
    _radii: Optional[torch.Tensor]
    """(num_spheres,) sphere radii."""
    _link_indices: Optional[torch.Tensor]
    """(num_spheres,) link index for each sphere."""

    # --- Capsule mode fields (None when mode == 'sphere') ---
    _local_points_a: Optional[torch.Tensor]
    """(num_capsules, 3) first endpoint in link-local frame."""
    _local_points_b: Optional[torch.Tensor]
    """(num_capsules, 3) second endpoint in link-local frame."""
    _capsule_radii: Optional[torch.Tensor]
    """(num_capsules,) capsule radii."""
    _capsule_link_indices: Optional[torch.Tensor]
    """(num_capsules,) link index for each capsule."""

    # --- Shared ---
    _active_pairs_i: tuple[int, ...]
    """First geometry index of each active self-collision pair."""
    _active_pairs_j: tuple[int, ...]
    """Second geometry index of each active self-collision pair."""

    # ------------------------------------------------------------------ #
    #  Constructors
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_urdf(
        urdf: "yourdfpy.URDF",
        model: RobotModel,
        ignore_pairs: list[tuple[str, str]] | None = None,
        num_adjacent_levels: int = 2,
        filter_below_rest_dist: float | None = 0.01,
        filter_q: torch.Tensor | None = None,
        filter_base_pose: torch.Tensor | None = None,
    ) -> "RobotCollision":
        """Build a capsule collision model from URDF collision geometry.

        For each link the URDF collision meshes are merged and wrapped with the
        minimum bounding cylinder.  Links with no collision geometry get a
        degenerate zero-radius capsule and are excluded from active pairs.

        Two filters prevent the conservative bounding capsules from creating
        spurious collision costs at normal configurations:

        * **Kinematic neighbourhood filter** (``num_adjacent_levels``): excludes
          all link pairs whose kinematic-chain distance is ≤ this value.  The
          default of 2 removes immediate neighbours *and* skip-one pairs
          (e.g. link0↔link2, link2↔link4), which always overlap due to
          capsule conservatism on a serial arm.

        * **Reference-pose filter** (``filter_below_rest_dist``): after the
          neighbourhood filter, any pair whose capsule-capsule distance at the
          reference configuration is below this threshold (metres) is also
          excluded.  Pairs that penetrate or nearly touch at the reference pose
          are structural overlaps, not genuine self-collision hazards.
          Default 0.01 m.  Set to ``None`` to disable.

        Args:
            urdf: Loaded yourdfpy URDF (collision meshes are loaded if absent).
            model: Corresponding RobotModel (used for link names and adjacency).
            ignore_pairs: Optional list of ``(link_name_a, link_name_b)`` pairs
                to exclude from self-collision checking.
            num_adjacent_levels: Kinematic-chain depth up to which link pairs are
                excluded.  1 = immediate parent-child only; 2 (default) also
                excludes skip-one pairs.
            filter_below_rest_dist: Exclude pairs with distance below this value
                at the reference config.  None disables the filter.
            filter_q: Joint config used for reference-pose filtering.  Defaults
                to ``model.q_default`` (midpoint of limits).  For floating-base
                robots pass the actual rest pose (e.g. standing configuration).
            filter_base_pose: Base pose ``[tx,ty,tz,qx,qy,qz,qw]`` used together
                with ``filter_q`` for floating-base robots.  None = fixed base.

        Returns:
            RobotCollision in capsule mode.
        """
        try:
            import trimesh as tm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "trimesh is required for RobotCollision.from_urdf(). "
                "Install it with: pip install trimesh"
            ) from exc

        # Reload with collision meshes if none are present.
        has_collision = any(link.collisions for link in urdf.link_map.values())
        if not has_collision:
            try:
                import yourdfpy as _yp
                urdf = _yp.URDF(
                    robot=urdf.robot,
                    filename_handler=urdf._filename_handler,
                    load_collision_meshes=True,
                )
            except Exception:
                pass  # proceed with whatever meshes are available

        link_names = model.links.names
        num_links = model.links.num_links

        points_a: list[list[float]] = []
        points_b: list[list[float]] = []
        cap_radii: list[float] = []
        cap_link_indices: list[int] = []

        for link_idx, link_name in enumerate(link_names):
            mesh = RobotCollision._get_trimesh_collision_geometries(urdf, link_name)
            capsule = Capsule.from_trimesh(mesh)
            points_a.append(capsule.point_a.tolist())
            points_b.append(capsule.point_b.tolist())
            cap_radii.append(capsule.radius)
            cap_link_indices.append(link_idx)

        # Kinematic-neighbourhood exclusion set.
        kin_neighbours = _build_kinematic_neighbours(model, num_adjacent_levels)

        # User-specified ignore pairs.
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}
        ignore_set: set[tuple[int, int]] = set()
        if ignore_pairs:
            for na, nb in ignore_pairs:
                if na in link_name_to_idx and nb in link_name_to_idx:
                    ia, ib = link_name_to_idx[na], link_name_to_idx[nb]
                    ignore_set.add((min(ia, ib), max(ia, ib)))

        # First-pass active pairs: exclude degenerate, kinematic neighbours, user pairs.
        idx_i: list[int] = []
        idx_j: list[int] = []
        for i in range(num_links):
            if cap_radii[i] == 0.0:
                continue
            for j in range(i + 1, num_links):
                if cap_radii[j] == 0.0:
                    continue
                pair = (i, j)  # one capsule per link → capsule idx == link idx
                if pair in kin_neighbours or pair in ignore_set:
                    continue
                idx_i.append(i)
                idx_j.append(j)

        local_pa = torch.tensor(points_a, dtype=torch.float32)
        local_pb = torch.tensor(points_b, dtype=torch.float32)
        cap_r = torch.tensor(cap_radii, dtype=torch.float32)
        cap_li = torch.tensor(cap_link_indices, dtype=torch.long)

        # Reference-pose filter: remove pairs that already penetrate or nearly
        # touch at the reference configuration (structural overlaps).
        if filter_below_rest_dist is not None and idx_i:
            ref_q = filter_q if filter_q is not None else model.q_default
            rc_tmp = RobotCollision(
                _mode="capsule",
                _local_centers=None, _radii=None, _link_indices=None,
                _local_points_a=local_pa, _local_points_b=local_pb,
                _capsule_radii=cap_r, _capsule_link_indices=cap_li,
                _active_pairs_i=tuple(idx_i), _active_pairs_j=tuple(idx_j),
            )
            with torch.no_grad():
                ref_dists = rc_tmp.compute_self_collision_distance(
                    model, ref_q, base_pose=filter_base_pose,
                )
            keep = ref_dists >= filter_below_rest_dist
            idx_i = [idx_i[k] for k in range(len(idx_i)) if keep[k].item()]
            idx_j = [idx_j[k] for k in range(len(idx_j)) if keep[k].item()]

        return RobotCollision(
            _mode="capsule",
            _local_centers=None,
            _radii=None,
            _link_indices=None,
            _local_points_a=local_pa,
            _local_points_b=local_pb,
            _capsule_radii=cap_r,
            _capsule_link_indices=cap_li,
            _active_pairs_i=tuple(idx_i),
            _active_pairs_j=tuple(idx_j),
        )

    @staticmethod
    def from_capsule_decomposition(
        capsule_decomposition: dict,
        model: RobotModel,
    ) -> "RobotCollision":
        """Build a capsule collision model from a manually specified decomposition.

        Args:
            capsule_decomposition: Dict mapping ``link_name`` to capsule specs.
                Single capsule per link::

                    {"panda_link3": {"point_a": [x,y,z], "point_b": [x,y,z], "radius": r}}

                Multiple capsules per link::

                    {"panda_link3": {"points_a": [[...], ...], "points_b": [[...], ...], "radii": [r, ...]}}

            model: RobotModel (for link names and adjacency).

        Returns:
            RobotCollision in capsule mode.
        """
        link_name_to_idx = {name: idx for idx, name in enumerate(model.links.names)}

        points_a: list[list[float]] = []
        points_b: list[list[float]] = []
        cap_radii: list[float] = []
        cap_link_indices: list[int] = []
        geom_counts: list[int] = [0] * model.links.num_links

        for link_name, data in capsule_decomposition.items():
            if link_name not in link_name_to_idx:
                continue
            link_idx = link_name_to_idx[link_name]

            if "points_a" in data:
                pas = data["points_a"]
                pbs = data["points_b"]
                radii = data["radii"]
            else:
                pas = [data["point_a"]]
                pbs = [data["point_b"]]
                radii = [data["radius"]]

            for pa, pb, r in zip(pas, pbs, radii):
                points_a.append(list(pa))
                points_b.append(list(pb))
                cap_radii.append(float(r))
                cap_link_indices.append(link_idx)
                geom_counts[link_idx] += 1

        if not points_a:
            raise ValueError("No capsules found in capsule_decomposition.")

        adjacent_links = _build_adjacent_set(model)

        # Build geometry-level pair indices (multiple capsules per link possible).
        # geom_offsets[i] = flat index of the first geometry for link i.
        geom_offsets = [0] * (model.links.num_links + 1)
        for i in range(model.links.num_links):
            geom_offsets[i + 1] = geom_offsets[i] + geom_counts[i]

        idx_i: list[int] = []
        idx_j: list[int] = []
        for li in range(model.links.num_links):
            for lj in range(li + 1, model.links.num_links):
                if (min(li, lj), max(li, lj)) in adjacent_links:
                    continue
                for gi in range(geom_counts[li]):
                    for gj in range(geom_counts[lj]):
                        idx_i.append(geom_offsets[li] + gi)
                        idx_j.append(geom_offsets[lj] + gj)

        return RobotCollision(
            _mode="capsule",
            _local_centers=None,
            _radii=None,
            _link_indices=None,
            _local_points_a=torch.tensor(points_a, dtype=torch.float32),
            _local_points_b=torch.tensor(points_b, dtype=torch.float32),
            _capsule_radii=torch.tensor(cap_radii, dtype=torch.float32),
            _capsule_link_indices=torch.tensor(cap_link_indices, dtype=torch.long),
            _active_pairs_i=tuple(idx_i),
            _active_pairs_j=tuple(idx_j),
        )

    @staticmethod
    def from_sphere_decomposition(
        sphere_decomposition: dict,
        model: RobotModel,
    ) -> "RobotCollision":
        """Create a RobotCollision from a sphere decomposition dict.

        Args:
            sphere_decomposition: Dict mapping link_name to list of
                {'center': [x,y,z], 'radius': r} dicts, OR to
                {'centers': [[x,y,z], ...], 'radii': [r, ...]} dicts.
            model: RobotModel instance (for link name/index mapping).

        Returns:
            RobotCollision in sphere mode.
        """
        link_name_to_idx = {name: idx for idx, name in enumerate(model.links.names)}

        all_centers: list[list[float]] = []
        all_radii: list[float] = []
        sphere_link_indices: list[int] = []
        geom_counts: list[int] = [0] * model.links.num_links

        for link_name, data in sphere_decomposition.items():
            if link_name not in link_name_to_idx:
                continue
            link_idx = link_name_to_idx[link_name]

            if 'centers' in data:
                centers = data['centers']
                radii = data['radii']
            else:
                centers = [data['center']]
                radii = [data['radius']]

            for center, radius in zip(centers, radii):
                all_centers.append(list(center))
                all_radii.append(float(radius))
                sphere_link_indices.append(link_idx)
                geom_counts[link_idx] += 1

        if not all_centers:
            raise ValueError("No spheres found in sphere_decomposition.")

        local_centers = torch.tensor(all_centers, dtype=torch.float32)
        radii = torch.tensor(all_radii, dtype=torch.float32)
        link_indices = torch.tensor(sphere_link_indices, dtype=torch.long)

        adjacent_links = _build_adjacent_set(model)

        num_spheres = len(all_centers)
        idx_i: list[int] = []
        idx_j: list[int] = []
        for i in range(num_spheres):
            for j in range(i + 1, num_spheres):
                li = sphere_link_indices[i]
                lj = sphere_link_indices[j]
                if li == lj:
                    continue
                if (min(li, lj), max(li, lj)) in adjacent_links:
                    continue
                idx_i.append(i)
                idx_j.append(j)

        return RobotCollision(
            _mode="sphere",
            _local_centers=local_centers,
            _radii=radii,
            _link_indices=link_indices,
            _local_points_a=None,
            _local_points_b=None,
            _capsule_radii=None,
            _capsule_link_indices=None,
            _active_pairs_i=tuple(idx_i),
            _active_pairs_j=tuple(idx_j),
        )

    # ------------------------------------------------------------------ #
    #  Internal geometry helpers
    # ------------------------------------------------------------------ #

    def _get_world_spheres(
        self,
        model: RobotModel,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> list[Sphere]:
        """Transform all spheres from link-local to world frame (sphere mode)."""
        assert self._mode == "sphere", "_get_world_spheres requires sphere mode"
        assert self._local_centers is not None
        assert self._radii is not None
        assert self._link_indices is not None

        fk = model.forward_kinematics(q, base_pose=base_pose)  # (num_links, 7)
        spheres = []
        for i in range(len(self._radii)):
            link_idx = int(self._link_indices[i].item())
            T = fk[link_idx]
            R = so3_rotation_matrix(T[3:7])
            t = T[:3]
            world_center = R @ self._local_centers[i].to(T.device) + t
            spheres.append(Sphere(center=world_center, radius=float(self._radii[i].item())))
        return spheres

    def _get_world_capsules(
        self,
        model: RobotModel,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> list[Capsule]:
        """Transform all capsules from link-local to world frame (capsule mode).

        Returns a list of Capsule objects for use with the generic compute_distance
        dispatcher (e.g. world-collision).  For self-collision use
        ``_get_world_capsules_batched`` instead.
        """
        assert self._mode == "capsule", "_get_world_capsules requires capsule mode"
        assert self._local_points_a is not None
        assert self._local_points_b is not None
        assert self._capsule_radii is not None
        assert self._capsule_link_indices is not None

        fk = model.forward_kinematics(q, base_pose=base_pose)  # (num_links, 7)
        capsules = []
        for i in range(len(self._capsule_radii)):
            link_idx = int(self._capsule_link_indices[i].item())
            T = fk[link_idx]
            R = so3_rotation_matrix(T[3:7])
            t = T[:3]
            pa = R @ self._local_points_a[i].to(T.device) + t
            pb = R @ self._local_points_b[i].to(T.device) + t
            capsules.append(Capsule(point_a=pa, point_b=pb, radius=float(self._capsule_radii[i].item())))
        return capsules

    def _get_world_capsules_batched(
        self,
        model: RobotModel,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorised FK application for all capsules.

        Returns:
            ``(pa_world, pb_world)`` — each ``(num_capsules, 3)`` in world frame.
        """
        assert self._mode == "capsule"
        assert self._local_points_a is not None
        assert self._local_points_b is not None
        assert self._capsule_link_indices is not None

        fk = model.forward_kinematics(q, base_pose=base_pose)          # (num_links, 7)
        cap_fk = fk[self._capsule_link_indices]                         # (N, 7)
        R = so3_rotation_matrix(cap_fk[:, 3:7])                        # (N, 3, 3)
        t = cap_fk[:, :3]                                               # (N, 3)
        local_pa = self._local_points_a.to(q.device)                   # (N, 3)
        local_pb = self._local_points_b.to(q.device)                   # (N, 3)
        # Batched rotation:  (N,3,3) @ (N,3,1) → (N,3,1) → (N,3)
        pa_world = (R @ local_pa.unsqueeze(-1)).squeeze(-1) + t
        pb_world = (R @ local_pb.unsqueeze(-1)).squeeze(-1) + t
        return pa_world, pb_world

    # ------------------------------------------------------------------ #
    #  Distance computation
    # ------------------------------------------------------------------ #

    def compute_self_collision_distance(
        self,
        model: RobotModel,
        q: torch.Tensor,
        base_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute signed distances for all active self-collision pairs.

        Capsule mode uses a fully vectorised batch computation (one FK call +
        batched segment-to-segment distance for all pairs simultaneously).
        Sphere mode falls back to the sequential loop.

        Args:
            model: RobotModel.
            q: Joint configuration ``(num_actuated,)``.
            base_pose: Optional base pose ``[tx,ty,tz,qx,qy,qz,qw]`` for
                floating-base robots.

        Returns:
            Shape (num_active_pairs,). Positive = separated, negative = penetrating.
        """
        if not self._active_pairs_i:
            return torch.zeros(0, dtype=q.dtype, device=q.device)

        if self._mode == "capsule":
            # --- Vectorised capsule path ---
            pa, pb = self._get_world_capsules_batched(model, q, base_pose)  # (N,3)
            ii = list(self._active_pairs_i)
            jj = list(self._active_pairs_j)
            pa1, pb1 = pa[ii], pb[ii]   # (M, 3)
            pa2, pb2 = pa[jj], pb[jj]   # (M, 3)
            assert self._capsule_radii is not None
            r1 = self._capsule_radii[ii].to(q.device)  # (M,)
            r2 = self._capsule_radii[jj].to(q.device)  # (M,)
            c1, c2 = _utils.closest_segment_to_segment_points(pa1, pb1, pa2, pb2)
            _, dist = _utils.normalize_with_norm(c2 - c1)
            return dist - (r1 + r2)
        else:
            # --- Sphere path (sequential, kept for compatibility) ---
            geoms = self._get_world_spheres(model, q, base_pose=base_pose)
            dists = [compute_distance(geoms[i], geoms[j])
                     for i, j in zip(self._active_pairs_i, self._active_pairs_j)]
            return torch.stack(dists)

    def compute_world_collision_distance(
        self,
        model: RobotModel,
        q: torch.Tensor,
        world_geom: list[CollGeom],
        base_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute signed distances from each robot geometry to all world geometries.

        Args:
            model: RobotModel.
            q: Joint configuration ``(num_actuated,)``.
            world_geom: List of world collision primitives.
            base_pose: Optional base pose for floating-base robots.

        Returns:
            Shape (num_robot_geoms * len(world_geom),). Negative = penetrating.
        """
        if self._mode == "sphere":
            robot_geoms = self._get_world_spheres(model, q, base_pose=base_pose)
        else:
            robot_geoms = self._get_world_capsules(model, q, base_pose=base_pose)

        dists = []
        for rg in robot_geoms:
            for wg in world_geom:
                dists.append(compute_distance(wg, rg))
        if not dists:
            return torch.zeros(0)
        return torch.stack(dists)

    # ------------------------------------------------------------------ #
    #  URDF mesh extraction (static helper)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_trimesh_collision_geometries(
        urdf: "yourdfpy.URDF",
        link_name: str,
    ) -> "tm.Trimesh":
        """Extract and merge all collision meshes for *link_name* from the URDF.

        Handles box, cylinder, sphere, and mesh geometry types.  Applies each
        collision element's origin transform before merging.

        Returns an empty :class:`trimesh.Trimesh` when the link has no collision
        geometry or *link_name* is not found in the URDF.
        """
        import trimesh as tm  # noqa: F811

        if link_name not in urdf.link_map:
            return tm.Trimesh()

        link = urdf.link_map[link_name]
        filename_handler = urdf._filename_handler
        coll_meshes: list["tm.Trimesh"] = []

        for collision in link.collisions:
            geom = collision.geometry
            mesh: Optional["tm.Trimesh"] = None

            transform = collision.origin if collision.origin is not None else _identity_4x4()

            if geom.box is not None:
                mesh = tm.creation.box(extents=geom.box.size)
            elif geom.cylinder is not None:
                mesh = tm.creation.cylinder(
                    radius=geom.cylinder.radius,
                    height=geom.cylinder.length,
                )
            elif geom.sphere is not None:
                mesh = tm.creation.icosphere(radius=geom.sphere.radius)
            elif geom.mesh is not None:
                try:
                    mesh_path = geom.mesh.filename
                    loaded = tm.load(
                        file_obj=filename_handler(mesh_path),
                        force="mesh",
                    )
                    scale = geom.mesh.scale if geom.mesh.scale is not None else [1.0, 1.0, 1.0]
                    if isinstance(loaded, tm.Trimesh):
                        mesh = loaded.copy()
                        mesh.apply_scale(scale)
                    elif isinstance(loaded, tm.Scene):
                        geoms_in_scene = list(loaded.geometry.values())
                        if geoms_in_scene and isinstance(geoms_in_scene[0], tm.Trimesh):
                            mesh = geoms_in_scene[0].copy()
                            mesh.apply_scale(scale)
                    if mesh is not None:
                        mesh.fix_normals()
                except Exception:
                    continue

            if mesh is not None:
                mesh.apply_transform(transform)
                coll_meshes.append(mesh)

        if not coll_meshes:
            return tm.Trimesh()
        return tm.util.concatenate(coll_meshes)


# ------------------------------------------------------------------ #
#  Module-level helpers
# ------------------------------------------------------------------ #

def _build_adjacent_set(model: RobotModel) -> set[tuple[int, int]]:
    """Return the set of (min_link_idx, max_link_idx) immediate parent-child pairs."""
    adjacent: set[tuple[int, int]] = set()
    for joint_idx in range(len(model._fk_joint_parent_link)):
        p = model._fk_joint_parent_link[joint_idx]
        c = model._fk_joint_child_link[joint_idx]
        adjacent.add((min(p, c), max(p, c)))
    return adjacent


def _build_kinematic_neighbours(
    model: RobotModel,
    max_depth: int,
) -> set[tuple[int, int]]:
    """Return all (min_i, max_i) link-index pairs within *max_depth* joints of each other.

    Uses BFS on the undirected link-adjacency graph.  Depth 1 is equivalent to
    ``_build_adjacent_set``; depth 2 also includes skip-one pairs, etc.
    """
    if max_depth < 1:
        return set()

    # Build undirected adjacency list over link indices.
    from collections import defaultdict
    adj: dict[int, list[int]] = defaultdict(list)
    for joint_idx in range(len(model._fk_joint_parent_link)):
        p = int(model._fk_joint_parent_link[joint_idx])
        c = int(model._fk_joint_child_link[joint_idx])
        adj[p].append(c)
        adj[c].append(p)

    result: set[tuple[int, int]] = set()
    for start in range(model.links.num_links):
        # BFS from 'start', recording all reachable nodes within max_depth steps.
        dist: dict[int, int] = {start: 0}
        q: deque[int] = deque([start])
        while q:
            node = q.popleft()
            d = dist[node]
            if d >= max_depth:
                continue
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = d + 1
                    q.append(nb)
        for nb, d in dist.items():
            if nb != start and d > 0:
                result.add((min(start, nb), max(start, nb)))

    return result


def _identity_4x4():
    import numpy as np
    return np.eye(4)
