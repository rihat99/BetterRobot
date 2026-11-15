"""The sphere files that ship in ``collision.SPHERE_SETS`` load on the tracked robots and keep the rule of
``tools/fit_spheres.py``: an ignored frame pair carries the shapes' evidence (never, always or at the neutral
configuration in collision), every kept pair is clear at the neutral configuration, and mirrored pairs behave alike."""

import json
import math
from pathlib import Path

import pytest
import torch

import better_robot as br
from better_robot import collision, io
from better_robot.exceptions import ModelError
from tests.support.robots import g1_urdf, panda_urdf

SETS = [("panda.json", panda_urdf()), ("g1.json", g1_urdf())]
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04]


def _swap(frame: str) -> str:
    side = {"left_": "right_", "right_": "left_"}
    return next((side[prefix] + frame[len(prefix) :] for prefix in side if frame.startswith(prefix)), frame)


@pytest.mark.parametrize(("name", "urdf"), SETS)
def test_ignored_pairs_carry_evidence_and_kept_pairs_are_clear_at_neutral(name: str, urdf: Path) -> None:
    model = br.load(urdf, dtype=torch.float64)
    spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / name, model)
    reasons = {row[2] for row in json.loads((collision.SPHERE_SETS / name).read_text())["ignore"]}
    assert reasons <= {"never", "always", "neutral"}
    frames = [model.frame_names[frame] for frame in spheres.frame_ids.tolist()]
    every = {frozenset((frames[i], frames[j])) for i, j in collision.self_collision_pairs(model, spheres).tolist()}
    assert {frozenset(pair) for pair in ignore} <= every  # adjacent pairs are dropped without an entry
    pairs = collision.self_collision_pairs(model, spheres, ignore=ignore)
    assert pairs.shape[0] > 0
    neutral = br.forward_kinematics(model, model.q_neutral)
    assert bool((collision.self_distances(spheres, neutral, pairs=pairs) > 0).all())


def test_mirrored_g1_pairs_have_equal_status_and_neutral_distance() -> None:
    model = br.load(g1_urdf(), dtype=torch.float64)
    spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / "g1.json", model)
    frames = [model.frame_names[frame] for frame in spheres.frame_ids.tolist()]
    every = collision.self_collision_pairs(model, spheres)
    distance = collision.self_distances(spheres, br.forward_kinematics(model, model.q_neutral), pairs=every).tolist()
    nearest: dict[frozenset[str], float] = {}
    for (i, j), value in zip(every.tolist(), distance, strict=True):
        key = frozenset((frames[i], frames[j]))
        nearest[key] = min(nearest.get(key, value), value)
    ignored = {frozenset(pair) for pair in ignore}
    for key, value in nearest.items():
        twin = frozenset(_swap(frame) for frame in key)
        assert (key in ignored) == (twin in ignored), sorted(key)
        assert abs(value - nearest[twin]) < 1e-4, sorted(key)


def test_g1_feet_reach_at_most_about_5_mm_below_the_sole() -> None:
    """The sole is the lowest point of the description's four contact spheres per foot (its display mesh's lowest point
    lies 0.4 mm lower); a floor constraint on the feet then holds the G1 on its soles."""
    model = br.load(g1_urdf(), dtype=torch.float64)
    spheres, _ = collision.load_spheres(collision.SPHERE_SETS / "g1.json", model)
    geometry = io.load_geometry(g1_urdf(), model)
    for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
        contacts = [item for item in geometry.shapes if item.body == model.body_id(foot) and item.collision]
        sole = min(item.shape.placement[2] - item.shape.radius for item in contacts)
        mine = spheres.frame_ids == model.frame_id(foot)
        lowest = float((spheres.offsets[mine, 2] - spheres.radii[mine]).min())
        assert sole - 0.006 <= lowest <= sole, foot


def test_panda_kept_pairs_are_several_millimetres_apart_at_the_ready_pose() -> None:
    """The Panda's neutral configuration folds the hand down beside link 5, where kept pairs clear by a fraction of a
    millimetre; a demo starts from the ready pose of the trajectory guide instead."""
    model = br.load(panda_urdf(), dtype=torch.float64)
    spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / "panda.json", model)
    pairs = collision.self_collision_pairs(model, spheres, ignore=ignore)
    ready = br.forward_kinematics(model, torch.tensor(READY, dtype=torch.float64))
    assert float(collision.self_distances(spheres, ready, pairs=pairs).min()) > 0.005


@pytest.mark.parametrize(
    "text",
    [
        "{",
        "[]",
        '{"spheres": {"panda_link1": [[0.0, 0.0, 0.1]]}}',
        '{"spheres": {"panda_link1": [3]}}',
        '{"spheres": {"panda_link1": [[0.0, "a", 0.1, 0.05]]}}',
        '{"spheres": {"panda_link1": [[0.0, 0.0, 0.1, NaN]]}}',
        '{"spheres": {"panda_link1": [[0.0, 0.0, Infinity, 0.05]]}}',
        '{"spheres": {"panda_link1": [[0.0, 0.0, 0.1, -0.05]]}}',
        '{"spheres": {"panda_link1": [[0.0, 0.0, 0.1, 0]]}}',
        '{"spheres": {}, "ignore": null}',
        '{"spheres": {}, "ignore": [1]}',
        '{"spheres": {}, "ignore": [["panda_link1", "panda_link3"]]}',
    ],
)
def test_a_file_of_another_form_raises(tmp_path: Path, text: str) -> None:
    path = tmp_path / "spheres.json"
    path.write_text(text)
    with pytest.raises(ModelError, match="sphere file|not JSON"):
        collision.load_spheres(path, br.load(panda_urdf()))
