"""The sphere files that ship in ``collision.SPHERE_SETS`` load on the tracked robots, and their ignore lists are exactly
the frame pairs whose spheres overlap at the neutral configuration, so every other pair is clear there."""

from pathlib import Path

import pytest
import torch

import better_robot as br
from better_robot import collision
from better_robot.exceptions import ModelError
from tests.support.robots import g1_urdf, panda_urdf


@pytest.mark.parametrize(("name", "urdf"), [("panda.json", panda_urdf()), ("g1.json", g1_urdf())])
def test_shipped_sets_ignore_exactly_the_pairs_that_overlap_at_neutral(name: str, urdf: Path) -> None:
    model = br.load(urdf, dtype=torch.float64)
    spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / name, model)
    neutral = br.forward_kinematics(model, model.q_neutral)
    every = collision.self_collision_pairs(model, spheres)
    frames = [model.frame_names[frame] for frame in spheres.frame_ids.tolist()]
    overlapping = {
        frozenset((frames[i], frames[j]))
        for (i, j), distance in zip(every.tolist(), collision.self_distances(spheres, neutral, pairs=every).tolist())
        if distance < 0
    }
    assert overlapping == {frozenset(pair) for pair in ignore}
    pairs = collision.self_collision_pairs(model, spheres, ignore=ignore)
    assert pairs.shape[0] > 0
    assert bool((collision.self_distances(spheres, neutral, pairs=pairs) > 0).all())


def test_a_file_of_another_form_raises(tmp_path: Path) -> None:
    path = tmp_path / "spheres.json"
    path.write_text('{"spheres": {"panda_link1": [[0.0, 0.0, 0.1]]}}')
    with pytest.raises(ModelError, match="sphere file"):
        collision.load_spheres(path, br.load(panda_urdf()))
