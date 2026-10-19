"""MJCF capsule and cylinder ``fromto`` reach the IR with MuJoCo's compiled placement and length."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from better_robot.io import parse

MJCF = """<mujoco><default><default class="col"><geom type="capsule" size="0.02"/></default></default><worldbody>
<body name="b1"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/><joint name="j1" type="hinge" axis="0 0 1"/>
<geom class="col" fromto="0 0 0 0 0 0.3"/>
<geom class="col" fromto="0 0 0.3 0 0 0"/>
<geom class="col" size="0.03" fromto="0.1 -0.2 0.05 -0.15 0.1 -0.2"/>
<geom type="cylinder" size="0.04" fromto="0.02 0 0 0.02 0 -0.08"/>
<geom type="capsule" size="0.05 0.1" pos="0.1 0.2 0.3" quat="0.9238795 0 0.3826834 0"/>
<geom type="sphere" size="0.07" pos="0 0 -0.08"/>
<body name="b2" pos="0 0 1"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
<joint name="j2" type="hinge" axis="0 1 0"/><geom class="col" fromto="-0.01 0 -0.01 0.08 0 -0.01"/></body>
</body></worldbody></mujoco>"""


def test_fromto_matches_mujoco_compiler(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    path = tmp_path / "fromto.xml"
    path.write_text(MJCF)
    compiled = mujoco.MjModel.from_xml_path(str(path))
    geoms = [geom for body in parse(path).bodies for geom in body.collision_geoms]
    assert len(geoms) == compiled.ngeom == 7
    for index, geom in enumerate(geoms):
        origin = geom.origin.double().numpy()
        np.testing.assert_allclose(origin[:3], compiled.geom_pos[index], atol=1e-12)
        quat_wxyz, expected = np.roll(origin[3:], 1), compiled.geom_quat[index]
        assert min(np.abs(quat_wxyz - expected).max(), np.abs(quat_wxyz + expected).max()) < 1e-7  # q and -q
        assert geom.params["radius"] == pytest.approx(compiled.geom_size[index, 0])
        if geom.kind in ("capsule", "cylinder"):
            assert geom.params["length"] == pytest.approx(2 * compiled.geom_size[index, 1])
    assert geoms[2].params["length"] == pytest.approx(float(np.linalg.norm([0.25, -0.3, 0.25])))
