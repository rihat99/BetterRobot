"""``parse_urdf`` raises ``ModelError`` naming the file and the element at fault (record 0047)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from better_robot.exceptions import ModelError
from better_robot.io import ModelBuilder, build_model, load, load_geometry
from better_robot.io.parser_urdf import parse_urdf
from better_robot.viewer import robot_meshes

from tests.support.robots import panda_urdf

TWO_LINKS = '<robot name="r"><link name="base"/><link name="tip"/>{joint}</robot>'


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "r.urdf"
    path.write_text(text)
    return path


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.urdf"
    with pytest.raises(ModelError, match=re.escape(str(path))):
        parse_urdf(path)


def test_empty_file_names_the_path(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    with pytest.raises(ModelError, match=re.escape(str(path))):
        parse_urdf(path)


def test_unknown_parent_link_names_file_and_joint(tmp_path: Path) -> None:
    joint = '<joint name="j" type="fixed"><parent link="nowhere"/><child link="tip"/></joint>'
    path = _write(tmp_path, TWO_LINKS.format(joint=joint))
    with pytest.raises(ModelError, match=re.escape(f"{path}: <joint name='j'>: parent link 'nowhere'")):
        parse_urdf(path)


def test_unknown_joint_type_names_file_and_joint(tmp_path: Path) -> None:
    joint = '<joint name="j" type="screw"><parent link="base"/><child link="tip"/></joint>'
    path = _write(tmp_path, TWO_LINKS.format(joint=joint))
    with pytest.raises(ModelError, match=re.escape(f"{path}: <joint name='j'>: type 'screw'")):
        parse_urdf(path)


def test_zero_axis_names_file_and_joint(tmp_path: Path) -> None:
    joint = '<joint name="j" type="revolute"><parent link="base"/><child link="tip"/><axis xyz="0 0 0"/></joint>'
    path = _write(tmp_path, TWO_LINKS.format(joint=joint))
    with pytest.raises(ModelError, match=re.escape(f"{path}: <joint name='j'>: <axis> is zero")):
        parse_urdf(path)


def test_unsupported_shape_names_file_and_tag(tmp_path: Path) -> None:
    visual = '<visual><geometry><capsule radius="0.1" length="0.5"/></geometry></visual>'
    path = _write(tmp_path, f'<robot name="r"><link name="base">{visual}</link></robot>')
    with pytest.raises(ModelError, match=re.escape(f"{path}: ") + "Unknown tag: capsule"):
        parse_urdf(path)


def test_unreadable_mesh_names_body_and_uri(tmp_path: Path) -> None:
    """Mesh files are read where meshes are drawn, not at parse time (record 0048)."""
    visual = '<visual><geometry><mesh filename="missing.stl"/></geometry></visual>'
    path = _write(tmp_path, f'<robot name="r"><link name="base">{visual}</link></robot>')
    model = load(path)
    with pytest.raises(ModelError, match=re.escape(f"body {model.body_id('base')}: mesh 'missing.stl' is not a file")):
        robot_meshes(load_geometry(path, model), root=path.parent)


def test_a_two_component_axis_names_file_and_joint(tmp_path: Path) -> None:
    joint = '<joint name="j" type="revolute"><parent link="base"/><child link="tip"/><axis xyz="1 0"/></joint>'
    path = _write(tmp_path, TWO_LINKS.format(joint=joint))
    with pytest.raises(ModelError, match=re.escape(f"{path}: <joint name='j'>: <axis> has 2 components")):
        parse_urdf(path)


@pytest.mark.parametrize(
    ("cut", "line"),
    [pytest.param(None, 1, id="unterminated_link"), pytest.param(0.8, None, id="panda_cut_at_80_percent")],
)
def test_malformed_xml_names_file_and_line(tmp_path: Path, cut: float | None, line: int | None) -> None:
    """yourdfpy would recover these silently (the cut Panda loads with 7 of its 8 coordinates)."""
    text = '<robot name="r"><link name="base">' if cut is None else panda_urdf().read_text()
    if cut is not None:
        text = text[: int(len(text) * cut)]
        line = text.count("\n") + 1
    path = _write(tmp_path, text)
    with pytest.raises(ModelError, match=re.escape(f"{path}: not well-formed XML at line {line}")):
        parse_urdf(path)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        pytest.param(
            '<link name="tip"/><joint name="j" type="fixed"><parent link="base"/><child link="tip"/></joint>',
            "two <link> elements are named 'tip'; rename one",
            id="duplicate_link",
        ),
        pytest.param(
            '<link name="end"/><joint name="j" type="fixed"><parent link="base"/><child link="tip"/></joint>'
            '<joint name="j" type="fixed"><parent link="tip"/><child link="end"/></joint>',
            "two <joint> elements are named 'j'; rename one",
            id="duplicate_joint",
        ),
        pytest.param(
            '<link name="stray"/><joint name="j" type="fixed"><parent link="base"/><child link="tip"/></joint>',
            "2 links are no joint's child ['base', 'stray']; a URDF tree has one root link",
            id="stray_link",
        ),
    ],
)
def test_a_description_that_is_not_one_tree_names_the_culprit(tmp_path: Path, body: str, message: str) -> None:
    path = _write(tmp_path, TWO_LINKS.format(joint=body))
    with pytest.raises(ModelError, match=re.escape(f"{path}: {message}")):
        parse_urdf(path)


def test_a_stray_link_listed_first_is_named_too(tmp_path: Path) -> None:
    joint = '<joint name="j" type="fixed"><parent link="base"/><child link="tip"/></joint>'
    path = _write(tmp_path, f'<robot name="r"><link name="stray"/><link name="base"/><link name="tip"/>{joint}</robot>')
    with pytest.raises(ModelError, match=re.escape("2 links are no joint's child ['stray', 'base']")):
        parse_urdf(path)


def test_lower_above_upper_raises_in_the_limits_stage(tmp_path: Path) -> None:
    limit = '<limit lower="1" upper="-1" velocity="1" effort="1"/>'
    joint = (
        f'<joint name="j" type="revolute"><parent link="base"/><child link="tip"/><axis xyz="0 0 1"/>{limit}</joint>'
    )
    message = re.escape("joint 'j': lower limit 1.0 > upper limit -1.0; swap them")
    with pytest.raises(ModelError, match=message):
        load(_write(tmp_path, TWO_LINKS.format(joint=joint)))
    builder = ModelBuilder("r")
    builder.add_body("base")
    builder.add_body("tip")
    builder.add_revolute("j", parent="base", child="tip", axis=(0.0, 0.0, 1.0), lower=1.0, upper=-1.0)
    with pytest.raises(ModelError, match=message):
        build_model(builder.finalize())
