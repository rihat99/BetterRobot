"""Every tracked test robot ships its licence and its provenance (decision record 0039)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROBOTS = Path(__file__).resolve().parents[1] / "fixtures" / "robots"


@pytest.mark.parametrize("robot", sorted(path for path in ROBOTS.iterdir() if path.is_dir()), ids=lambda p: p.name)
def test_robot_fixture_has_licence_and_provenance(robot: Path) -> None:
    """``LICENSE`` is the upstream licence; ``README.md`` states source, version, licence and changes."""
    missing = [name for name in ("LICENSE", "README.md") if not (robot / name).is_file()]
    assert not missing, (
        f"tests/fixtures/robots/{robot.name} lacks {missing}: copy the upstream licence, write the provenance"
    )
