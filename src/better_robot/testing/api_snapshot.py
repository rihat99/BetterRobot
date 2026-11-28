"""Write, check or diff the public-API snapshot of a package: the command line over ``testing.snapshot_differences``.

The snapshot maps every path of the package's ``__all__`` tree to one line describing it
(``better_robot.testing.public_api``). ``--write`` rewrites it after an intended API change; ``--check`` prints the
paths added (``+``), removed (``-``) and changed (``~``) and exits 1 when there are any; ``--diff`` prints the same
against an older snapshot, such as the last release's, and exits 0. Run it from the repository root::

    uv run python -m better_robot.testing.api_snapshot better_robot tests/contract/public_api.json --check
    uv run python -m better_robot.testing.api_snapshot better_robot old_public_api.json --diff
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import public_api, snapshot_differences


def main() -> int:
    """Run the command line; the exit code is 1 only for ``--check`` with differences."""
    parser = argparse.ArgumentParser(
        prog="python -m better_robot.testing.api_snapshot",
        description="Write, check or diff the public-API snapshot of a package.",
    )
    parser.add_argument("package", help="the importable package, such as better_robot")
    parser.add_argument("snapshot", type=Path, help="its snapshot, such as tests/contract/public_api.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="rewrite the snapshot from the installed package")
    action.add_argument("--check", action="store_true", help="print the differences; exit 1 when there are any")
    action.add_argument("--diff", action="store_true", help="print the differences from an older snapshot")
    arguments = parser.parse_args()
    if arguments.write:
        arguments.snapshot.write_text(json.dumps(public_api(arguments.package), indent=2, sort_keys=True) + "\n")
        return 0
    lines = snapshot_differences(arguments.package, arguments.snapshot)
    sys.stdout.write("".join(f"{line}\n" for line in lines))
    return 1 if arguments.check and lines else 0


if __name__ == "__main__":
    sys.exit(main())
