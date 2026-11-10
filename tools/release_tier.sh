#!/usr/bin/env bash
# The release tier: every check of the repository, the CUDA tests and the CUDA counters included, stopping at the
# first failure. Run by hand before a tag, from any directory:
#     bash tools/release_tier.sh --gpu <index>
# <index> is the nvidia-smi (PCI bus) index of a free GPU; the run fails before its first test when torch cannot use
# it. docs/guides/run-the-tests.md lists the three tiers.
set -euo pipefail

if [[ $# -ne 2 || $1 != --gpu ]]; then
    echo "usage: bash tools/release_tier.sh --gpu <index>" >&2
    exit 2
fi
gpu=$2
cd "$(dirname "$0")/.."

step() {
    echo "== $*"
    "$@"
}
on_gpu() {
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

step uv lock --check
step on_gpu uv run python -c \
    "import sys, torch; sys.exit(None if torch.cuda.is_available() else 'release_tier.sh: torch cannot use GPU $gpu')"
step uv run ruff check .
step uv run ruff format --check .
step uv run python tools/api_snapshot.py --check
step uv run mkdocs build --strict -d "$(mktemp -d)"
step on_gpu uv run pytest tests -q -m cuda
step on_gpu uv run python benchmarks/run.py --device cuda --counters-only
step uv run pytest tests -q -m "not cuda"
# tests/conftest.py turns the internal checks on unless the environment sets the switch, so the full tier above ran
# with them; this run is the configuration users get.
step env BETTER_ROBOT_INTERNAL_CHECKS=0 uv run pytest tests -q -m "not cuda"
COVERAGE_FILE=$(mktemp)  # outside the checkout, so the run leaves the working tree clean
export COVERAGE_FILE
step uv run coverage run --branch --source=src/better_robot/lie,src/better_robot/spatial \
    -m pytest tests/lie tests/spatial -q -m "not cuda"
step uv run coverage report --fail-under=100
echo "== release tier passed"
