"""Execute the Python snippets in ``docs/**/*.md`` and ``README.md``; check that every referenced image exists."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
DOCS = REPO / "docs"
FAKE_SERVER = REPO / "tests" / "viewer" / "fake_server.py"
PAGE_TIMEOUT_S = 120

# Fence info string: ```python [tags...]; a ``no-run`` tag opts the block out of execution.
FENCE = re.compile(r"^```[ \t]*python(?P<info>[^\n]*)\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
IMAGE = re.compile(r"!\[[^\]]*\]\((?P<path>[^)\s]+)")
FIGURE = re.compile(r"(?P<path>(?:\.\./)*(?:figures|images)/[\w./-]+\.(?:svg|png|jpg|gif))")

# Runs in a fresh interpreter so the CPU-only environment is set before torch loads, and so viser never binds:
# ``viser.ViserServer`` is replaced by the recording fake from the viewer tests.
RUNNER = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("docs_fake_server", sys.argv[1])
fake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake)
try:
    import viser
    viser.ViserServer = lambda *args, **kwargs: fake.FakeServer()
except ImportError:
    pass
namespace = {"__name__": "__docs__"}
for index, source in enumerate(sys.argv[2:]):
    exec(compile(source, f"<block {index}>", "exec"), namespace)
"""


def _pages() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


def _runnable_blocks(page: Path) -> list[str]:
    return [m["body"] for m in FENCE.finditer(page.read_text()) if "no-run" not in m["info"].split()]


def _pages_with_code() -> list[Path]:
    return [page for page in [*_pages(), REPO / "README.md"] if _runnable_blocks(page)]


@pytest.mark.parametrize("page", _pages_with_code(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_page_snippets_execute(page: Path) -> None:
    """Run a page's Python blocks in order in one namespace, CPU only, with a fake viser server."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
    completed = subprocess.run(
        [sys.executable, "-c", RUNNER, str(FAKE_SERVER), *_runnable_blocks(page)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=PAGE_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, f"{page.relative_to(REPO)} failed:\n{completed.stderr[-4000:]}"


def test_index_has_runnable_snippet() -> None:
    """The landing page example is part of the snippet run."""
    assert _runnable_blocks(DOCS / "index.md")


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.relative_to(DOCS).as_posix())
def test_referenced_images_exist(page: Path) -> None:
    """Every ``![...](path)`` and ``figures/*.svg`` / ``images/*`` reference resolves to a file."""
    text = page.read_text()
    paths = {m["path"] for m in IMAGE.finditer(text)} | {m["path"] for m in FIGURE.finditer(text)}
    missing = [
        path
        for path in sorted(paths)
        if "://" not in path and not (page.parent / path.split("#")[0]).resolve().exists()
    ]
    assert not missing, f"{page.relative_to(REPO)} references missing files: {missing}"
