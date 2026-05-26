"""Structural smoke test for the reviewer-facing E2E demo notebook.

We don't execute the notebook in CI — it needs Pangeo streaming and several
minutes of wall clock. Instead, we verify the file is valid JSON in nbformat
4, has the expected anchor cells, and imports from the modules a reviewer
would care about.
"""

from __future__ import annotations

import json
from pathlib import Path

_NB_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "e2e_demo.ipynb"


def _load() -> dict:
    return json.loads(_NB_PATH.read_text())


def test_notebook_is_nbformat_4() -> None:
    nb = _load()
    assert nb["nbformat"] == 4
    assert "cells" in nb
    assert len(nb["cells"]) >= 8


def test_first_cell_is_title_markdown() -> None:
    nb = _load()
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown"
    source = "".join(first["source"])
    assert "Atmospheric Rivers Pipeline" in source


def _all_source() -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in _load()["cells"] if cell["cell_type"] == "code"
    )


def test_imports_anchor_to_src_modules() -> None:
    src = _all_source()
    for needle in (
        "from src import config",
        "physics_pipeline",
        "ar_detection",
        "event_post",
        "topography",
    ):
        assert needle in src, f"missing import or reference: {needle}"


def test_uses_one_month_demo_window() -> None:
    # The demo is constrained to 1998-01-* so wall clock stays under ~10 min.
    # If someone extends the window inadvertently, this guard fires.
    src = _all_source()
    assert "1998-01-01" in src
    assert "1998-01-31" in src


def test_notebook_has_kernelspec() -> None:
    nb = _load()
    ks = nb["metadata"]["kernelspec"]
    assert ks["language"] == "python"
