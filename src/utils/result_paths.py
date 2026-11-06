"""Resolve the result root for ordinary runs and figure reproduction."""

import os
from pathlib import Path


def result_root(project_root: Path) -> Path:
    configured = os.environ.get("SEG_RESULTS_ROOT")
    return Path(configured).resolve() if configured else project_root / "results"
