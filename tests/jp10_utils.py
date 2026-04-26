from pathlib import Path

import pytest


JP10_CANDIDATES = [
    Path("canterax/jp10.yaml"),
    Path("canterax/tests/jp10.yaml"),
    Path("jp10.yaml"),
]


def resolve_jp10_path() -> str:
    for path in JP10_CANDIDATES:
        if path.exists():
            return str(path)
    pytest.skip("jp10.yaml is not available in this workspace")


def find_jp10_path() -> str | None:
    for path in JP10_CANDIDATES:
        if path.exists():
            return str(path)
    return None
