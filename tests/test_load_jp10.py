from pathlib import Path

import cantera as ct
import pytest


JP10_CANDIDATES = [
    Path("canterax/jp10.yaml"),
    Path("canterax/tests/jp10.yaml"),
    Path("jp10.yaml"),
]


@pytest.mark.skipif(
    not any(path.exists() for path in JP10_CANDIDATES),
    reason="jp10.yaml is not available in this workspace",
)
def test_load_jp10():
    mech_path = next(path for path in JP10_CANDIDATES if path.exists())
    sol = ct.Solution(str(mech_path))
    assert sol.n_species > 0
    assert sol.n_reactions > 0
    assert sol.species_index("C5H11CO") >= 0
