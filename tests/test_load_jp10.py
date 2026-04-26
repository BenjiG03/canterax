import cantera as ct
import pytest
from jp10_utils import resolve_jp10_path
from jp10_utils import find_jp10_path


@pytest.mark.skipif(find_jp10_path() is None, reason="jp10.yaml is not available in this workspace")
def test_load_jp10():
    sol = ct.Solution(resolve_jp10_path())
    assert sol.n_species > 0
    assert sol.n_reactions > 0
    assert sol.species_index("C5H11CO") >= 0
