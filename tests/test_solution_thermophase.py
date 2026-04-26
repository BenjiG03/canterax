import os
import sys

import cantera as ct
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from canterax.solution import Solution
from jp10_utils import find_jp10_path


JP10_PATH = find_jp10_path()


STATE_CASES = [
    (900.0, ct.one_atm, "H2:0.15,O2:0.25,H2O:0.05,N2:0.55"),
    (1650.0, 2.3 * ct.one_atm, "CH4:1.0,O2:2.0,N2:7.52"),
]

SCALAR_PROPS = [
    "cp_mole",
    "cp_mass",
    "cv_mole",
    "cv_mass",
    "enthalpy_mole",
    "enthalpy_mass",
    "int_energy_mole",
    "int_energy_mass",
    "entropy_mole",
    "entropy_mass",
    "gibbs_mole",
    "gibbs_mass",
    "density_mass",
    "density_mole",
    "volume_mass",
    "volume_mole",
    "mean_molecular_weight",
    "viscosity",
    "thermal_conductivity",
]

VECTOR_PROPS = [
    "standard_cp_R",
    "standard_enthalpies_RT",
    "standard_entropies_R",
    "standard_int_energies_RT",
    "standard_gibbs_RT",
    "partial_molar_cp",
    "partial_molar_enthalpies",
    "partial_molar_entropies",
    "partial_molar_int_energies",
    "chemical_potentials",
]


def make_solutions(T, P, X):
    sol_ct = ct.Solution("gri30.yaml")
    sol_jx = Solution("gri30.yaml")
    sol_ct.TPX = T, P, X
    sol_jx.TPX = T, P, X
    return sol_ct, sol_jx


def make_solutions_for_mech(mech, T, P, X):
    sol_ct = ct.Solution(mech)
    sol_jx = Solution(mech)
    sol_ct.TPX = T, P, X
    sol_jx.TPX = T, P, X
    return sol_ct, sol_jx


@pytest.mark.parametrize("T,P,X", STATE_CASES)
def test_scalar_property_parity(T, P, X):
    sol_ct, sol_jx = make_solutions(T, P, X)
    for name in SCALAR_PROPS:
        assert getattr(sol_jx, name) == pytest.approx(getattr(sol_ct, name), rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("T,P,X", STATE_CASES)
def test_vector_property_parity(T, P, X):
    sol_ct, sol_jx = make_solutions(T, P, X)
    for name in VECTOR_PROPS:
        np.testing.assert_allclose(getattr(sol_jx, name), getattr(sol_ct, name), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("basis", ["mass", "molar"])
@pytest.mark.parametrize("T,P,X", STATE_CASES)
def test_basis_alias_parity(T, P, X, basis):
    sol_ct, sol_jx = make_solutions(T, P, X)
    sol_ct.basis = basis
    sol_jx.basis = basis
    for name in ["h", "u", "s", "g", "cp", "cv", "v", "density"]:
        assert getattr(sol_jx, name) == pytest.approx(getattr(sol_ct, name), rel=1e-8, abs=1e-10)


def _state_metrics(sol):
    return {
        "T": sol.T,
        "P": sol.P,
        "Y": np.array(sol.Y),
        "X": np.array(sol.X),
        "h": sol.h,
        "u": sol.u,
        "s": sol.s,
        "density": sol.density,
        "v": sol.v,
    }


def _assert_state_close(sol_jx, sol_ct, rtol=2e-7, atol=1e-9):
    metrics_jx = _state_metrics(sol_jx)
    metrics_ct = _state_metrics(sol_ct)
    assert metrics_jx["T"] == pytest.approx(metrics_ct["T"], rel=rtol, abs=atol)
    assert metrics_jx["P"] == pytest.approx(metrics_ct["P"], rel=rtol, abs=atol)
    assert metrics_jx["h"] == pytest.approx(metrics_ct["h"], rel=rtol, abs=atol)
    assert metrics_jx["u"] == pytest.approx(metrics_ct["u"], rel=rtol, abs=atol)
    assert metrics_jx["s"] == pytest.approx(metrics_ct["s"], rel=rtol, abs=atol)
    assert metrics_jx["density"] == pytest.approx(metrics_ct["density"], rel=rtol, abs=atol)
    assert metrics_jx["v"] == pytest.approx(metrics_ct["v"], rel=rtol, abs=atol)
    np.testing.assert_allclose(metrics_jx["Y"], metrics_ct["Y"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(metrics_jx["X"], metrics_ct["X"], rtol=rtol, atol=atol)


@pytest.mark.parametrize("basis", ["mass", "molar"])
def test_state_setter_parity(basis):
    base_T = 1200.0
    base_P = 1.7 * ct.one_atm
    base_X = "CH4:1.0,O2:2.0,N2:7.52"
    sol_ct, sol_jx = make_solutions(base_T, base_P, base_X)
    sol_ct.basis = basis
    sol_jx.basis = basis

    sol_ct.HPY = sol_ct.HPY
    sol_jx.HPY = sol_jx.HPY
    _assert_state_close(sol_jx, sol_ct)

    sol_ct.UVX = sol_ct.UVX
    sol_jx.UVX = sol_jx.UVX
    _assert_state_close(sol_jx, sol_ct)

    sol_ct.SPY = sol_ct.SPY
    sol_jx.SPY = sol_jx.SPY
    _assert_state_close(sol_jx, sol_ct)

    sol_ct.SVX = sol_ct.SVX
    sol_jx.SVX = sol_jx.SVX
    _assert_state_close(sol_jx, sol_ct)

    sol_ct.TDY = sol_ct.TDY
    sol_jx.TDY = sol_jx.TDY
    _assert_state_close(sol_jx, sol_ct)

    sol_ct.DPX = sol_ct.DPX
    sol_jx.DPX = sol_jx.DPX
    _assert_state_close(sol_jx, sol_ct)


def test_none_state_setter_semantics():
    sol_ct, sol_jx = make_solutions(1100.0, ct.one_atm, "CO:0.3,O2:0.2,N2:0.5")
    sol_ct.HPY = sol_ct.h, None, sol_ct.Y
    sol_jx.HPY = sol_jx.h, None, sol_jx.Y
    _assert_state_close(sol_jx, sol_ct)

    sol_ct.SPY = None, sol_ct.P, sol_ct.Y
    sol_jx.SPY = None, sol_jx.P, sol_jx.Y
    _assert_state_close(sol_jx, sol_ct)


def test_species_and_element_metadata():
    sol_ct, sol_jx = make_solutions(1000.0, ct.one_atm, "H2:0.1,O2:0.2,N2:0.7")
    assert sol_jx.n_elements == sol_ct.n_elements
    assert sol_jx.element_names == sol_ct.element_names
    np.testing.assert_allclose(sol_jx.atomic_weights, sol_ct.atomic_weights)
    np.testing.assert_allclose(sol_jx.molecular_weights, sol_ct.molecular_weights)
    assert sol_jx.species_name(0) == sol_ct.species_name(0)
    assert sol_jx.species_index("N2") == sol_ct.species_index("N2")
    assert sol_jx.element_name(0) == sol_ct.element_name(0)
    assert sol_jx.element_index("O") == sol_ct.element_index("O")
    assert sol_jx.n_atoms("H2O", "H") == sol_ct.n_atoms("H2O", "H")
    assert sol_jx.mass_fraction_dict(1e-12).keys() == sol_ct.mass_fraction_dict(1e-12).keys()
    assert sol_jx.mole_fraction_dict(1e-12).keys() == sol_ct.mole_fraction_dict(1e-12).keys()
    assert sol_jx.thermo_model == sol_ct.thermo_model
    assert sol_jx.phase_of_matter == sol_ct.phase_of_matter
    assert sol_jx.reference_pressure == pytest.approx(sol_ct.reference_pressure)
    assert sol_jx.min_temp == pytest.approx(sol_ct.min_temp)
    assert sol_jx.max_temp == pytest.approx(sol_ct.max_temp)
    assert sol_jx.state_size == sol_ct.state_size


EQUILIBRIUM_CASES = [
    ("gri30.yaml", 1000.0, ct.one_atm, "O2:0.21,N2:0.79", 5e-5),
    ("gri30.yaml", 2000.0, ct.one_atm, "CH4:1.0,O2:2.0,N2:7.52", 5e-5),
    ("h2o2.yaml", 1250.0, ct.one_atm, "H2:2.0,O2:1.0,AR:4.0", 2e-5),
    ("gri30_highT.yaml", 2200.0, 1.5 * ct.one_atm, "CO:1.0,O2:0.5,N2:1.88", 8e-5),
    (JP10_PATH, 1500.0, ct.one_atm, "C10H16:1,O2:14,N2:52.64", 8e-5),
]


@pytest.mark.parametrize("XY", ["TP", "HP"])
@pytest.mark.parametrize("mech,T,P,X,y_tol", EQUILIBRIUM_CASES)
def test_equilibrium_parity(XY, mech, T, P, X, y_tol):
    if mech is None:
        pytest.skip("jp10.yaml is not available in this workspace")
    sol_ct, sol_jx = make_solutions_for_mech(mech, T, P, X)
    sol_ct.equilibrate(XY)
    sol_jx.equilibrate(XY)
    assert sol_jx.T == pytest.approx(sol_ct.T, rel=5e-6, abs=5e-5)
    assert sol_jx.P == pytest.approx(sol_ct.P, rel=5e-8, abs=1e-6)
    np.testing.assert_allclose(sol_jx.Y, sol_ct.Y, rtol=y_tol, atol=1e-8)
