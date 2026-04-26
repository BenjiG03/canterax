"""Cantera-like solution wrapper built on top of Canterax primitives."""

import numpy as np
import jax.numpy as jnp

from .constants import ONE_ATM, R_GAS
from .kinetics import compute_wdot
from .loader import load_mechanism
from .thermo import (
    chemical_potentials,
    compute_thermo_state,
    mass_to_mole_fractions,
    mean_molecular_weight,
    mixture_thermal_conductivity,
    mixture_viscosity,
    mole_to_mass_fractions,
    partial_molar_cp,
    partial_molar_enthalpies,
    partial_molar_entropies,
    partial_molar_int_energies,
    standard_cp_R,
    standard_enthalpies_RT,
    standard_entropies_R,
    standard_gibbs_RT,
    standard_int_energies_RT,
)


class Solution:
    """A Cantera-like ideal-gas ThermoPhase wrapper for Canterax."""

    def __init__(self, yaml_file: str):
        """Create a solution object from a Cantera-compatible YAML mechanism."""
        self.mech = load_mechanism(yaml_file)
        self.n_species = self.mech.n_species
        self.n_reactions = self.mech.n_reactions
        self.species_names = list(self.mech.species_names)

        self._basis = "mass"
        self._T = 300.0
        self._P = ONE_ATM
        self._Y = jnp.zeros(self.n_species)
        if "N2" in self.species_names:
            self._Y = self._Y.at[self.species_names.index("N2")].set(1.0)
        else:
            self._Y = self._Y.at[0].set(1.0)

    def _thermo_state(self):
        return compute_thermo_state(self._T, self._P, self._Y, self.mech)

    def _parse_composition(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            res = np.zeros(self.n_species, dtype=float)
            parts = [p.strip() for p in value.split(",") if p.strip()]
            for p in parts:
                spec, val = p.split(":")
                idx = self.species_index(spec.strip())
                res[idx] = float(val)
            return jnp.array(res)
        return jnp.array(value, dtype=float)

    def _normalize_mass(self, value):
        Y = self._parse_composition(value)
        if Y is None:
            return self._Y
        total = jnp.sum(Y)
        if float(total) <= 0.0:
            raise ValueError("Mass fractions must sum to a positive value.")
        return Y / total

    def _normalize_mole(self, value):
        X = self._parse_composition(value)
        if X is None:
            return self.X
        total = jnp.sum(X)
        if float(total) <= 0.0:
            raise ValueError("Mole fractions must sum to a positive value.")
        return X / total

    def _basis_value(self, mass_value, mole_value):
        return float(mass_value if self._basis == "mass" else mole_value)

    def _basis_density(self, state):
        return self._basis_value(state["density_mass"], state["density_mole"])

    def _basis_volume(self, state):
        return self._basis_value(state["volume_mass"], state["volume_mole"])

    def _basis_enthalpy(self, state):
        return self._basis_value(state["h_mass"], state["h_mole"])

    def _basis_internal_energy(self, state):
        return self._basis_value(state["u_mass"], state["u_mole"])

    def _basis_entropy(self, state):
        return self._basis_value(state["s_mass"], state["s_mole"])

    def _basis_gibbs(self, state):
        return self._basis_value(state["g_mass"], state["g_mole"])

    def _basis_cp(self, state):
        return self._basis_value(state["cp_mass"], state["cp_mole"])

    def _basis_cv(self, state):
        return self._basis_value(state["cv_mass"], state["cv_mole"])

    def _density_to_pressure(self, density, T):
        mw = float(self.mean_molecular_weight)
        if self._basis == "mass":
            return density * R_GAS * T / mw
        return density * R_GAS * T

    def _pressure_from_volume(self, volume, T):
        mw = float(self.mean_molecular_weight)
        if self._basis == "mass":
            return R_GAS * T / (mw * volume)
        return R_GAS * T / volume

    def _solve_temperature(self, func, target):
        lo = max(1.0, 0.5 * self.min_temp)
        hi = max(self.max_temp, self.T)

        f_lo = func(lo) - target
        f_hi = func(hi) - target

        expand = 0
        while f_lo > 0.0 and lo > 1e-6 and expand < 40:
            lo *= 0.5
            f_lo = func(lo) - target
            expand += 1

        expand = 0
        while f_hi < 0.0 and expand < 40:
            hi *= 1.5
            f_hi = func(hi) - target
            expand += 1

        if f_lo > 0.0 or f_hi < 0.0:
            raise ValueError("Unable to bracket temperature for requested state.")

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = func(mid) - target
            if abs(f_mid) < 1e-11 * max(1.0, abs(target)):
                return mid
            if f_mid > 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def _set_state_TP(self, T, P):
        self._T = float(T)
        self._P = float(P)

    def _set_state_TPY(self, T=None, P=None, Y=None):
        if Y is not None:
            self._Y = self._normalize_mass(Y)
        self._T = float(self._T if T is None else T)
        self._P = float(self._P if P is None else P)

    def _set_state_TPX(self, T=None, P=None, X=None):
        if X is not None:
            self._Y = mole_to_mass_fractions(self._normalize_mole(X), self.mech.mol_weights)
        self._T = float(self._T if T is None else T)
        self._P = float(self._P if P is None else P)

    def _set_state_HP(self, H=None, P=None):
        target_P = self.P if P is None else float(P)
        if H is None:
            H = self.h

        def objective(T):
            state = compute_thermo_state(T, target_P, self._Y, self.mech)
            return self._basis_enthalpy(state)

        T = self._solve_temperature(objective, float(H))
        self._set_state_TP(T, target_P)

    def _set_state_UV(self, U=None, V=None):
        if U is None:
            U = self.u
        if V is None:
            V = self.v

        def objective(T):
            state = compute_thermo_state(T, self._P, self._Y, self.mech)
            return self._basis_internal_energy(state)

        T = self._solve_temperature(objective, float(U))
        P = self._pressure_from_volume(float(V), T)
        self._set_state_TP(T, P)

    def _set_state_SP(self, S=None, P=None):
        target_P = self.P if P is None else float(P)
        if S is None:
            S = self.s

        def objective(T):
            state = compute_thermo_state(T, target_P, self._Y, self.mech)
            return self._basis_entropy(state)

        T = self._solve_temperature(objective, float(S))
        self._set_state_TP(T, target_P)

    def _set_state_SV(self, S=None, V=None):
        if S is None:
            S = self.s
        if V is None:
            V = self.v
        target_V = float(V)

        def objective(T):
            P = self._pressure_from_volume(target_V, T)
            state = compute_thermo_state(T, P, self._Y, self.mech)
            return self._basis_entropy(state)

        T = self._solve_temperature(objective, float(S))
        P = self._pressure_from_volume(target_V, T)
        self._set_state_TP(T, P)

    def _set_state_TD(self, T=None, D=None):
        target_T = self.T if T is None else float(T)
        target_D = self.density if D is None else float(D)
        P = self._density_to_pressure(target_D, target_T)
        self._set_state_TP(target_T, P)

    def _set_state_DP(self, D=None, P=None):
        target_D = self.density if D is None else float(D)
        target_P = self.P if P is None else float(P)
        mw = float(self.mean_molecular_weight)
        if self._basis == "mass":
            T = target_P * mw / (target_D * R_GAS)
        else:
            T = target_P / (target_D * R_GAS)
        self._set_state_TP(T, target_P)

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, value):
        value = value.lower()
        if value not in {"mass", "molar"}:
            raise ValueError("basis must be 'mass' or 'molar'")
        self._basis = value

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = float(value)

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        self._P = float(value)

    @property
    def Y(self):
        return np.array(self._Y)

    @Y.setter
    def Y(self, value):
        self._Y = self._normalize_mass(value)

    @property
    def X(self):
        return np.array(mass_to_mole_fractions(self._Y, self.mech.mol_weights))

    @X.setter
    def X(self, value):
        X = self._normalize_mole(value)
        self._Y = mole_to_mass_fractions(X, self.mech.mol_weights)

    @property
    def TP(self):
        return self.T, self.P

    @TP.setter
    def TP(self, value):
        T, P = value
        self._set_state_TP(self.T if T is None else T, self.P if P is None else P)

    @property
    def TPY(self):
        return self.T, self.P, self.Y

    @TPY.setter
    def TPY(self, value):
        T, P, Y = value
        self._set_state_TPY(T, P, Y)

    @property
    def TPX(self):
        return self.T, self.P, self.X

    @TPX.setter
    def TPX(self, value):
        T, P, X = value
        self._set_state_TPX(T, P, X)

    @property
    def HP(self):
        return self.h, self.P

    @HP.setter
    def HP(self, value):
        H, P = value
        self._set_state_HP(H, P)

    @property
    def HPY(self):
        return self.h, self.P, self.Y

    @HPY.setter
    def HPY(self, value):
        H, P, Y = value
        if Y is not None:
            self.Y = Y
        self._set_state_HP(H, P)

    @property
    def HPX(self):
        return self.h, self.P, self.X

    @HPX.setter
    def HPX(self, value):
        H, P, X = value
        if X is not None:
            self.X = X
        self._set_state_HP(H, P)

    @property
    def UV(self):
        return self.u, self.v

    @UV.setter
    def UV(self, value):
        U, V = value
        self._set_state_UV(U, V)

    @property
    def UVY(self):
        return self.u, self.v, self.Y

    @UVY.setter
    def UVY(self, value):
        U, V, Y = value
        if Y is not None:
            self.Y = Y
        self._set_state_UV(U, V)

    @property
    def UVX(self):
        return self.u, self.v, self.X

    @UVX.setter
    def UVX(self, value):
        U, V, X = value
        if X is not None:
            self.X = X
        self._set_state_UV(U, V)

    @property
    def SP(self):
        return self.s, self.P

    @SP.setter
    def SP(self, value):
        S, P = value
        self._set_state_SP(S, P)

    @property
    def SPY(self):
        return self.s, self.P, self.Y

    @SPY.setter
    def SPY(self, value):
        S, P, Y = value
        if Y is not None:
            self.Y = Y
        self._set_state_SP(S, P)

    @property
    def SPX(self):
        return self.s, self.P, self.X

    @SPX.setter
    def SPX(self, value):
        S, P, X = value
        if X is not None:
            self.X = X
        self._set_state_SP(S, P)

    @property
    def SV(self):
        return self.s, self.v

    @SV.setter
    def SV(self, value):
        S, V = value
        self._set_state_SV(S, V)

    @property
    def SVY(self):
        return self.s, self.v, self.Y

    @SVY.setter
    def SVY(self, value):
        S, V, Y = value
        if Y is not None:
            self.Y = Y
        self._set_state_SV(S, V)

    @property
    def SVX(self):
        return self.s, self.v, self.X

    @SVX.setter
    def SVX(self, value):
        S, V, X = value
        if X is not None:
            self.X = X
        self._set_state_SV(S, V)

    @property
    def TD(self):
        return self.T, self.density

    @TD.setter
    def TD(self, value):
        T, D = value
        self._set_state_TD(T, D)

    @property
    def TDY(self):
        return self.T, self.density, self.Y

    @TDY.setter
    def TDY(self, value):
        T, D, Y = value
        if Y is not None:
            self.Y = Y
        self._set_state_TD(T, D)

    @property
    def TDX(self):
        return self.T, self.density, self.X

    @TDX.setter
    def TDX(self, value):
        T, D, X = value
        if X is not None:
            self.X = X
        self._set_state_TD(T, D)

    @property
    def DP(self):
        return self.density, self.P

    @DP.setter
    def DP(self, value):
        D, P = value
        self._set_state_DP(D, P)

    @property
    def DPY(self):
        return self.density, self.P, self.Y

    @DPY.setter
    def DPY(self, value):
        D, P, Y = value
        if Y is not None:
            self.Y = Y
        self._set_state_DP(D, P)

    @property
    def DPX(self):
        return self.density, self.P, self.X

    @DPX.setter
    def DPX(self, value):
        D, P, X = value
        if X is not None:
            self.X = X
        self._set_state_DP(D, P)

    def set_TPY(self, T, P, Y):
        self.TPY = T, P, Y

    def set_TPX(self, T, P, X):
        self.TPX = T, P, X

    @property
    def thermo_model(self):
        return self.mech.thermo_model

    @property
    def phase_of_matter(self):
        return self.mech.phase_of_matter

    @property
    def is_compressible(self):
        return True

    @property
    def is_pure(self):
        return False

    @property
    def reference_pressure(self):
        return self.mech.reference_pressure

    @property
    def min_temp(self):
        return self.mech.min_temp

    @property
    def max_temp(self):
        return self.mech.max_temp

    @property
    def state_size(self):
        return self.n_species + 2

    @property
    def n_elements(self):
        return self.mech.n_elements

    @property
    def element_names(self):
        return list(self.mech.element_names)

    @property
    def atomic_weights(self):
        return np.array(self.mech.atomic_weights)

    @property
    def molecular_weights(self):
        return np.array(self.mech.mol_weights)

    @property
    def mean_molecular_weight(self):
        return float(mean_molecular_weight(self._Y, self.mech.mol_weights))

    def species_name(self, k: int):
        """Return the species name for index ``k``."""
        return self.species_names[k]

    def species_index(self, species):
        """Return the species index for a name or integer-like identifier."""
        if isinstance(species, (int, np.integer)):
            return int(species)
        return self.species_names.index(species)

    def element_name(self, m: int):
        """Return the element name for index ``m``."""
        return self.element_names[m]

    def element_index(self, element):
        """Return the element index for a name or integer-like identifier."""
        if isinstance(element, (int, np.integer)):
            return int(element)
        return self.element_names.index(element)

    def n_atoms(self, species, element):
        """Return the number of atoms of ``element`` in ``species``."""
        return int(self.mech.element_matrix[self.element_index(element), self.species_index(species)])

    def mass_fraction_dict(self, threshold: float = 0.0):
        """Return species mass fractions above ``threshold`` as a dictionary."""
        return {name: float(val) for name, val in zip(self.species_names, self.Y) if float(val) > threshold}

    def mole_fraction_dict(self, threshold: float = 0.0):
        """Return species mole fractions above ``threshold`` as a dictionary."""
        return {name: float(val) for name, val in zip(self.species_names, self.X) if float(val) > threshold}

    @property
    def standard_cp_R(self):
        return np.array(standard_cp_R(self.T, self.mech))

    @property
    def standard_enthalpies_RT(self):
        return np.array(standard_enthalpies_RT(self.T, self.mech))

    @property
    def standard_entropies_R(self):
        return np.array(standard_entropies_R(self.T, self.P, self.mech))

    @property
    def standard_int_energies_RT(self):
        return np.array(standard_int_energies_RT(self.T, self.mech))

    @property
    def standard_gibbs_RT(self):
        return np.array(standard_gibbs_RT(self.T, self.P, self.mech))

    @property
    def partial_molar_cp(self):
        return np.array(partial_molar_cp(self.T, self.mech))

    @property
    def partial_molar_enthalpies(self):
        return np.array(partial_molar_enthalpies(self.T, self.mech))

    @property
    def partial_molar_int_energies(self):
        return np.array(partial_molar_int_energies(self.T, self.mech))

    @property
    def partial_molar_entropies(self):
        return np.array(partial_molar_entropies(self.T, self.P, self._Y, self.mech))

    @property
    def chemical_potentials(self):
        return np.array(chemical_potentials(self.T, self.P, self._Y, self.mech))

    @property
    def cp_mole(self):
        return float(self._thermo_state()["cp_mole"])

    @property
    def cp_mass(self):
        return float(self._thermo_state()["cp_mass"])

    @property
    def cv_mole(self):
        return float(self._thermo_state()["cv_mole"])

    @property
    def cv_mass(self):
        return float(self._thermo_state()["cv_mass"])

    @property
    def enthalpy_mole(self):
        return float(self._thermo_state()["h_mole"])

    @property
    def enthalpy_mass(self):
        return float(self._thermo_state()["h_mass"])

    @property
    def int_energy_mole(self):
        return float(self._thermo_state()["u_mole"])

    @property
    def int_energy_mass(self):
        return float(self._thermo_state()["u_mass"])

    @property
    def entropy_mole(self):
        return float(self._thermo_state()["s_mole"])

    @property
    def entropy_mass(self):
        return float(self._thermo_state()["s_mass"])

    @property
    def gibbs_mole(self):
        return float(self._thermo_state()["g_mole"])

    @property
    def gibbs_mass(self):
        return float(self._thermo_state()["g_mass"])

    @property
    def density(self):
        return self._basis_density(self._thermo_state())

    @property
    def density_mass(self):
        return float(self._thermo_state()["density_mass"])

    @property
    def density_mole(self):
        return float(self._thermo_state()["density_mole"])

    @property
    def volume_mass(self):
        return float(self._thermo_state()["volume_mass"])

    @property
    def volume_mole(self):
        return float(self._thermo_state()["volume_mole"])

    @property
    def h(self):
        state = self._thermo_state()
        return self._basis_enthalpy(state)

    @property
    def u(self):
        state = self._thermo_state()
        return self._basis_internal_energy(state)

    @property
    def s(self):
        state = self._thermo_state()
        return self._basis_entropy(state)

    @property
    def g(self):
        state = self._thermo_state()
        return self._basis_gibbs(state)

    @property
    def cp(self):
        state = self._thermo_state()
        return self._basis_cp(state)

    @property
    def cv(self):
        state = self._thermo_state()
        return self._basis_cv(state)

    @property
    def v(self):
        state = self._thermo_state()
        return self._basis_volume(state)

    @property
    def viscosity(self):
        return float(mixture_viscosity(self.T, self._Y, self.mech))

    @property
    def thermal_conductivity(self):
        return float(mixture_thermal_conductivity(self.T, self._Y, self.mech))

    @property
    def transport_model(self):
        return self.mech.transport_model

    @property
    def net_production_rates(self):
        wdot, _, _, _, _ = compute_wdot(self.T, self.P, self._Y, self.mech)
        return np.array(wdot)

    def equilibrate(
        self,
        XY="TP",
        solver="auto",
        rtol=1e-9,
        max_steps=1000,
        max_iter=100,
        estimate_equil=0,
        log_level=0,
    ):
        """Mutate the state to equilibrium using the requested constraint pair."""
        from .equilibrate import equilibrate as eq_func

        return eq_func(
            self,
            XY=XY,
            solver=solver,
            rtol=rtol,
            max_steps=max_steps,
            max_iter=max_iter,
            estimate_equil=estimate_equil,
            log_level=log_level,
        )
