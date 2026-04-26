"""Ideal-gas thermodynamic and transport property routines."""

import jax
import jax.numpy as jnp

from .constants import R_GAS

_TINY = 1e-300


@jax.jit
def get_cp_R(T, nasa_low, nasa_high, T_mid):
    """Compute non-dimensional heat capacity Cp/R for all species."""
    T = jnp.atleast_1d(T)
    mask = T > T_mid
    coeffs = jnp.where(mask[:, None], nasa_high, nasa_low)
    return (
        coeffs[:, 0]
        + coeffs[:, 1] * T
        + coeffs[:, 2] * T**2
        + coeffs[:, 3] * T**3
        + coeffs[:, 4] * T**4
    )


@jax.jit
def get_h_RT(T, nasa_low, nasa_high, T_mid):
    """Compute non-dimensional enthalpy H/RT for all species."""
    T = jnp.atleast_1d(T)
    mask = T > T_mid
    coeffs = jnp.where(mask[:, None], nasa_high, nasa_low)
    return (
        coeffs[:, 0]
        + coeffs[:, 1] * T / 2.0
        + coeffs[:, 2] * T**2 / 3.0
        + coeffs[:, 3] * T**3 / 4.0
        + coeffs[:, 4] * T**4 / 5.0
        + coeffs[:, 5] / T
    )


@jax.jit
def get_s_R(T, nasa_low, nasa_high, T_mid):
    """Compute non-dimensional entropy S/R for all species."""
    T = jnp.atleast_1d(T)
    mask = T > T_mid
    coeffs = jnp.where(mask[:, None], nasa_high, nasa_low)
    return (
        coeffs[:, 0] * jnp.log(T)
        + coeffs[:, 1] * T
        + coeffs[:, 2] * T**2 / 2.0
        + coeffs[:, 3] * T**3 / 3.0
        + coeffs[:, 4] * T**4 / 4.0
        + coeffs[:, 6]
    )


@jax.jit
def mass_to_mole_fractions(Y, mol_weights):
    """Convert species mass fractions to mole fractions."""
    y_mw = Y / mol_weights
    return y_mw / jnp.sum(y_mw)


@jax.jit
def mole_to_mass_fractions(X, mol_weights):
    """Convert species mole fractions to mass fractions."""
    y_unnorm = X * mol_weights
    return y_unnorm / jnp.sum(y_unnorm)


@jax.jit
def mean_molecular_weight(Y, mol_weights):
    """Return the mixture mean molecular weight from mass fractions."""
    return 1.0 / jnp.sum(Y / mol_weights)


@jax.jit
def standard_cp_R(T, mech):
    """Return standard-state species heat capacities normalized by ``R``."""
    return get_cp_R(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)


@jax.jit
def standard_enthalpies_RT(T, mech):
    """Return standard-state species enthalpies normalized by ``RT``."""
    return get_h_RT(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)


@jax.jit
def standard_entropies_R(T, P, mech):
    """Return standard-state species entropies normalized by ``R``."""
    return get_s_R(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid) - jnp.log(
        P / mech.reference_pressure
    )


@jax.jit
def standard_int_energies_RT(T, mech):
    """Return standard-state species internal energies normalized by ``RT``."""
    return standard_enthalpies_RT(T, mech) - 1.0


@jax.jit
def standard_gibbs_RT(T, P, mech):
    """Return standard-state species Gibbs energies normalized by ``RT``."""
    return standard_enthalpies_RT(T, mech) - standard_entropies_R(T, P, mech)


@jax.jit
def partial_molar_cp(T, mech):
    """Return species partial molar heat capacities."""
    return standard_cp_R(T, mech) * R_GAS


@jax.jit
def partial_molar_enthalpies(T, mech):
    """Return species partial molar enthalpies."""
    return standard_enthalpies_RT(T, mech) * R_GAS * T


@jax.jit
def partial_molar_int_energies(T, mech):
    """Return species partial molar internal energies."""
    return standard_int_energies_RT(T, mech) * R_GAS * T


@jax.jit
def chemical_potentials(T, P, Y, mech):
    """Return species chemical potentials for the current mixture state."""
    X = mass_to_mole_fractions(Y, mech.mol_weights)
    log_term = jnp.log(jnp.maximum(X, _TINY))
    return (standard_gibbs_RT(T, P, mech) + log_term) * R_GAS * T


@jax.jit
def partial_molar_entropies(T, P, Y, mech):
    """Return species partial molar entropies for the current mixture state."""
    X = mass_to_mole_fractions(Y, mech.mol_weights)
    log_term = jnp.log(jnp.maximum(X, _TINY))
    return standard_entropies_R(T, P, mech) * R_GAS - R_GAS * log_term


@jax.jit
def compute_thermo_state(T, P, Y, mech):
    """Compute mixture and species thermodynamic properties for an ideal gas."""
    cp_R = standard_cp_R(T, mech)
    h_RT = standard_enthalpies_RT(T, mech)
    s_R0 = standard_entropies_R(T, P, mech)
    u_RT = h_RT - 1.0
    g_RT0 = h_RT - s_R0

    cp_molar_partial = cp_R * R_GAS
    h_partial = h_RT * R_GAS * T
    u_partial = u_RT * R_GAS * T

    X = mass_to_mole_fractions(Y, mech.mol_weights)
    mw_mix = mean_molecular_weight(Y, mech.mol_weights)

    log_mix = jnp.log(jnp.maximum(X, _TINY))
    s_partial = s_R0 * R_GAS - R_GAS * log_mix
    mu = (g_RT0 + log_mix) * R_GAS * T

    cp_mole = jnp.sum(X * cp_molar_partial)
    h_mole = jnp.sum(X * h_partial)
    u_mole = jnp.sum(X * u_partial)
    s_mole = jnp.sum(X * s_partial)
    g_mole = jnp.sum(X * mu)

    cp_mass = cp_mole / mw_mix
    cv_mole = cp_mole - R_GAS
    cv_mass = cv_mole / mw_mix
    h_mass = h_mole / mw_mix
    u_mass = u_mole / mw_mix
    s_mass = s_mole / mw_mix
    g_mass = g_mole / mw_mix

    density_mole = P / (R_GAS * T)
    density_mass = density_mole * mw_mix
    volume_mole = 1.0 / density_mole
    volume_mass = 1.0 / density_mass

    return {
        "X": X,
        "mw_mix": mw_mix,
        "cp_mole": cp_mole,
        "cp_mass": cp_mass,
        "cv_mole": cv_mole,
        "cv_mass": cv_mass,
        "h_mole": h_mole,
        "h_mass": h_mass,
        "u_mole": u_mole,
        "u_mass": u_mass,
        "s_mole": s_mole,
        "s_mass": s_mass,
        "g_mole": g_mole,
        "g_mass": g_mass,
        "density_mole": density_mole,
        "density_mass": density_mass,
        "volume_mole": volume_mole,
        "volume_mass": volume_mass,
        "partial_molar_cp": cp_molar_partial,
        "partial_molar_enthalpies": h_partial,
        "partial_molar_int_energies": u_partial,
        "partial_molar_entropies": s_partial,
        "chemical_potentials": mu,
        "standard_cp_R": cp_R,
        "standard_enthalpies_RT": h_RT,
        "standard_entropies_R": s_R0,
        "standard_int_energies_RT": u_RT,
        "standard_gibbs_RT": g_RT0,
    }


@jax.jit
def compute_mixture_props(T, P, Y, mech):
    """Compute legacy mixture thermodynamic properties.

    Returns:
        cp_mass: mixture mass-weighted heat capacity (J/kg/K)
        h_mass: mixture mass-weighted enthalpy (J/kg)
        rho: mixture density (kg/m^3)
        h_mol: partial molar enthalpies (J/kmol)
    """
    state = compute_thermo_state(T, P, Y, mech)
    return (
        state["cp_mass"],
        state["h_mass"],
        state["density_mass"],
        state["partial_molar_enthalpies"],
    )


@jax.jit
def species_viscosities(T, mech):
    """Evaluate fitted pure-species viscosities at temperature ``T``."""
    logt = jnp.log(T)
    poly = jnp.stack([jnp.ones_like(logt), logt, logt**2, logt**3, logt**4])
    sqvisc = jnp.sqrt(jnp.sqrt(T)) * (mech.viscosity_poly @ poly)
    return sqvisc * sqvisc


@jax.jit
def species_thermal_conductivities(T, mech):
    """Evaluate fitted pure-species thermal conductivities at ``T``."""
    logt = jnp.log(T)
    poly = jnp.stack([jnp.ones_like(logt), logt, logt**2, logt**3, logt**4])
    return jnp.sqrt(T) * (mech.conductivity_poly @ poly)


@jax.jit
def mixture_viscosity(T, Y, mech):
    """Compute mixture viscosity using Wilke-style mixing."""
    X = mass_to_mole_fractions(Y, mech.mol_weights)
    X = jnp.maximum(X, 1e-20)
    X = X / jnp.sum(X)
    mu = species_viscosities(T, mech)
    mw = mech.mol_weights

    mu_ratio = mu[:, None] / mu[None, :]
    mw_ratio = mw[None, :] / mw[:, None]
    factor1 = 1.0 + jnp.sqrt(mu_ratio) * jnp.power(mw_ratio, 0.25)
    denom = jnp.sqrt(8.0 * (1.0 + mw[:, None] / mw[None, :]))
    phi = (factor1 * factor1) / denom
    visc_denom = phi @ X
    return jnp.sum(X * mu / visc_denom)


@jax.jit
def mixture_thermal_conductivity(T, Y, mech):
    """Compute mixture thermal conductivity from species fits and mixing."""
    X = mass_to_mole_fractions(Y, mech.mol_weights)
    X = jnp.maximum(X, 1e-20)
    X = X / jnp.sum(X)
    lam = species_thermal_conductivities(T, mech)
    sum1 = jnp.sum(X * lam)
    sum2 = jnp.sum(X / lam)
    return 0.5 * (sum1 + 1.0 / sum2)
