import equinox as eqx
import jax.numpy as jnp
import optimistix as optx

from .constants import ONE_ATM
from .thermo import compute_thermo_state, get_h_RT, get_s_R


@eqx.filter_jit
def _solve_equil_core(y0, gi_const, A_active, b_active, rtol, max_steps):
    n_active_species = A_active.shape[1]
    n_active_elements = A_active.shape[0]

    def kkt_system(y, args):
        s = y[:n_active_species]
        lams = y[n_active_species : n_active_species + n_active_elements]
        log_n_total = y[-1]

        n = jnp.exp(s)
        n_total = jnp.exp(log_n_total)

        eq1 = gi_const + s - log_n_total - (A_active.T @ lams)
        eq2 = (A_active @ n - b_active) / (b_active + 1e-15)
        eq3 = (jnp.sum(n) - n_total) / n_total
        return jnp.concatenate([eq1, eq2, jnp.array([eq3])])

    solver = optx.LevenbergMarquardt(rtol=rtol, atol=rtol * 1e-3)
    return optx.least_squares(
        kkt_system,
        solver,
        y0=y0,
        max_steps=max_steps,
        throw=False,
    )


def _equilibrate_tp_state(mech, T, P, Y0, rtol, max_steps):
    n0 = Y0 / mech.mol_weights
    b = jnp.dot(mech.element_matrix, n0)

    present_elements = b > 1e-15
    species_can_exist = jnp.all(
        jnp.where(mech.element_matrix > 0, present_elements[:, None], True),
        axis=0,
    )

    active_species_idx = jnp.where(species_can_exist)[0]
    active_elements_idx = jnp.where(present_elements)[0]

    if len(active_species_idx) == 0:
        return Y0, None

    A_active = mech.element_matrix[active_elements_idx][:, active_species_idx]
    b_active = b[active_elements_idx]

    h_RT = get_h_RT(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    s_R = get_s_R(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    g_standard_RT = h_RT - s_R
    gi_const = (g_standard_RT + jnp.log(P / ONE_ATM))[active_species_idx]

    n_active0 = n0[active_species_idx]
    ntot0 = jnp.maximum(jnp.sum(n_active0), 1e-10)
    weights = jnp.sqrt(jnp.maximum(n_active0 / ntot0, 1e-6))
    rhs = (gi_const + jnp.log(jnp.maximum(n_active0 / ntot0, 1e-10))) * weights
    design_mat = A_active.T * weights[:, None]
    lams0, _, _, _ = jnp.linalg.lstsq(design_mat, rhs)

    s0 = (A_active.T @ lams0) - gi_const + jnp.log(ntot0)
    y0 = jnp.concatenate([s0, lams0, jnp.array([jnp.log(ntot0)])])
    res = _solve_equil_core(y0, gi_const, A_active, b_active, rtol, max_steps)

    s_equil = res.value[: len(active_species_idx)]
    n_equil_active = jnp.exp(s_equil)
    n_full = jnp.zeros(mech.n_species).at[active_species_idx].set(n_equil_active)
    Y_equil = n_full * mech.mol_weights
    Y_equil = Y_equil / jnp.sum(Y_equil)
    return Y_equil, res


def _basis_enthalpy(state, basis):
    return float(state["h_mass"] if basis == "mass" else state["h_mole"])


def equilibrate(
    sol,
    XY="TP",
    solver="auto",
    rtol=1e-9,
    max_steps=1000,
    max_iter=100,
    estimate_equil=0,
    log_level=0,
):
    """Perform equilibrium calculation with a Cantera-like API subset."""
    if XY is None:
        XY = "TP"
    XY = XY.upper()

    if solver not in {"auto", "element_potential"}:
        raise NotImplementedError(f"Unsupported equilibrium solver: {solver}")
    if estimate_equil not in {0, -1, 1}:
        raise NotImplementedError("estimate_equil values other than -1, 0, and 1 are unsupported.")
    if log_level != 0:
        raise NotImplementedError("log_level != 0 is unsupported.")
    if XY not in {"TP", "HP"}:
        raise NotImplementedError("Supported equilibrium modes are TP and HP.")

    mech = sol.mech
    T0 = sol.T
    P = sol.P
    Y0 = jnp.array(sol.Y)

    if XY == "TP":
        Y_equil, res = _equilibrate_tp_state(mech, T0, P, Y0, rtol, max_steps)
        sol.TPY = T0, P, Y_equil
        return res

    target_h = sol.h
    basis = sol.basis

    def enthalpy_residual(T):
        Y_equil, _ = _equilibrate_tp_state(mech, T, P, Y0, rtol, max_steps)
        state = compute_thermo_state(T, P, Y_equil, mech)
        return _basis_enthalpy(state, basis) - target_h

    lo = max(1.0, 0.5 * sol.min_temp)
    hi = max(sol.max_temp, sol.T)
    f_lo = enthalpy_residual(lo)
    f_hi = enthalpy_residual(hi)

    it = 0
    while f_lo > 0.0 and it < max_iter:
        lo *= 0.5
        f_lo = enthalpy_residual(lo)
        it += 1

    it = 0
    while f_hi < 0.0 and it < max_iter:
        hi *= 1.5
        f_hi = enthalpy_residual(hi)
        it += 1

    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError("Unable to bracket HP equilibrium temperature.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = enthalpy_residual(mid)
        if abs(f_mid) < max(1e-8, rtol * max(1.0, abs(target_h))):
            lo = hi = mid
            break
        if f_mid > 0.0:
            hi = mid
        else:
            lo = mid

    T_equil = 0.5 * (lo + hi)
    Y_equil, res = _equilibrate_tp_state(mech, T_equil, P, Y0, rtol, max_steps)
    sol.TPY = T_equil, P, Y_equil
    return res
