"""Equilibrium solvers exposed through the ``Solution.equilibrate`` API."""

from functools import partial

import equinox as eqx
import jax
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
    """Solve fixed-``T``/``P`` equilibrium for a starting mass-fraction state."""
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


def _equilibrate_tp_state_fixed_shape(mech, T, P, Y0, rtol, max_steps):
    """Fixed-shape sibling of ``_equilibrate_tp_state`` for ``jax.jit`` callers.

    ``_equilibrate_tp_state`` prunes species/elements the mixture cannot reach
    via dynamic-length boolean indexing (``jnp.where(...)`` with no ``size=``),
    which is exactly the pattern ``jax.jit`` cannot trace. This sibling skips
    the pruning and uses every species/element in ``mech`` unconditionally,
    trading generality for jittability. Valid whenever the unburned mixture
    already spans every element the mechanism's species draw on (true for any
    complete combustion mechanism fed a fuel+air mixture, e.g. a JP-10/air
    mechanism over C/H/O/N/Ar) -- callers with a genuinely sparse mixture
    (an element entirely absent from ``Y0``) need ``_equilibrate_tp_state``.
    """
    n0 = Y0 / mech.mol_weights
    b = mech.element_matrix @ n0

    h_RT = get_h_RT(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    s_R = get_s_R(T, mech.nasa_low, mech.nasa_high, mech.nasa_T_mid)
    gi_const = h_RT - s_R + jnp.log(P / ONE_ATM)

    ntot0 = jnp.maximum(jnp.sum(n0), 1e-10)
    weights = jnp.sqrt(jnp.maximum(n0 / ntot0, 1e-6))
    rhs = (gi_const + jnp.log(jnp.maximum(n0 / ntot0, 1e-10))) * weights
    design_mat = mech.element_matrix.T * weights[:, None]
    lams0, _, _, _ = jnp.linalg.lstsq(design_mat, rhs)

    s0 = (mech.element_matrix.T @ lams0) - gi_const + jnp.log(ntot0)
    y0 = jnp.concatenate([s0, lams0, jnp.array([jnp.log(ntot0)])])
    res = _solve_equil_core(y0, gi_const, mech.element_matrix, b, rtol, max_steps)

    Y_equil = jnp.exp(res.value[: mech.n_species]) * mech.mol_weights
    Y_equil = Y_equil / jnp.sum(Y_equil)
    return Y_equil, res


@partial(jax.jit, static_argnames=("max_steps", "max_bisect_iter"))
def equilibrate_hp_fixed_shape(mech, T0, P, Y0, target_h, rtol=1e-9, max_steps=1000,
                               max_bisect_iter=60):
    """Fused, jittable HP (constant-enthalpy/pressure) equilibrium solve.

    ``Solution.equilibrate("HP")`` is a Cantera-compatible convenience API:
    every property access it touches (``sol.h``, ``compute_thermo_state``, the
    Python-level bisection over T) dispatches as its own eager XLA op, since
    that flexibility (arbitrary constraint pairs, dynamic active-species
    pruning) cannot be captured in one static jaxpr. That is fine for
    interactive use but costs ~200 separately-traced-and-compiled kernels for
    a single HP solve -- prohibitive for a setup-time call inside a
    JAX-heavy build path (jax_dmrj's flameholder ignited-branch seed is the
    motivating case; see its call site).

    This is the fused alternative: the whole bisection is one ``lax.fori_loop``
    (a FIXED trip count, unlike ``equilibrate``'s early-exit ``while``) around
    ``_equilibrate_tp_state_fixed_shape`` (no dynamic-shape species pruning,
    see its docstring), so the entire solve traces and compiles as ONE XLA
    program. Same "all species/elements active" precondition as
    ``_equilibrate_tp_state_fixed_shape``; same accuracy trade as fixing the
    bisection trip count instead of stopping early once converged (bisection
    error after ``max_bisect_iter`` steps is ``(hi0 - lo0) / 2**max_bisect_iter``,
    below 1e-9 K for the default 60 steps over any physical mechanism's
    min/max NASA-poly temperature range).

    Parameters
    ----------
    mech : MechData
        Canterax mechanism (supplies ``element_matrix``, NASA coefficients,
        ``min_temp``/``max_temp``).
    T0, P, Y0 : float, float, ndarray
        Unburned mixture temperature [K], pressure [Pa], mass fractions.
    target_h : float
        Target mixture-mass-basis specific enthalpy [J/kg] to match (the
        unburned mixture's enthalpy, for a constant-enthalpy/pressure solve).
    rtol : float
        Relative tolerance passed to the inner element-potential KKT solve.
    max_steps : int
        Max iterations of the inner Levenberg-Marquardt KKT solve.
    max_bisect_iter : int
        Fixed bisection trip count for the outer temperature search.

    Returns
    -------
    T_equil, Y_equil : float, ndarray
        Equilibrium product temperature [K] and mass fractions.
    """
    def enthalpy_residual(T):
        Y_equil, _ = _equilibrate_tp_state_fixed_shape(mech, T, P, Y0, rtol, max_steps)
        return compute_thermo_state(T, P, Y_equil, mech)["h_mass"] - target_h

    lo0 = jnp.maximum(1.0, 0.5 * mech.min_temp)
    hi0 = jnp.maximum(mech.max_temp, T0)

    def bisect_step(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        f_mid = enthalpy_residual(mid)
        lo = jnp.where(f_mid > 0.0, lo, mid)
        hi = jnp.where(f_mid > 0.0, mid, hi)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, max_bisect_iter, bisect_step, (lo0, hi0))
    T_equil = 0.5 * (lo + hi)
    Y_equil, _ = _equilibrate_tp_state_fixed_shape(mech, T_equil, P, Y0, rtol, max_steps)
    return T_equil, Y_equil


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
    """Perform ``TP`` or ``HP`` equilibrium with a Cantera-like API subset."""
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
