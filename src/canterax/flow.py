import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import lineax

from .kinetics import compute_wdot
from .thermo import compute_mixture_props

_compute_wdot = getattr(compute_wdot, "__wrapped__", compute_wdot)
_compute_mixture_props = getattr(compute_mixture_props, "__wrapped__", compute_mixture_props)


@jax.jit
def open_constp_reactor_rhs(t, state, args):
    """Open constant-pressure reactor RHS with mass flow controllers.

    This models a single well-stirred ideal gas reactor at constant pressure P,
    with time-dependent inlet/outlet mass flow rates and fixed inlet state.

    State vector: [T, Y_0, Y_1, ..., Y_{n-1}, m]
      - T: temperature [K]
      - Y: species mass fractions [-]
      - m: total reactor mass [kg]

    Args: (P, mech, mdot_in_ab, Tin, Yin, mdot_out_ab)
      - mdot_*_ab: length-2 array [a, b] defining mdot(t) = a + b * t [kg/s]
        (Use b=0 for constant mass flow.)
      - Tin, Yin: inlet temperature and composition (mass fractions)
    """
    T = state[0]
    Y = state[1:-1]
    m = state[-1]

    P, mech, mdot_in_ab, Tin, Yin, mdot_out_ab = args

    mdot_in = mdot_in_ab[0] + mdot_in_ab[1] * t
    mdot_out = mdot_out_ab[0] + mdot_out_ab[1] * t

    # Mixture properties / chemistry for reactor state
    # wdot: [mol/m3/s], h_mass: [J/kg], cp_mass: [J/kg/K], rho: [kg/m3], h_mol: [J/mol]
    # Use the underlying (non-jitted) functions so the outer jit can compile
    # the full RHS as one fused XLA computation.
    wdot, h_mass, cp_mass, rho, h_mol = _compute_wdot(T, P, Y, mech)

    # Inlet thermodynamic state (use same mechanism, pressure)
    cp_in, h_in, _, _ = _compute_mixture_props(Tin, P, Yin, mech)

    # Species: chemistry + flow mixing
    dYdt_chem = wdot * mech.mol_weights / rho
    dYdt_flow = (mdot_in * (Yin - Y)) / jnp.maximum(m, 1e-300)
    dYdt = dYdt_chem + dYdt_flow

    # Energy: chemistry + enthalpy flow (constant pressure)
    energy_term = jnp.sum(h_mol * wdot)  # [J/m3/s]
    dTdt_chem = -energy_term / (rho * cp_mass)
    dTdt_flow = (mdot_in * (h_in - h_mass)) / (jnp.maximum(m, 1e-300) * cp_mass)
    dTdt = dTdt_chem + dTdt_flow

    # Total mass
    dmdt = mdot_in - mdot_out

    return jnp.concatenate([jnp.array([dTdt]), dYdt, jnp.array([dmdt])])


@eqx.filter_jit
class OpenReactorNet(eqx.Module):
    """Single open constant-pressure reactor with inlet/outlet MFCs."""

    mech: any

    @eqx.filter_jit
    def advance(
        self,
        T0,
        P,
        Y0,
        m0,
        t_end,
        *,
        Tin,
        Yin,
        mdot_in,
        mdot_out,
        rtol=1e-7,
        atol=1e-10,
        solver=None,
        saveat=None,
        max_steps=100000,
        dt0=1e-8,
        stepsize_controller=None,
    ):
        """Advance the open reactor state to time t_end.

        mdot_in / mdot_out:
          - scalar (constant kg/s), or
          - length-2 array [a, b] for mdot(t) = a + b*t
        """
        mdot_in_ab = (
            jnp.array([mdot_in, 0.0]) if jnp.ndim(mdot_in) == 0 else jnp.asarray(mdot_in)
        )
        mdot_out_ab = (
            jnp.array([mdot_out, 0.0]) if jnp.ndim(mdot_out) == 0 else jnp.asarray(mdot_out)
        )

        state0 = jnp.concatenate([jnp.array([T0]), Y0, jnp.array([m0])])
        args = (P, self.mech, mdot_in_ab, Tin, Yin, mdot_out_ab)

        term = diffrax.ODETerm(open_constp_reactor_rhs)
        if solver is None:
            solver = diffrax.Kvaerno5(
                scan_kind="lax",
                root_finder=diffrax.VeryChord(
                    rtol=rtol,
                    atol=atol,
                    kappa=0.5,
                    linear_solver=lineax.LU(),
                ),
            )
        if saveat is None:
            saveat = diffrax.SaveAt(t1=True)
        if stepsize_controller is None:
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_end,
            dt0=dt0,
            y0=state0,
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=max_steps,
        )
        return sol

